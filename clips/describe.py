"""Which subject: pick the one that matches a description, using a local vision model in Ollama.

`--subject-desc "the guy in the red jacket"` (with `--subject`, default auto = people first):
the detector and tracker run as usual, then the most prominent candidate tracks (up to
CANDIDATES) are each shown to a vision-language model running in Ollama on this machine as
a close-up of their best detection, and asked how well it matches the description (0-10).
The best match (>= MIN_MATCH) becomes the subject; look-alike tracks still extend it after
occlusions. If Ollama is not running, the model is missing or nothing matches, a warning is
printed and the most prominent candidate is used, exactly as without a description.

The model only sees a handful of small images once per clip, so its speed does not matter;
the answer is cached next to the clip's detections. Ollama is local and free (no paid API);
`OLLAMA_HOST` sets where it listens (default http://127.0.0.1:11434).
"""
from __future__ import annotations

import base64
import json
import math
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from . import reframe as RF
from . import subject as S
from .ffmpeg import ClipsError, ffmpeg_exe

CANDIDATES = 6          # tracks shown to the model
MIN_MATCH = 5           # 0-10 score the best candidate needs
CROP = 384              # px, square close-up per candidate
FLAT_W = 1280           # flat frames are read at this width for cropping
TIMEOUT = 180           # s per request (first call loads the model)
PROMPT = ("Look only at the subject in the bright middle of the picture (the darkened edges "
          "are just context). Is it {desc}? Rate how well it matches from 0 (clearly not) to "
          "10 (clearly yes). Answer as JSON: {{\"match\": <0-10>}}")
DIM = 0.35              # brightness kept outside the detection box
FORMAT = {"type": "object", "properties": {"match": {"type": "integer"}}, "required": ["match"]}


def host() -> str:
    h = os.environ.get("OLLAMA_HOST") or "127.0.0.1:11434"
    if h.startswith("0.0.0.0"):
        h = "127.0.0.1" + h[len("0.0.0.0"):]
    return (h if "://" in h else "http://" + h).rstrip("/")


# ---------------------------------------------------------------- close-ups
def best_sample(t: S.Track) -> Tuple[int, S.Det]:
    """The track's clearest view: highest score x area."""
    k = max(t.samples, key=lambda k: t.samples[k].score * t.samples[k].w * t.samples[k].h)
    return k, t.samples[k]


def view_crop(eq: np.ndarray, d: S.Det, size: int = CROP) -> np.ndarray:
    """Perspective close-up of a 360 detection from an equirect RGB frame."""
    h, w = eq.shape[:2]
    fov = float(np.clip(1.5 * d.size, 25.0, 120.0))
    lon, lat = S.ray_angles(S.rotate(S.tile_rays(size, fov), d.x, d.y))
    col = np.clip(((lon + 180.0) / 360.0 * w).astype(int), 0, w - 1)
    row = np.clip(((90.0 - lat) / 180.0 * h).astype(int), 0, h - 1)
    t = math.tan(math.radians(fov) / 2)
    hw = math.tan(math.radians(min(d.w, 170) / 2)) / t * size / 2
    hh = math.tan(math.radians(min(d.h, 170) / 2)) / t * size / 2
    c = size / 2
    return dim_outside(eq[row, col], (c - hw, c - hh, c + hw, c + hh))


def flat_crop(img: np.ndarray, d: S.Det, margin: float = 0.25) -> np.ndarray:
    """Box of a flat detection (fractions) plus a margin, clamped to the frame."""
    h, w = img.shape[:2]
    bw, bh = d.w * (1 + 2 * margin), d.h * (1 + 2 * margin)
    x0, x1 = max(0, round((d.x - bw / 2) * w)), min(w, round((d.x + bw / 2) * w))
    y0, y1 = max(0, round((d.y - bh / 2) * h)), min(h, round((d.y + bh / 2) * h))
    crop = img[y0:max(y1, y0 + 2), x0:max(x1, x0 + 2)]
    return dim_outside(crop, ((d.x - d.w / 2) * w - x0, (d.y - d.h / 2) * h - y0,
                              (d.x + d.w / 2) * w - x0, (d.y + d.h / 2) * h - y0))


def dim_outside(img: np.ndarray, box) -> np.ndarray:
    """Darken everything outside box (x0, y0, x1, y1 px), so the model judges the subject,
    not a neighbour who happens to share the close-up."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    out = (img.astype(np.float32) * DIM).astype(np.uint8)
    xs, ys = slice(max(0, x0), max(0, min(w, x1))), slice(max(0, y0), max(0, min(h, y1)))
    out[ys, xs] = img[ys, xs]
    return out


def read_frame(path: Path, view: RF.View, t: float, src_size: Tuple[int, int]) -> np.ndarray:
    """One RGB frame at t: equirect EQ_W x EQ_H for 360, FLAT_W wide for flat footage."""
    inp, inp2 = RF.input_labels(view, 0)
    if view.is360:
        w, h = S.EQ_W, S.EQ_H
        pre, opts = RF.unwrap_graph(view, inp, inp2, "u").split("|")
        conv = "" if view.kind == "equirect" else f"v360={opts}:output=e,"
        fc = f"{pre};[u]{conv}scale={w}:{h}[o]"
    else:
        sw, sh = src_size
        if not sw or not sh:
            from .ffmpeg import probe
            info = probe(path)
            sw, sh = info.width or 16, info.height or 9
        w, h = FLAT_W, max(2, int(round(FLAT_W * sh / sw / 2)) * 2)
        fc = f"[{inp}]scale={w}:{h}[o]"
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           *RF.window_inputs(path, view, t, 1.0), "-filter_complex", fc, "-map", "[o]",
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or len(proc.stdout) < w * h * 3:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ClipsError(f"cannot read frame at {t:.2f}s: {msg[-1] if msg else 'no frame'}")
    return np.frombuffer(proc.stdout[: w * h * 3], np.uint8).reshape(h, w, 3)


def jpeg(img: np.ndarray) -> bytes:
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img[: h // 2 * 2, : w // 2 * 2])
    h, w = img.shape[:2]
    proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
                           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", "-",
                           "-frames:v", "1", "-q:v", "3", "-f", "image2pipe", "-c:v", "mjpeg", "-"],
                          input=img.tobytes(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise ClipsError("cannot encode candidate image")
    return proc.stdout


def close_ups(path: Path, view: RF.View, tracks: Sequence[S.Track],
              src_size: Tuple[int, int] = (0, 0)) -> List[bytes]:
    out = []
    for t in tracks:
        k, d = best_sample(t)
        frame = read_frame(path, view, k * S.STEP, src_size)
        out.append(jpeg(view_crop(frame, d) if view.is360 else flat_crop(frame, d)))
    return out


# ---------------------------------------------------------------- Ollama
def _post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read().decode())


def score(image: bytes, desc: str, model: str) -> int:
    """0-10: how well the image matches the description, per the local VLM."""
    data = _post(host() + "/api/chat", {
        "model": model, "stream": False, "format": FORMAT, "options": {"temperature": 0},
        "messages": [{"role": "user", "content": PROMPT.format(desc=desc),
                      "images": [base64.b64encode(image).decode()]}]})
    text = (data.get("message") or {}).get("content", "")
    try:
        m = json.loads(text).get("match")
    except (ValueError, AttributeError):
        m = None
    if not isinstance(m, (int, float)):
        digits = "".join(c if c.isdigit() else " " for c in text).split()
        m = int(digits[0]) if digits else 0
    return int(np.clip(m, 0, 10))


def pick(images: Sequence[bytes], desc: str, model: str) -> Tuple[Optional[int], List[int]]:
    """(index of the best match or None, scores). None also when Ollama cannot be used."""
    scores = []
    try:
        for img in images:
            scores.append(score(img, desc, model))
    except urllib.error.HTTPError as e:
        hint = (f"; run `ollama pull {model}`" if e.code == 404 else "")
        print(f"warning: Ollama refused the request ({e.code}){hint}; using the most prominent "
              "subject", file=sys.stderr)
        return None, scores
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"warning: Ollama not reachable at {host()} ({getattr(e, 'reason', e)}); start "
              f"`ollama serve` (and `ollama pull {model}`); using the most prominent subject",
              file=sys.stderr)
        return None, scores
    if not scores or max(scores) < MIN_MATCH:
        print(f"warning: no subject looks like '{desc}' (best {max(scores, default=0)}/10); "
              "using the most prominent one", file=sys.stderr)
        return None, scores
    return int(np.argmax(scores)), scores


# ---------------------------------------------------------------- entry point
def chooser(path: Path, view: RF.View, duration: float, opts: S.Options,
            src_size: Tuple[int, int] = (0, 0)) -> Callable[[List[S.Track]], Optional[S.Track]]:
    """choose() for subject.subject_path: asks the VLM once per clip/description, cached."""
    cf = S._cache_file(path, view, duration, opts.model)
    pf = cf.with_name(cf.stem + ".picks.json")
    key = f"{opts.vlm}|{opts.desc.strip().lower()}"

    def choose(pool: List[S.Track]) -> Optional[S.Track]:
        try:
            picks = json.loads(pf.read_text()) if pf.exists() else {}
        except ValueError:
            picks = {}
        if key in picks:
            tid = picks[key]
            if tid < 0:
                print(f"note: no subject in {Path(path).name} looked like '{opts.desc}' last "
                      "time (cached); using the most prominent one", file=sys.stderr)
            return next((t for t in pool if t.id == tid), None)
        cands = pool[:CANDIDATES]
        print(f"asking {opts.vlm} which of {len(cands)} subjects in {Path(path).name} is "
              f"'{opts.desc}' ...", file=sys.stderr)
        i, scores = pick(close_ups(path, view, cands, src_size), opts.desc, opts.vlm)
        if scores:
            print("  match scores: " + ", ".join(str(s) for s in scores), file=sys.stderr)
        if len(scores) == len(cands):          # a real answer (even "none"): remember it
            picks[key] = cands[i].id if i is not None else -1
            pf.parent.mkdir(parents=True, exist_ok=True)
            pf.write_text(json.dumps(picks, indent=1))
        return cands[i] if i is not None else None

    return choose
