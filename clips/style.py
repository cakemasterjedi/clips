"""Style learner: measure how a creator edits (local DSP only) and save it as a profile.

Video side works on downscaled grayscale frames piped from ffmpeg: cuts are frame-difference
spikes, flashes are bright frames right after a cut that fade within ~0.2 s, punch zooms are a
zoom-out over the first frames of a shot, and the fit mode comes from the top/bottom bands
(black = letterbox, smooth = blurred backdrop). Audio side uses librosa and the speech
heuristics: beats, how cuts sit on them, how shot length follows loudness, a dialogue intro,
how the music enters under it, and how long after the music starts the big drop lands.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Sequence

import numpy as np

from . import audio as A
from . import speech
from .ffmpeg import ClipsError, ffmpeg_exe, probe
from .sources import VIDEO_EXTS

VERSION = 1
FW, FH = 144, 256            # analysis frame size (9:16)
AFPS = 30                    # analysis frame rate
CUT_MIN_GAP = 0.2            # s; a fade/zoom right after a cut is not another cut
CUT_ABS = 12.0               # min mean abs diff (0-255) of the block-averaged frames
CUT_REL = 4.0                # ... and this many times the local median diff
ON_BEAT_TOL = 0.07           # s
FLASH_JUMP = 25.0            # luma rise at the cut vs the frame before (0-255)
FLASH_DECAY = 20.0           # luma drop over the next FLASH_LOOK frames
FLASH_LOOK = 5
ZOOM_LOOK = 6                # frames after the cut compared with the first one
ZOOM_SCALES = np.round(np.arange(1.0, 1.205, 0.02), 3)
ZOOM_MIN = 1.05
ZOOM_GAIN = 0.75             # best-scale error must be below this share of the no-zoom error
BAND = 0.12                  # top/bottom band height for fit detection
FRAME_STEP = 6               # frames between the pairs compared for camera motion (0.2 s)
MOVE_PAN = 0.04              # frame widths/s of pan that count as a moving camera
MOVE_PUSH = 0.03             # zoom per second (log scale) that counts as a push-in/pull-out
PUSH_SCALES = np.round(np.arange(1.0, 1.0605, 0.005), 3)
DEFAULT_CACHE = Path.home() / ".cache" / "clips" / "refs"
URL_RE = re.compile(r"^https?://", re.I)


# ---------------------------------------------------------------- fetching
def is_url(s: str) -> bool:
    return bool(URL_RE.match(s))


BLOCKED_HINT = ("the site refused the download. Run the updater (newer yt-dlp), make sure "
                "curl_cffi is installed (it lets yt-dlp look like a browser), or pass "
                "--cookies-from-browser firefox|chrome to use your browser's login")


def impersonate_target():
    """yt-dlp's Chrome impersonation target when curl_cffi is installed, else None."""
    try:
        from yt_dlp.networking.impersonate import ImpersonateTarget
        import curl_cffi  # noqa: F401
        return ImpersonateTarget.from_str("chrome")
    except Exception:
        return None


def download(url: str, cache: Path, cookies: Optional[str] = None) -> Path:
    """Download one reference video with yt-dlp into `cache` (reused if already there).
    A failed attempt is retried once impersonating Chrome (TikTok answers plain requests with
    "Unexpected response from webpage request"); `cookies` = a browser to borrow cookies from."""
    try:
        import yt_dlp
    except ImportError:
        raise ClipsError("downloading links needs yt-dlp: .venv/bin/python -m pip install yt-dlp") from None
    cache.mkdir(parents=True, exist_ok=True)
    base = {"outtmpl": str(cache / "%(extractor)s-%(id)s.%(ext)s"),
            "format": "bv*[height<=1920]+ba/b[height<=1920]/b",
            "merge_output_format": "mp4", "ffmpeg_location": ffmpeg_exe(),
            "noplaylist": True, "quiet": True, "no_warnings": True}
    if cookies:
        base["cookiesfrombrowser"] = (cookies,)
    attempts = [base]
    imp = impersonate_target()
    if imp is not None:
        attempts.append(dict(base, impersonate=imp))
    print(f"downloading {url} (yt-dlp {yt_dlp.version.__version__}"
          f"{', browser impersonation available' if imp else ', no curl_cffi'})...", file=sys.stderr)
    err = ""
    for k, opts in enumerate(attempts):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
                if not path.exists():
                    path = Path(ydl.prepare_filename(info))
            break
        except Exception as exc:   # yt-dlp raises many types; surface one line
            err = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            if k + 1 < len(attempts):
                print(f"download failed ({err}); retrying as Chrome", file=sys.stderr)
    else:
        hint = f"; {BLOCKED_HINT}" if re.search(r"Unexpected response|403|blocked|Unable to extract"
                                                 r"|login|cookies", err, re.I) else ""
        raise ClipsError(f"cannot download {url}: {err}{hint}") from None
    if not path.exists():
        raise ClipsError(f"download of {url} produced no file")
    return path


def resolve_refs(items: Sequence[str], cache: Path = DEFAULT_CACHE,
                 cookies: Optional[str] = None) -> List[Path]:
    """Links are downloaded; local files are used in place; folders contribute their videos."""
    out: List[Path] = []
    for it in items:
        if is_url(it):
            out.append(download(it, cache, cookies))
            continue
        p = Path(it)
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS))
        elif p.is_file():
            out.append(p)
        else:
            raise ClipsError(f"reference not found: {it}")
    if not out:
        raise ClipsError("no reference videos")
    return out


# ---------------------------------------------------------------- frame analysis
def read_frames(path, fps: int = AFPS) -> np.ndarray:
    """(n, FH, FW) uint8 luma, aspect kept (padded black) so letterboxing stays visible."""
    vf = (f"fps={fps},scale={FW}:{FH}:force_original_aspect_ratio=decrease,"
          f"pad={FW}:{FH}:(ow-iw)/2:(oh-ih)/2:black,format=gray")
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-an", "-vf", vf, "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ClipsError(f"cannot read video {path}: {msg[-1] if msg else 'ffmpeg failed'}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (FW * FH)
    return buf[: n * FW * FH].reshape(n, FH, FW)


def _blocks(frames: np.ndarray, k: int = 8, chunk: int = 256) -> np.ndarray:
    n, h, w = frames.shape
    out = []
    for i in range(0, n, chunk):   # chunked so long uint8 videos never become one float array
        f = frames[i:i + chunk, : h // k * k, : w // k * k].astype(np.float32)
        out.append(f.reshape(len(f), h // k, k, w // k, k).mean(axis=(2, 4)))
    return np.concatenate(out)


def luma(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=(1, 2), dtype=np.float64)


def detect_cuts(frames: np.ndarray, fps: int = AFPS) -> List[int]:
    """Frame indices where a new shot starts (0 excluded). Block-averaged frames make the
    difference robust to motion; a spike must beat both an absolute floor and the local median."""
    if len(frames) < 2:
        return []
    b = _blocks(frames)
    d = np.abs(np.diff(b, axis=0)).mean(axis=(1, 2))         # d[i]: frame i -> i+1
    gap = max(1, int(round(CUT_MIN_GAP * fps)))
    cuts: List[int] = []
    for i in range(d.size):
        lo, hi = max(0, i - 7), min(d.size, i + 8)
        local = np.median(np.delete(d[lo:hi], i - lo)) if hi - lo > 1 else 0.0
        if d[i] >= CUT_ABS and d[i] >= CUT_REL * max(local, 1.0):
            f = i + 1
            if cuts and f - cuts[-1] < gap:
                if d[i] > d[cuts[-1] - 1]:
                    cuts[-1] = f
                continue
            cuts.append(f)
    return cuts


def is_flash(frames: np.ndarray, cut: int, lum: Optional[np.ndarray] = None) -> bool:
    lum = luma(frames) if lum is None else lum
    if cut < 1 or cut + FLASH_LOOK >= len(lum):
        return False
    return (lum[cut] - lum[cut - 1] >= FLASH_JUMP
            and lum[cut] - lum[cut + FLASH_LOOK] >= FLASH_DECAY)


def zoom(img: np.ndarray, s: float) -> np.ndarray:
    """Centre zoom by s (>=1): crop the middle 1/s and scale back up (bilinear)."""
    h, w = img.shape
    ys = (np.arange(h) - (h - 1) / 2) / s + (h - 1) / 2
    xs = (np.arange(w) - (w - 1) / 2) / s + (w - 1) / 2
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 2)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 2)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    a = img[y0][:, x0]
    b = img[y0][:, x0 + 1]
    c = img[y0 + 1][:, x0]
    d = img[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _norm(img: np.ndarray) -> np.ndarray:
    return (img - img.mean()) / max(float(img.std()), 1e-3)


def punch_scale(frames: np.ndarray, cut: int, end: int, lum: Optional[np.ndarray] = None) -> float:
    """Zoom of the shot's first frame relative to ZOOM_LOOK frames later (1.0 = no punch).
    On a flash the all-white first frame is skipped; frames are normalized (zero mean, unit
    std) so the rest of the white fade, an affine change in luma, does not count as a
    difference. Only the central area is compared (borders differ after zoom)."""
    first = cut + 1 if is_flash(frames, cut, lum) else cut
    later = first + ZOOM_LOOK
    if later >= min(end, len(frames)):
        return 1.0
    h, w = frames.shape[1:]
    m = (slice(h // 6, h - h // 6), slice(w // 6, w - w // 6))
    f0, f1 = _norm(frames[first]), _norm(frames[later])
    errs = [float(np.abs(zoom(f1, s)[m] - f0[m]).mean()) for s in ZOOM_SCALES]
    i = int(np.argmin(errs))
    if ZOOM_SCALES[i] >= ZOOM_MIN and errs[i] < ZOOM_GAIN * errs[0]:
        return float(ZOOM_SCALES[i])
    return 1.0


def detect_fit(frames: np.ndarray) -> str:
    """letterbox: top/bottom bands are black; blur: bands are smooth but not black; else crop."""
    if len(frames) == 0:
        return "crop"
    sample = frames[:: max(1, len(frames) // 60)].astype(np.float32)
    bh = max(2, int(FH * BAND))
    bands = np.concatenate([sample[:, :bh], sample[:, -bh:]], axis=1)
    mid = sample[:, FH // 3: 2 * FH // 3]
    if np.median(bands.mean(axis=(1, 2))) < 16 and np.median(bands.std(axis=(1, 2))) < 6:
        return "letterbox"

    def grad(x):
        return float(np.abs(np.diff(x, axis=2)).mean() + np.abs(np.diff(x, axis=1)).mean())

    g_band, g_mid = grad(bands), grad(mid)
    if g_mid > 2.0 and g_band < 0.25 * g_mid:
        return "blur"
    return "crop"


# ---------------------------------------------------------------- audio analysis
def detect_intro(path, length: float) -> float:
    """Length of a dialogue intro at the very start (0 if none)."""
    y = A.decode(path, 0.0, None, speech.SR)
    if y.size == 0:
        return 0.0
    regs = speech.detect_regions(y, speech.SR)
    first = [r for r in regs if r.start <= 1.0]
    return float(min(first[0].end, length)) if first else 0.0


def music_entry(env: np.ndarray, env_times: np.ndarray, beats: np.ndarray,
                t_intro: float, step: float = 1.0) -> float:
    """Seconds of music already playing under the end of the intro (0 = drop).
    The beat grid of the music section is extended back into the intro; walking back from the
    intro end, a stretch counts as music while onset strength on the grid clearly beats the
    strength between grid points (speech alone is irregular, so the ratio stays near 1)."""
    post = beats[beats >= t_intro]
    if t_intro <= 0 or post.size < 4:
        return 0.0
    period = float(np.median(np.diff(post)))
    grid = post[0] - period * np.arange(1, int(post[0] / period) + 1)
    grid = grid[grid >= 0]

    def val(t):   # strongest onset within +-50 ms (the extended grid drifts a little)
        m = np.abs(env_times - t) <= 0.05
        return float(env[m].max()) if m.any() else 0.0

    rise = 0.0
    t = t_intro
    while t - step >= 0:
        g = grid[(grid >= t - step) & (grid < t)]
        if g.size == 0:
            t -= step
            continue
        on = np.mean([val(x) for x in g])
        off = np.mean([val(x + period / 2) for x in g])
        if on < 1.5 * max(off, 1e-6) or on < 0.05 * float(env.max()):
            break
        rise = t_intro - (t - step)
        t -= step
    return round(min(rise, t_intro), 2)


def lead_in(rms: np.ndarray, rms_times: np.ndarray, start: float, end: float,
            search: float = 15.0, min_jump: float = 0.15) -> float:
    """Seconds from the start of the music section to its biggest loudness jump (0 if the
    music section opens at full energy)."""
    m = (rms_times >= start) & (rms_times < end)
    if m.sum() < 4 or rms[m].max() <= 0:
        return 0.0
    t = rms_times[m]
    hop = float(t[1] - t[0]) if t.size > 1 else 0.023
    k = max(1, int(round(0.5 / hop)))          # 0.5 s moving mean: loudness, not transients
    r = np.convolve(rms[m], np.ones(k) / k, mode="same")
    r = r / max(float(r.max()), 1e-12)
    best, best_t = 0.0, start
    for tt in t[(t >= start + 1.0) & (t < start + search)]:   # needs some music before it
        pre = r[(t >= tt - A.DROP_PRE) & (t < tt)]
        post = r[(t >= tt) & (t < tt + A.DROP_POST)]
        if pre.size and post.size and post.mean() - pre.mean() > best:
            best, best_t = post.mean() - pre.mean(), tt
    return round(float(best_t - start), 2) if best >= min_jump else 0.0


def fit_max_shot(lengths: np.ndarray, energies: np.ndarray, period: float) -> float:
    """Least-squares M in  len = period + (M - period) * (1 - e)^1.5  (the auto pacing curve)."""
    w = (1.0 - np.clip(energies, 0, 1)) ** 1.5
    if lengths.size == 0 or float((w * w).sum()) < 1e-9:
        return float(np.clip(np.median(lengths) if lengths.size else A.AUTO_MAX_SHOT,
                             period, A.MAX_SHOT))
    m = period + float((w * (lengths - period)).sum() / (w * w).sum())
    return float(np.clip(m, max(period, A.MIN_SHOT), A.MAX_SHOT))


# ---------------------------------------------------------------- framing
def phase_shift(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """(dx, dy) in px that moves image a onto b (phase correlation, sub-pixel peak)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2(_norm(a) * win), np.fft.fft2(_norm(b) * win)
    r = fb * np.conj(fa)
    r = np.real(np.fft.ifft2(r / np.maximum(np.abs(r), 1e-9)))
    iy, ix = np.unravel_index(int(np.argmax(r)), r.shape)
    h, w = r.shape

    def sub(c, m, n):     # parabola through the peak and its neighbours (circular)
        l, mid, rr = m(c - 1), m(c), m(c + 1)
        den = l - 2 * mid + rr
        off = 0.5 * (l - rr) / den if abs(den) > 1e-9 else 0.0
        v = c + float(np.clip(off, -0.5, 0.5))
        return v - n if v > n / 2 else v

    dx = sub(ix, lambda c: r[iy, c % w], w)
    dy = sub(iy, lambda c: r[c % h, ix], h)
    return dx, dy


def push_rate(a: np.ndarray, b: np.ndarray, dt: float, shift=(0.0, 0.0)) -> float:
    """Zoom per second (log scale, + = push in) between frames a and b dt apart; b is first
    moved back by the pan `shift` (dx, dy px) so a pan does not read as a zoom."""
    h, w = a.shape
    m = (slice(h // 6, h - h // 6), slice(w // 6, w - w // 6))
    b = np.roll(b, (-int(round(shift[1])), -int(round(shift[0]))), axis=(0, 1))
    na, nb = _norm(a.astype(np.float32)), _norm(b.astype(np.float32))
    best = (float(np.abs(na[m] - nb[m]).mean()), 1.0)
    for sc in PUSH_SCALES[1:]:
        best = min(best, (float(np.abs(zoom(na, sc)[m] - nb[m]).mean()), sc),
                   (float(np.abs(zoom(nb, sc)[m] - na[m]).mean()), 1 / sc))
    return float(np.log(best[1]) / dt)


def content_rows(frames: np.ndarray) -> slice:
    """Rows of the picture, without black/blurred letterbox bands (fit != crop)."""
    fit = detect_fit(frames[:: max(1, len(frames) // 20)])
    if fit == "crop":
        return slice(0, frames.shape[1])
    col = frames[:: max(1, len(frames) // 20)].astype(np.float32)
    var = np.abs(np.diff(col, axis=2)).mean(axis=(0, 2))       # texture per row
    rows = np.where(var > 0.5 * np.median(var[var > np.percentile(var, 60)]))[0]
    if rows.size < frames.shape[1] // 4:
        return slice(0, frames.shape[1])
    return slice(int(rows[0]), int(rows[-1]) + 1)


def shot_framing(frames: np.ndarray, start: int, end: int, rows: slice) -> Optional[dict]:
    """Camera motion and subject height inside one shot (None if too short)."""
    a = start + ZOOM_LOOK + 2                 # skip a punch-in settling after the cut
    if end - a < 2 * FRAME_STEP:
        return None
    f = frames[a:end, rows].astype(np.float32)
    idx = np.arange(0, len(f) - FRAME_STEP, FRAME_STEP)
    shifts = np.array([phase_shift(f[i], f[i + FRAME_STEP]) for i in idx])
    dt = FRAME_STEP / AFPS
    h, w = f.shape[1:]
    pan = float(np.hypot(*np.median(shifts, axis=0))) / w / dt     # frame widths / s
    pushes = [push_rate(f[i], f[i + FRAME_STEP], dt, sh) for i, sh in zip(idx, shifts)]
    push = float(np.median(pushes))
    d = np.abs(np.diff(f[:: 3], axis=0))
    d = np.where(d > 12, d, 0).sum(axis=(0, 2))                 # motion per row
    y = float((d * (np.arange(h) + 0.5) / h).sum() / d.sum()) if d.sum() > 0 else 0.5
    return {"pan": pan, "push": push, "subject_y": y, "len": (end - a) / AFPS}


def framing_stats(frames: np.ndarray, starts: Sequence[int], ends: Sequence[int]) -> dict:
    rows = content_rows(frames)
    shots = [s for s in (shot_framing(frames, a, b, rows) for a, b in zip(starts, ends)) if s]
    if not shots:
        return {}
    moving = [s for s in shots if s["pan"] >= MOVE_PAN]
    pushing = [s for s in shots if abs(s["push"]) >= MOVE_PUSH]
    return {"moving": round(len(moving) / len(shots), 3),
            "pan_speed": round(float(np.median([s["pan"] for s in moving])), 3) if moving else 0.0,
            "push_share": round(len(pushing) / len(shots), 3),
            "push": round(float(np.median([s["push"] for s in pushing])), 3) if pushing else 0.0,
            "subject_y": round(float(np.median([s["subject_y"] for s in shots])), 3)}


# ---------------------------------------------------------------- one reference
@dataclass
class RefStats:
    source: str
    duration: float
    tempo: float
    t_intro: float
    cuts: List[float]
    on_beat: float
    beats_per_cut: Optional[int]
    max_shot: float
    median_shot: float
    onset_cuts: float
    punch: float
    punch_zoom: float
    flash: float
    fit: str
    music_in: str
    rise: float
    lead_in: float
    extra: Dict = field(default_factory=dict)

    def to_dict(self):
        d = dict(self.__dict__)
        d["cuts"] = [round(c, 3) for c in self.cuts]
        return d


def analyze_reference(path, label: Optional[str] = None) -> RefStats:
    import librosa

    path = Path(path)
    info = probe(path)
    if not info.has_video:
        raise ClipsError(f"no video stream in {path}")
    frames = read_frames(path)
    duration = len(frames) / AFPS
    cut_f = detect_cuts(frames)
    starts = [0] + cut_f
    ends = cut_f + [len(frames)]
    lum = luma(frames)
    scales = [punch_scale(frames, s, e, lum) for s, e in zip(starts[1:], ends[1:])]
    flashes = [is_flash(frames, c, lum) for c in cut_f]
    fit = detect_fit(frames)
    framing = framing_stats(frames, starts, ends)
    del frames

    t_intro, tempo, beats = 0.0, 0.0, np.zeros(0)
    music_in, rise, lead = "drop", 0.0, 0.0
    cuts = [c / AFPS for c in cut_f]
    on_beat, bpc, onset_cuts = 0.0, None, 0.0
    period = 0.5
    energies = np.zeros(0)
    if info.has_audio:
        y = A.decode(path)
        env = librosa.onset.onset_strength(y=y, sr=A.SR, hop_length=A.HOP)
        env_t = librosa.frames_to_time(np.arange(env.size), sr=A.SR, hop_length=A.HOP)
        rms = librosa.feature.rms(y=y, hop_length=A.HOP)[0]
        rms_t = librosa.frames_to_time(np.arange(rms.size), sr=A.SR, hop_length=A.HOP)
        t_intro = detect_intro(path, duration)
        # an intro is usually one uncut shot ending on the music's first cut; the speech
        # detector tends to stop a little early once music comes in under the voice
        if t_intro > 0 and cuts:
            after = [c for c in cuts if c >= t_intro - 0.6]
            if after and after[0] <= t_intro + 1.5:
                t_intro = after[0]
        seg = (env_t >= t_intro)
        tempo_a, bf = librosa.beat.beat_track(onset_envelope=env[seg], sr=A.SR, hop_length=A.HOP)
        tempo = float(np.atleast_1d(tempo_a)[0])
        beats = librosa.frames_to_time(bf, sr=A.SR, hop_length=A.HOP) + (env_t[seg][0] if seg.any() else 0)
        if beats.size > 1:
            period = float(np.median(np.diff(beats)))
        music_in_rise = music_entry(env, env_t, beats, t_intro)
        music_in, rise = ("rise", music_in_rise) if music_in_rise >= 0.5 else ("drop", 0.0)
        lead = lead_in(rms, rms_t, t_intro, duration)
        mcuts = np.array([c for c in cuts if c > t_intro + 1e-3])
        if mcuts.size and beats.size:
            dist = np.array([np.min(np.abs(beats - c)) for c in mcuts])
            on_beat = float((dist <= ON_BEAT_TOL).mean())
            thr = np.percentile(env, 85)
            strong = np.array([np.interp(c, env_t, env) >= thr for c in mcuts])
            onset_cuts = float(((dist > ON_BEAT_TOL) & strong).mean())
        bounds = np.concatenate([[t_intro], mcuts])
        lens = np.diff(np.append(bounds, duration))
        if lens.size:
            energies = A.interval_energy(bounds, rms, rms_t, duration, smooth=1)
            in_beats = np.round(lens / period).astype(int)
            mode = int(np.bincount(np.maximum(in_beats, 0)).argmax()) if in_beats.size else 0
            # the last shot is cut off by the end of the video, so it does not count
            body = in_beats[:-1] if in_beats.size > 2 else in_beats
            if mode >= 1 and body.size >= 3 and (body == mode).mean() >= 0.8 and on_beat >= 0.8:
                bpc = mode
    else:
        bounds = np.array([0.0] + cuts)
        lens = np.diff(np.append(bounds, duration))
    body = slice(None, -1) if lens.size > 2 else slice(None)
    max_shot = fit_max_shot(lens[body], energies[body] if energies.size else np.full(lens[body].size, 0.5),
                            period)
    pz = [s for s in scales if s > 1.0]
    return RefStats(
        source=label or path.name, duration=round(duration, 3), tempo=round(tempo, 2),
        t_intro=round(t_intro, 3), cuts=cuts, on_beat=round(on_beat, 3), beats_per_cut=bpc,
        max_shot=round(max_shot, 3), median_shot=round(float(np.median(lens)) if lens.size else 0.0, 3),
        onset_cuts=round(onset_cuts, 3),
        punch=round(len(pz) / len(scales), 3) if scales else 0.0,
        punch_zoom=round(float(np.median(pz)) - 1.0, 3) if pz else 0.0,
        flash=round(float(np.mean(flashes)), 3) if flashes else 0.0,
        fit=fit, music_in=music_in, rise=rise, lead_in=lead, extra={"framing": framing})


# ---------------------------------------------------------------- profile
def build_profile(stats: Sequence[RefStats], name: str) -> dict:
    """One creator profile: medians over the references (majority vote for categorical)."""
    if not stats:
        raise ClipsError("no references analysed")

    def med(key):
        return round(float(np.median([getattr(s, key) for s in stats])), 3)

    def vote(key):
        vals = [getattr(s, key) for s in stats]
        return max(sorted(set(vals), key=str), key=vals.count)

    bpcs = [s.beats_per_cut for s in stats]
    bpc = bpcs[0] if all(b == bpcs[0] for b in bpcs) else None
    intro_on = sum(s.t_intro > 0 for s in stats) * 2 > len(stats)
    with_intro = [s for s in stats if s.t_intro > 0]
    rises = [s for s in with_intro if s.music_in == "rise"]
    music_in = "rise" if rises and len(rises) * 2 >= len(with_intro) else "drop"
    return {
        "version": VERSION, "name": name,
        "pacing": {"beats_per_cut": bpc, "max_shot": med("max_shot"),
                   "median_shot": med("median_shot"), "on_beat": med("on_beat"),
                   "onset_cuts": med("onset_cuts") >= 0.15},
        "fx": {"punch": med("punch"), "flash": med("flash"),
               "punch_zoom": round(float(np.median([s.punch_zoom for s in stats if s.punch > 0])), 3)
               if any(s.punch > 0 for s in stats) else 0.0},
        "fit": vote("fit"),
        "intro": {"use": intro_on,
                  "max": round(float(np.median([s.t_intro for s in with_intro])), 2) if with_intro else 0.0},
        "music": {"music_in": music_in,
                  "rise": round(float(np.median([s.rise for s in rises])), 2) if rises else 0.0,
                  "lead_in": med("lead_in")},
        "framing": merge_framing([s.extra.get("framing") or {} for s in stats]),
        "refs": [s.to_dict() for s in stats],
    }


FRAMING_DEFAULT = {"moving": 0.5, "pan_speed": 0.15, "push_share": 0.0, "push": 0.0,
                   "subject_y": 0.5}


def merge_framing(items: Sequence[dict]) -> dict:
    """Median of each framing measure over the references that have one."""
    items = [f for f in items if f]
    if not items:
        return {}
    out = {}
    for k in FRAMING_DEFAULT:
        vals = [f[k] for f in items if k in f]
        if k in ("pan_speed", "push"):        # only from the videos that move / push
            share = "moving" if k == "pan_speed" else "push_share"
            vals = [f[k] for f in items if f.get(share, 0) > 0]
        out[k] = round(float(np.median(vals)), 3) if vals else 0.0
    return out


def learn(items: Sequence[str], name: str, cache: Path = DEFAULT_CACHE,
          cookies: Optional[str] = None) -> dict:
    stats = []
    for item, path in zip_refs(items, cache, cookies):
        print(f"analysing {path.name}...", file=sys.stderr)
        stats.append(analyze_reference(path, item if is_url(item) else path.name))
    return build_profile(stats, name)


def zip_refs(items: Sequence[str], cache: Path, cookies: Optional[str] = None):
    for it in items:
        for p in resolve_refs([it], cache, cookies):
            yield it, p


def load_profile(path) -> dict:
    try:
        prof = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClipsError(f"cannot read style profile {path}: {exc}") from None
    if not isinstance(prof, dict) or prof.get("version") != VERSION:
        raise ClipsError(f"{path} is not a clips style profile (version {VERSION})")
    if prof.get("fit") not in (None, "crop", "letterbox", "blur"):
        raise ClipsError(f"{path}: bad fit {prof.get('fit')!r}")
    if prof.get("music", {}).get("music_in") not in (None, "rise", "drop"):
        raise ClipsError(f"{path}: bad music_in {prof['music']['music_in']!r}")
    return prof
