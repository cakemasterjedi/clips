"""Subject tracking: a local object detector tells the framer *what* to follow.

Motion alone follows whatever moves most (a passing car, waving trees). With `--subject` the
framer follows a chosen subject instead (a person, a dog, the biggest thing in view), and
falls back to motion whenever the subject is not found.

Pipeline, all local:

1. Frames every STEP s from FFmpeg. 360 clips are unwrapped to an equirectangular frame and
   cut into TILES perspective views around the horizon (numpy gnomonic remap): detectors are
   trained on ordinary photos and miss stretched or seam-split people on the raw equirect.
2. YOLOX (Apache-2.0, Megvii's official ONNX exports) through ONNX Runtime. The model file is
   downloaded once into the cache and checked against its SHA-256. GPU when a GPU build of
   ONNX Runtime is installed (DirectML on Windows, MIGraphX/ROCm on Linux AMD, CUDA, CoreML),
   otherwise CPU.
3. Detections from overlapping tiles are merged in angle space (yaw/pitch, degrees); flat
   footage uses frame fractions (0..1) instead.
4. A ByteTrack-style tracker links detections into tracks (high-confidence matches first,
   then low-confidence ones, constant-velocity prediction, yaw wraps at +-180).
5. The subject is the most prominent track of the asked class (confidence x size x time on
   screen). When it is lost, a later (or earlier) track of the same class with a similar
   colour histogram takes over, so the same person is picked up again after an occlusion.

Detections are cached per clip (path, size, mtime, model), so variations and re-renders only
redo the cheap tracking. Requires `pip install -r requirements-vision.txt` (onnxruntime);
everything else in Clips works without it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import reframe as RF
from .ffmpeg import ClipsError, ffmpeg_exe

STEP = 0.5                  # s between analysed frames (same as the auto framer)
EQ_W, EQ_H = 1920, 960      # equirect size the 360 tiles are cut from
TILES = 5                   # perspective tiles around the horizon
TILE_FOV = 100.0            # deg; 5 x 72 deg apart leaves 28 deg of overlap for split subjects
CONF = 0.25                 # detections kept
HIGH = 0.5                  # detections that may start a track / match first
NMS_IOU = 0.45
MAX_GAP = 2.0               # s a track may go unseen before it ends
FILL_GAP = 2.0              # s of missing subject bridged by interpolation
GATE = 1.0                  # match if centre distance < GATE x the larger box size
REID_SIM = 0.6              # colour-histogram intersection to hand over to another track
HEAD_UP = 0.15              # aim this share of a person's height above the box centre
CACHE_VERSION = 1

COCO = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush")
ALIASES = {"people": "person", "me": "person", "human": "person", "bike": "bicycle",
           "motorbike": "motorcycle", "ball": "sports ball", "phone": "cell phone"}

_REL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
MODELS = {   # name: (file, input size, sha256)   all Apache-2.0, COCO 80 classes
    "s": ("yolox_s.onnx", 640, "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"),
    "tiny": ("yolox_tiny.onnx", 416, "427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7"),
    "nano": ("yolox_nano.onnx", 416, "c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d"),
}
GPU_PROVIDERS = ("CUDAExecutionProvider", "MIGraphXExecutionProvider", "ROCMExecutionProvider",
                 "DmlExecutionProvider", "CoreMLExecutionProvider")
GPU_HELP = ("no GPU build of ONNX Runtime found. Windows (any GPU, incl. AMD Radeon): "
            "pip uninstall onnxruntime && pip install onnxruntime-directml. Linux AMD: install "
            "ROCm + MIGraphX, then pip install onnxruntime-migraphx (RX 6000 cards such as the "
            "6700 XT also need HSA_OVERRIDE_GFX_VERSION=10.3.0). Or use --device cpu.")


@dataclass
class Options:
    what: str = "auto"          # auto | biggest | class name(s), comma-separated
    model: str = "s"
    device: str = "auto"        # auto | cpu | gpu
    desc: Optional[str] = None  # "the guy in the red jacket": pick which one with a local VLM
    vlm: str = "qwen2.5vl"      # Ollama model for desc


def options_from(args) -> Optional[Options]:
    what = getattr(args, "subject", None)
    desc = getattr(args, "subject_desc", None)
    if not what and not desc:
        return None
    return Options(what or "auto", getattr(args, "detector", "s") or "s",
                   getattr(args, "device", "auto") or "auto", desc or None,
                   getattr(args, "ollama_model", None) or Options.vlm)


def classes_of(what: str) -> Optional[List[int]]:
    """COCO class ids for a --subject value; None means any class (auto/biggest)."""
    w = what.strip().lower()
    if w in ("auto", "biggest", "any"):
        return None
    out = []
    for name in (n.strip() for n in w.split(",") if n.strip()):
        name = ALIASES.get(name, name)
        if name not in COCO:
            raise ClipsError(f"--subject: unknown subject '{name}' (use auto, biggest, or one of: "
                             + ", ".join(COCO) + ")")
        out.append(COCO.index(name))
    return out


# ---------------------------------------------------------------- model + runtime
def cache_dir() -> Path:
    base = os.environ.get("CLIPS_CACHE")
    if base:
        return Path(base)
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "clips"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "clips"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(name: str, base_url: str = _REL) -> Path:
    """Local path of a detector model, downloading and verifying it on first use."""
    if name not in MODELS:
        raise ClipsError(f"--detector must be one of {', '.join(MODELS)}")
    fname, _, sha = MODELS[name]
    path = cache_dir() / "models" / fname
    if path.exists() and _sha256(path) == sha:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    print(f"downloading detector {fname} (once) ...", file=sys.stderr)
    try:
        with urllib.request.urlopen(base_url + fname, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise ClipsError(f"cannot download {base_url + fname} ({e}); download it by hand into "
                         f"{path.parent}") from None
    if _sha256(tmp) != sha:
        tmp.unlink(missing_ok=True)
        raise ClipsError(f"{fname}: checksum mismatch, download discarded")
    tmp.replace(path)
    return path


def providers(device: str, available: Sequence[str]) -> List[str]:
    """ONNX Runtime execution providers for --device, best first, CPU always last."""
    if device not in ("auto", "cpu", "gpu"):
        raise ClipsError("--device must be auto, cpu or gpu")
    if device == "cpu":
        return ["CPUExecutionProvider"]
    gpu = [p for p in GPU_PROVIDERS if p in available]
    if device == "gpu" and not gpu:
        raise ClipsError(GPU_HELP)
    return gpu + ["CPUExecutionProvider"]


def _ort():
    try:
        import onnxruntime
    except ImportError:
        raise ClipsError("--subject needs ONNX Runtime: pip install -r requirements-vision.txt "
                         "(or onnxruntime-directml on Windows for GPU)") from None
    onnxruntime.set_default_logger_severity(3)
    return onnxruntime


class Detector:
    """YOLOX ONNX model: RGB uint8 image (h, w <= size) -> (n, 6) x0, y0, x1, y1, score, cls."""

    def __init__(self, model: str = "s", device: str = "auto"):
        ort = _ort()
        self.size = MODELS[model][1] if model in MODELS else 640
        provs = providers(device, ort.get_available_providers())
        opts = ort.SessionOptions()
        if "DmlExecutionProvider" in provs:       # DirectML wants these two off
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path(model)), sess_options=opts,
                                            providers=provs)
        self.input = self.session.get_inputs()[0].name
        self.provider = self.session.get_providers()[0]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        out = self.session.run(None, {self.input: preprocess(img, self.size)})[0][0]
        return postprocess(out, self.size)


def preprocess(img: np.ndarray, size: int) -> np.ndarray:
    """Top-left placed on a 114-grey square, BGR, CHW float32 0..255 (YOLOX >= 0.1.1rc0)."""
    h, w = img.shape[:2]
    if h > size or w > size:
        raise ValueError("image larger than the model input")
    pad = np.full((size, size, 3), 114, np.uint8)
    pad[:h, :w] = img
    return np.ascontiguousarray(pad[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)


def _grids(size: int):
    gs, ss = [], []
    for s in (8, 16, 32):
        n = size // s
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        gs.append(np.stack([xv, yv], -1).reshape(-1, 2))
        ss.append(np.full((n * n, 1), s))
    return np.concatenate(gs).astype(np.float32), np.concatenate(ss).astype(np.float32)


def postprocess(out: np.ndarray, size: int, conf: float = CONF) -> np.ndarray:
    grid, stride = _grids(size)
    xy = (out[:, :2] + grid) * stride
    wh = np.exp(np.minimum(out[:, 2:4], 10)) * stride
    cls = out[:, 5:].argmax(1)
    score = out[:, 4] * out[np.arange(len(out)), 5 + cls]
    keep = score >= conf
    if not keep.any():
        return np.zeros((0, 6), np.float32)
    xy, wh, cls, score = xy[keep], wh[keep], cls[keep], score[keep]
    boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1)
    k = nms(boxes + cls[:, None] * 4096.0, score, NMS_IOU)       # per class
    return np.concatenate([boxes[k], score[k, None], cls[k, None]], 1).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    order = scores.argsort()[::-1]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.maximum(0, xx1 - xx0) * np.maximum(0, yy1 - yy0)
        o = inter / (area[i] + area[order[1:]] - inter + 1e-9)
        order = order[1:][o <= iou]
    return np.array(keep, dtype=int)


# ---------------------------------------------------------------- 360 tiles
def tile_rays(size: int, fov: float = TILE_FOV) -> np.ndarray:
    """Unit view rays (size, size, 3) of a flat square view looking along +z (x right, y up)."""
    t = math.tan(math.radians(fov) / 2)
    c = ((np.arange(size) + 0.5) / size * 2 - 1) * t
    x, y = np.meshgrid(c, -c)
    r = np.stack([x, y, np.ones_like(x)], -1)
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def rotate(rays: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    p, q = math.radians(pitch), math.radians(yaw)
    x, y, z = rays[..., 0], rays[..., 1], rays[..., 2]
    y, z = y * math.cos(p) + z * math.sin(p), -y * math.sin(p) + z * math.cos(p)
    x, z = x * math.cos(q) + z * math.sin(q), -x * math.sin(q) + z * math.cos(q)
    return np.stack([x, y, z], -1)


def ray_angles(rays: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(yaw, pitch) in degrees; yaw 0 = equirect centre, +90 = right of it, pitch + = up."""
    yaw = np.degrees(np.arctan2(rays[..., 0], rays[..., 2]))
    pitch = np.degrees(np.arcsin(np.clip(rays[..., 1], -1, 1)))
    return yaw, pitch


def tile_yaws(n: int = TILES) -> List[float]:
    return [RF._wrap(360.0 * i / n) for i in range(n)]


def tile_maps(size: int, eq: Tuple[int, int] = (EQ_W, EQ_H), n: int = TILES,
              fov: float = TILE_FOV) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Per tile, (row, col) equirect indices of every tile pixel (nearest neighbour)."""
    w, h = eq
    base = tile_rays(size, fov)
    maps = []
    for yaw in tile_yaws(n):
        lon, lat = ray_angles(rotate(base, yaw, 0.0))
        col = np.clip(((lon + 180.0) / 360.0 * w).astype(int), 0, w - 1)
        row = np.clip(((90.0 - lat) / 180.0 * h).astype(int), 0, h - 1)
        maps.append((row, col))
    return maps


def tile_to_angles(u: float, v: float, size: int, yaw: float,
                   fov: float = TILE_FOV) -> Tuple[float, float]:
    """Tile pixel (u right, v down) -> (yaw, pitch) degrees."""
    t = math.tan(math.radians(fov) / 2)
    ray = np.array([(u / size * 2 - 1) * t, -(v / size * 2 - 1) * t, 1.0])
    yw, pt = ray_angles(rotate(ray / np.linalg.norm(ray), yaw, 0.0))
    return float(yw), float(pt)


# ---------------------------------------------------------------- detections
@dataclass
class Det:
    x: float          # 360: yaw deg; flat: 0..1
    y: float          # 360: pitch deg (up +); flat: 0..1 (down +)
    w: float          # size in the same units
    h: float
    score: float
    cls: int
    hist: Optional[List[float]] = None

    @property
    def size(self) -> float:
        return max(self.w, self.h)


def colour_hist(crop: np.ndarray) -> List[float]:
    """Normalised 8 hue x 4 saturation histogram (appearance for re-identification)."""
    c = crop.reshape(-1, 3).astype(np.float32) / 255.0
    if not len(c):
        return [0.0] * 32
    mx, mn = c.max(1), c.min(1)
    d = mx - mn + 1e-6
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    hue = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) / 6
    sat = np.where(mx > 0, (mx - mn) / (mx + 1e-6), 0)
    hb = np.clip((hue * 8).astype(int), 0, 7)
    sb = np.clip((sat * 4).astype(int), 0, 3)
    hist = np.bincount(hb * 4 + sb, minlength=32).astype(np.float64)
    return [round(float(v), 4) for v in hist / hist.sum()]


def _crop(img: np.ndarray, box) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box[:4])
    # the middle of the box: less background than the full rectangle
    cx0, cx1 = x0 + (x1 - x0) // 4, x1 - (x1 - x0) // 4
    cy0, cy1 = y0 + (y1 - y0) // 5, y1 - (y1 - y0) // 5
    return img[max(0, cy0):min(h, cy1), max(0, cx0):min(w, cx1)]


def dets_360(tiles: Sequence[np.ndarray], boxes: Sequence[np.ndarray], size: int,
             fov: float = TILE_FOV) -> List[Det]:
    """Tile detections -> merged angle-space detections."""
    yaws = tile_yaws(len(tiles))
    deg = fov / size
    out = []
    for img, bx, yaw in zip(tiles, boxes, yaws):
        for b in bx:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            ay, ap = tile_to_angles(cx, cy, size, yaw, fov)
            # boxes cut by the tile edge are partial; flag them so the full copy wins
            edge = min(b[0], b[1], size - b[2], size - b[3]) < 2
            score = float(b[4]) * (0.8 if edge else 1.0)
            out.append(Det(round(ay, 2), round(ap, 2), round(float(b[2] - b[0]) * deg, 2),
                           round(float(b[3] - b[1]) * deg, 2), round(score, 3), int(b[5]),
                           colour_hist(_crop(img, b))))
    return merge(out, wrap=True)


def dets_flat(img: np.ndarray, boxes: np.ndarray) -> List[Det]:
    h, w = img.shape[:2]
    return [Det(round(float(b[0] + b[2]) / 2 / w, 4), round(float(b[1] + b[3]) / 2 / h, 4),
                round(float(b[2] - b[0]) / w, 4), round(float(b[3] - b[1]) / h, 4),
                round(float(b[4]), 3), int(b[5]), colour_hist(_crop(img, b))) for b in boxes]


def _dx(a: float, b: float, wrap: bool) -> float:
    return RF._wrap(a - b) if wrap else a - b


def merge(dets: List[Det], wrap: bool) -> List[Det]:
    """Drop duplicates of the same object seen by two tiles (keep the best-scoring copy)."""
    keep: List[Det] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if not any(k.cls == d.cls and math.hypot(_dx(d.x, k.x, wrap), d.y - k.y) < 0.5 * max(k.size, d.size)
                   for k in keep):
            keep.append(d)
    return keep


# ---------------------------------------------------------------- frames
def _frames(cmd: List[str], w: int, h: int):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace").strip().splitlines()
        if proc.wait() != 0:
            raise ClipsError(f"cannot analyse video: {err[-1] if err else 'ffmpeg failed'}")


def _cmd(path: Path, view: RF.View, duration: float, vf: str) -> List[str]:
    inp, inp2 = RF.input_labels(view, 0)
    if view.is360:
        pre, opts = RF.unwrap_graph(view, inp, inp2, "u").split("|")
        conv = "" if view.kind == "equirect" else f"v360={opts}:output=e,"
        fc = f"{pre};[u]fps={1 / STEP:g},{conv}{vf}[o]"
    else:
        fc = f"[{inp}]fps={1 / STEP:g},{vf}[o]"
    return [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
            *RF.window_inputs(path, view, 0.0, duration), "-filter_complex", fc, "-map", "[o]",
            "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]


def detect_clip(path: Path, view: RF.View, duration: float, det: Detector,
                src_size: Tuple[int, int] = (0, 0)) -> List[List[Det]]:
    """Detections for every STEP s of the clip."""
    out = []
    s = det.size
    if view.is360:
        maps = tile_maps(s)
        for eq in _frames(_cmd(path, view, duration, f"scale={EQ_W}:{EQ_H}"), EQ_W, EQ_H):
            tiles = [eq[r, c] for r, c in maps]
            out.append(dets_360(tiles, [det(t) for t in tiles], s))
    else:
        sw, sh = src_size
        if not sw or not sh:
            sw, sh = 16, 9
        k = s / max(sw, sh)
        w, h = max(2, int(sw * k) // 2 * 2), max(2, int(sh * k) // 2 * 2)
        for img in _frames(_cmd(path, view, duration, f"scale={w}:{h}"), w, h):
            out.append(dets_flat(img, det(img)))
    return out


# ---------------------------------------------------------------- tracking
@dataclass
class Track:
    id: int
    cls: int
    samples: Dict[int, Det] = field(default_factory=dict)    # frame index -> detection
    vx: float = 0.0
    vy: float = 0.0

    @property
    def first(self) -> int:
        return min(self.samples)

    @property
    def last(self) -> int:
        return max(self.samples)

    def predict(self, k: int, wrap: bool) -> Tuple[float, float]:
        d = self.samples[self.last]
        n = k - self.last
        x = d.x + self.vx * n
        return (RF._wrap(x) if wrap else x), d.y + self.vy * n

    def add(self, k: int, d: Det, wrap: bool):
        if self.samples:
            p = self.samples[self.last]
            n = max(1, k - self.last)
            vx, vy = _dx(d.x, p.x, wrap) / n, (d.y - p.y) / n
            self.vx, self.vy = 0.5 * self.vx + 0.5 * vx, 0.5 * self.vy + 0.5 * vy
        self.samples[k] = d

    def hist(self) -> np.ndarray:
        hs = [d.hist for d in self.samples.values() if d.hist]
        return np.mean(hs, axis=0) if hs else np.zeros(32)

    def prominence(self, biggest: bool) -> float:
        return sum(d.w * d.h * (1.0 if biggest else d.score) for d in self.samples.values())


def track(frames: Sequence[Sequence[Det]], wrap: bool, step: float = STEP) -> List[Track]:
    """Link detections into tracks: greedy nearest match per class, confident detections first."""
    tracks: List[Track] = []
    active: List[Track] = []
    max_gap = max(1, int(round(MAX_GAP / step)))
    for k, dets in enumerate(frames):
        active = [t for t in active if k - t.last <= max_gap]
        free = list(active)
        for si, stage in enumerate((lambda d: d.score >= HIGH, lambda d: d.score < HIGH)):
            cand = []
            for i, d in enumerate(dets):
                if not stage(d):
                    continue
                for t in free:
                    if t.cls != d.cls:
                        continue
                    px, py = t.predict(k, wrap)
                    g = math.hypot(_dx(d.x, px, wrap), d.y - py) / max(d.size, t.samples[t.last].size, 1e-6)
                    if g < GATE:
                        cand.append((g, i, t))
            used = set()
            for g, i, t in sorted(cand, key=lambda c: c[0]):
                if i in used or t not in free:
                    continue
                t.add(k, dets[i], wrap)
                used.add(i)
                free.remove(t)
            if si == 0:                               # confident leftovers start new tracks
                for i, d in enumerate(dets):
                    if i not in used and d.score >= HIGH:
                        t = Track(len(tracks), d.cls)
                        t.add(k, d, wrap)
                        tracks.append(t)
                        active.append(t)
    return [t for t in tracks if len(t.samples) >= 2]


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def candidates(tracks: Sequence[Track], classes: Optional[List[int]], biggest: bool) -> List[Track]:
    """Tracks that may be the subject, most prominent first."""
    pool = [t for t in tracks if classes is None or t.cls in classes]
    if classes is None and not biggest and any(t.cls == 0 for t in pool):
        pool = [t for t in pool if t.cls == 0]          # auto: people first
    return sorted(pool, key=lambda t: -t.prominence(biggest))


def pick_chain(tracks: Sequence[Track], classes: Optional[List[int]], biggest: bool,
               choose: Optional[Callable[[List[Track]], Optional[Track]]] = None) -> List[Track]:
    """The subject: the chosen (default: most prominent) track, extended by look-alike tracks
    before and after it."""
    pool = candidates(tracks, classes, biggest)
    if not pool:
        return []
    main = (choose(pool) if choose else None) or pool[0]
    ref = main.hist()
    chain = [main]
    while True:          # forward
        nxt = [t for t in pool if t not in chain and t.first > chain[-1].last
               and t.cls == main.cls and similarity(t.hist(), ref) >= REID_SIM]
        if not nxt:
            break
        chain.append(min(nxt, key=lambda t: (t.first, -similarity(t.hist(), ref))))
    while True:          # backward
        prv = [t for t in pool if t not in chain and t.last < chain[0].first
               and t.cls == main.cls and similarity(t.hist(), ref) >= REID_SIM]
        if not prv:
            break
        chain.insert(0, max(prv, key=lambda t: (t.last, similarity(t.hist(), ref))))
    return chain


def aim(d: Det, wrap: bool) -> Tuple[float, float]:
    """Where to point at a detection: a person's upper body rather than their belt."""
    if d.cls != 0:
        return d.x, d.y
    return (d.x, d.y + HEAD_UP * d.h) if wrap else (d.x, d.y - HEAD_UP * d.h)


def subject_path(frames: Sequence[Sequence[Det]], what: str, wrap: bool, step: float = STEP,
                 choose: Optional[Callable[[List[Track]], Optional[Track]]] = None
                 ) -> List[Optional[Tuple[float, float]]]:
    """Per analysed frame, where the subject is (None where it is not found). choose picks the
    subject among the candidate tracks (most prominent first); None = the first."""
    classes = classes_of(what)
    chain = pick_chain(track(frames, wrap, step), classes, what.strip().lower() == "biggest", choose)
    path: List[Optional[Tuple[float, float]]] = [None] * len(frames)
    for t in chain:
        for k, d in t.samples.items():
            path[k] = aim(d, wrap)
    return fill_gaps(path, wrap, max(1, int(round(FILL_GAP / step))))


def fill_gaps(path, wrap: bool, max_gap: int):
    known = [i for i, p in enumerate(path) if p is not None]
    out = list(path)
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= max_gap + 1:
            (xa, ya), (xb, yb) = path[a], path[b]
            dx = _dx(xb, xa, wrap)
            for i in range(a + 1, b):
                f = (i - a) / (b - a)
                x = xa + dx * f
                out[i] = (round(RF._wrap(x) if wrap else x, 3), round(ya + (yb - ya) * f, 3))
    return out


# ---------------------------------------------------------------- cache + entry points
def _cache_file(path: Path, view: RF.View, duration: float, model: str) -> Path:
    st = Path(path).stat()
    key = json.dumps([CACHE_VERSION, str(Path(path).resolve()), st.st_size, st.st_mtime_ns,
                      view.kind, view.lenses, str(view.pair), round(duration, 3), model, STEP,
                      TILES, TILE_FOV])
    return cache_dir() / "subjects" / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")


_MEMO: Dict[str, List[List[Det]]] = {}
_DETECTORS: Dict[Tuple[str, str], Detector] = {}
_PATHS: Dict[tuple, List[Optional[Tuple[float, float]]]] = {}


def clip_detections(path: Path, view: RF.View, duration: float, opts: Options,
                    src_size: Tuple[int, int] = (0, 0)) -> List[List[Det]]:
    cf = _cache_file(path, view, duration, opts.model)
    if str(cf) in _MEMO:
        return _MEMO[str(cf)]
    frames = None
    if cf.exists():
        try:
            frames = [[Det(**d) for d in fr] for fr in json.loads(cf.read_text())["frames"]]
        except (ValueError, KeyError, TypeError):
            frames = None
    if frames is None:
        key = (opts.model, opts.device)
        if key not in _DETECTORS:
            _DETECTORS[key] = Detector(opts.model, opts.device)
        det = _DETECTORS[key]
        print(f"finding subjects in {Path(path).name} ({det.provider.replace('ExecutionProvider', '')})"
              " ...", file=sys.stderr)
        frames = detect_clip(path, view, duration, det, src_size)
        cf.parent.mkdir(parents=True, exist_ok=True)
        cf.write_text(json.dumps({"frames": [[d.__dict__ for d in fr] for fr in frames]}))
    _MEMO[str(cf)] = frames
    return frames


def clip_path(path: Path, view: RF.View, duration: float, opts: Options,
              src_size: Tuple[int, int] = (0, 0)) -> List[Optional[Tuple[float, float]]]:
    memo = (str(_cache_file(path, view, duration, opts.model)), opts.what, opts.desc, opts.vlm)
    if memo in _PATHS:
        return _PATHS[memo]
    frames = clip_detections(path, view, duration, opts, src_size)
    choose = None
    if opts.desc:
        from .describe import chooser
        choose = chooser(path, view, duration, opts, src_size)
    _PATHS[memo] = subject_path(frames, opts.what, view.is360, choose=choose)
    return _PATHS[memo]


def targets(path: Sequence[Optional[Tuple[float, float]]], motion: Sequence[Tuple[float, float, float]],
            step: float = STEP) -> List[Tuple[float, float, float]]:
    """(t, yaw, pitch) camera targets: the subject where found, motion elsewhere."""
    out = [(round(k * step, 3), p[0], float(np.clip(p[1], -RF.MAX_PITCH, RF.MAX_PITCH)))
           for k, p in enumerate(path) if p is not None]
    have = [t for t, _, _ in out]
    for t, y, p in motion:
        if not have or min(abs(t - h) for h in have) > step:
            out.append((t, y, p))
    return sorted(out)


def window(path: Sequence[Optional[Tuple[float, float]]], start: float, length: float,
           step: float = STEP) -> List[Tuple[float, float]]:
    k0, k1 = int(math.floor(start / step)), int(math.ceil((start + length) / step))
    return [p for p in path[max(0, k0):max(0, k1)] if p is not None]


def mean_angle(points: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    a = np.radians([p[0] for p in points])
    yaw = math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean()))
    return round(yaw, 2), round(float(np.mean([p[1] for p in points])), 2)


def shot_view(path, view: RF.View, start: float, length: float) -> Optional[Tuple[float, float]]:
    """360 shot: (yaw, pitch) of the subject over the shot, None if it is not in it."""
    pts = window(path, start, length)
    if not pts:
        return None
    y, p = mean_angle(pts)
    return y, float(np.clip(p, -RF.MAX_PITCH, RF.MAX_PITCH))


def shot_track(path, start: float, length: float) -> Optional[Tuple[float, float, float, float]]:
    """Flat shot: subject centre (x0, y0, x1, y1) over the first and last third, like
    reframe.motion_track; None if the subject is not in the shot."""
    third = length / 3
    a = window(path, start, third) or window(path, start, length)
    b = window(path, start + length - third, third) or a
    if not a:
        return None
    (x0, y0), (x1, y1) = np.mean(a, axis=0), np.mean(b, axis=0)
    return tuple(round(float(np.clip(v, 0, 1)), 4) for v in (x0, y0, x1, y1))
