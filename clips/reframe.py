"""Reframing: 360 footage (Insta360 .insv and equirectangular MP4) and motion-tracked crop.

360 sources are turned into a flat 9:16 view with FFmpeg's v360 filter: equirectangular input
as is, Insta360 dual fisheye (.insv) stitched with a generic fisheye model. Native .insv comes
in three layouts: both lenses side by side in one stream, one stream per lens, or one file per
lens (VID_..._00_NNN.insv + VID_..._10_NNN.insv). There is no lens calibration, seam blending,
gyro stabilization or horizon levelling (those need Insta360's own software), so the auto view
keeps away from the lens seams at yaw +-90.

The view direction for each shot follows motion: the shot's window is unwrapped to a small
equirectangular strip, frame differences are summed per longitude, and the yaw with the most
motion wins (pitch from where that motion sits vertically, capped at +-30 deg). `--yaw/--pitch`
fix the view instead.

`--fit track` (flat footage): the 9:16 crop window follows the motion centroid of the shot,
panning linearly from where the action is in the first third to where it is in the last third.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg import ClipsError, ffmpeg_exe

FISHEYE_FOV = 200.0       # Insta360 X-series lens field of view (deg), per lens
H_FOV = 60.0              # default horizontal field of view of the 9:16 view
SEAM_WIDTH = 25.0         # deg around yaw +-90 the auto view avoids on dual fisheye
SEAM_WEIGHT = 0.3
MAX_PITCH = 30.0
AW, AH = 256, 128         # analysis strip (equirectangular)
AFPS = 6
TRACK_W = 160             # analysis width for tracked crop
PAIR_RE = re.compile(r"_00_(\d+)\.insv$", re.I)


@dataclass
class View:
    kind: str = "flat"                    # flat | equirect | dfisheye
    lenses: Optional[str] = None          # dfisheye: sbs | streams | pair
    pair: Optional[Path] = None           # dfisheye pair: the _10_ file
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = H_FOV
    track: Optional[Tuple[float, float, float, float]] = None   # flat: x0, y0, x1, y1 (0..1)

    @property
    def is360(self) -> bool:
        return self.kind in ("equirect", "dfisheye")


# ---------------------------------------------------------------- detection
def pair_of(path: Path) -> Optional[Path]:
    """The _10_ lens file belonging to a _00_ Insta360 file, if it exists."""
    m = PAIR_RE.search(path.name)
    if not m:
        return None
    other = path.with_name(path.name[: m.start()] + f"_10_{m.group(1)}" + path.suffix)
    return other if other.exists() else None


def is_pair_secondary(path: Path) -> bool:
    """A _10_ file whose _00_ partner exists (it is read together with the partner)."""
    m = re.search(r"_10_(\d+)\.insv$", path.name, re.I)
    return bool(m) and path.with_name(path.name[: m.start()] + f"_00_{m.group(1)}"
                                      + path.suffix).exists()


def detect_view(path: Path, info, mode: str = "auto") -> View:
    """View kind for a source. mode: auto | off | equirect | dfisheye."""
    path = Path(path)
    if mode == "off":
        return View()
    insv = path.suffix.lower() == ".insv"
    if mode == "dfisheye" or (mode == "auto" and insv):
        if getattr(info, "video_streams", 1) >= 2:
            return View("dfisheye", "streams")
        pair = pair_of(path)
        if pair is not None:
            return View("dfisheye", "pair", pair)
        return View("dfisheye", "sbs")
    if mode == "equirect":
        return View("equirect")
    w, h = info.width or 0, info.height or 0
    if getattr(info, "spherical", False) or (w >= 3840 and h and abs(w / h - 2.0) < 0.01):
        return View("equirect")
    return View()


# ---------------------------------------------------------------- filter graphs
def v_fov(h_fov: float, size: Tuple[int, int]) -> float:
    w, h = size
    return float(np.degrees(2 * np.arctan(np.tan(np.radians(h_fov) / 2) * h / w)))


def input_labels(view: View, idx: int) -> Tuple[str, Optional[str]]:
    """(label, second-lens label or None) for a source whose first input file is #idx
    (a pair source also uses input #idx+1)."""
    if view.kind == "dfisheye" and view.lenses == "streams":
        return f"{idx}:v:0", f"{idx}:v:1"
    if view.kind == "dfisheye" and view.lenses == "pair":
        return f"{idx}:v", f"{idx + 1}:v"
    return f"{idx}:v", None


def unwrap_graph(view: View, inp: str, inp2: Optional[str], out: str) -> str:
    """[inp](+[inp2]) -> [out] as one equirectangular-or-dual-fisheye stream plus v360 input
    options, returned as 'graph|v360-input-options'."""
    if view.kind == "equirect":
        return f"[{inp}]null[{out}]|input=e"
    pre = f"[{inp}][{inp2}]hstack=inputs=2[{out}]" if inp2 else f"[{inp}]null[{out}]"
    return f"{pre}|input=dfisheye:ih_fov={FISHEYE_FOV:g}:iv_fov={FISHEYE_FOV:g}"


def view_graph(view: View, fps: int, inp: str, inp2: Optional[str], out: str,
               size: Tuple[int, int]) -> str:
    """360 source -> flat 9:16 view [out] of exactly `size`."""
    w, h = size
    pre, opts = unwrap_graph(view, inp, inp2, out + "u").split("|")
    return (f"{pre};[{out}u]fps={fps},v360={opts}:output=flat:h_fov={view.fov:.3f}:"
            f"v_fov={v_fov(view.fov, size):.3f}:yaw={view.yaw:.2f}:pitch={view.pitch:.2f}:"
            f"w={w}:h={h}:interp=linear[{out}]")


def track_crop(view: Optional[View], length: float, size: Tuple[int, int]) -> str:
    """crop=W:H:x:y with a linear pan between the tracked centroids (centre if untracked)."""
    w, h = size
    if view is None or view.track is None:
        return f"crop={w}:{h}"
    x0, y0, x1, y1 = view.track
    L = max(length, 1e-3)

    def axis(c0, c1, full, win):
        return (f"'max(0,min({full}-{win},({c0:.4f}+({c1 - c0:.4f})*min(t/{L:.4f},1))*{full}"
                f"-{win}/2))'")

    return f"crop={w}:{h}:x={axis(x0, x1, 'iw', w)}:y={axis(y0, y1, 'ih', h)}"


# ---------------------------------------------------------------- analysis
def _read(args: Sequence[str], w: int, h: int) -> np.ndarray:
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *args,
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ClipsError(f"cannot analyse video: {msg[-1] if msg else 'ffmpeg failed'}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (w * h)
    return buf[: n * w * h].reshape(n, h, w).astype(np.float32)


def window_inputs(path: Path, view: View, start: float, length: float) -> List[str]:
    ins = ["-ss", f"{start:.6f}", "-t", f"{length:.6f}", "-i", str(path)]
    if view.kind == "dfisheye" and view.lenses == "pair" and view.pair is not None:
        ins += ["-ss", f"{start:.6f}", "-t", f"{length:.6f}", "-i", str(view.pair)]
    return ins


def strip_frames(path: Path, view: View, start: float, length: float) -> np.ndarray:
    """Small equirectangular gray frames of the window (AW x AH at AFPS)."""
    inp, inp2 = input_labels(view, 0)
    pre, opts = unwrap_graph(view, inp, inp2, "u").split("|")
    conv = "" if view.kind == "equirect" else f"v360={opts}:output=e,"
    fc = f"{pre};[u]fps={AFPS},{conv}scale={AW}:{AH}[o]"
    return _read(window_inputs(path, view, start, length) + ["-filter_complex", fc, "-map", "[o]"],
                 AW, AH)


def yaw_pitch_from_motion(frames: np.ndarray, seams: bool) -> Tuple[float, float]:
    """Yaw/pitch (deg) of the strongest motion in equirectangular gray frames."""
    if len(frames) < 2:
        return 0.0, 0.0
    d = np.abs(np.diff(frames, axis=0)).sum(axis=0)          # (AH, AW)
    lat = (0.5 - (np.arange(AH) + 0.5) / AH) * 180.0
    band = np.abs(lat) <= 60
    col = d[band].sum(axis=0)
    k = max(1, int(AW * 30 / 360))                            # 30 deg circular smoothing
    kern = np.hanning(2 * k + 3)[1:-1]                        # peaked, so the max is centred
    col = np.convolve(np.concatenate([col[-k:], col, col[:k]]), kern, "same")[k:-k]
    yaws = (np.arange(AW) + 0.5) / AW * 360.0 - 180.0
    if seams:
        near = np.minimum(np.abs(yaws - 90), np.abs(yaws + 90)) < SEAM_WIDTH
        col = np.where(near, col * SEAM_WEIGHT, col)
    if col.max() <= 1e-6 * max(1.0, d.size):
        return 0.0, 0.0
    j = int(np.argmax(col))
    lo, hi = max(0, j - k), min(AW, j + k + 1)
    rows = d[:, lo:hi].sum(axis=1)
    pitch = float((rows * lat).sum() / rows.sum()) if rows.sum() > 0 else 0.0
    return round(float(yaws[j]), 2), round(float(np.clip(pitch, -MAX_PITCH, MAX_PITCH)), 2)


def auto_view(path: Path, view: View, start: float, length: float) -> View:
    y, p = yaw_pitch_from_motion(strip_frames(path, view, start, length), view.kind == "dfisheye")
    return View(view.kind, view.lenses, view.pair, y, p, view.fov)


def motion_track(frames: np.ndarray) -> Tuple[float, float, float, float]:
    """(x0, y0, x1, y1): motion centroid (0..1) over the first and last third of the frames;
    the centre when there is no motion."""
    if len(frames) < 2:
        return 0.5, 0.5, 0.5, 0.5
    d = np.abs(np.diff(frames, axis=0))
    d = np.where(d > 12, d, 0)                                # ignore noise / compression
    h, w = d.shape[1:]
    xs, ys = (np.arange(w) + 0.5) / w, (np.arange(h) + 0.5) / h

    def centroid(block):
        m = block.sum(axis=0)
        s = m.sum()
        if s <= 0:
            return None
        return float((m.sum(axis=0) * xs).sum() / s), float((m.sum(axis=1) * ys).sum() / s)

    n = len(d)
    a = centroid(d[: max(1, n // 3)]) or centroid(d) or (0.5, 0.5)
    b = centroid(d[n - max(1, n // 3):]) or a
    return round(a[0], 4), round(a[1], 4), round(b[0], 4), round(b[1], 4)


def track_view(path: Path, start: float, length: float, src_w: int, src_h: int) -> View:
    th = max(2, int(round(TRACK_W * src_h / max(src_w, 1) / 2)) * 2)
    frames = _read(["-ss", f"{start:.6f}", "-t", f"{length:.6f}", "-i", str(path), "-an",
                    "-vf", f"fps={AFPS * 2},scale={TRACK_W}:{th}"], TRACK_W, th)
    return View(track=motion_track(frames))
