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

Full control: a keyframe file next to the clip (`<clip>.view.json`, written by `clips frame`)
animates yaw/pitch/fov/roll over the clip. Values ease (cosine) between keyframes, yaw takes
the short way round, and they hold before the first and after the last keyframe. The keyframe
file wins over the auto view; `--yaw` wins over both. Re-solving v360's projection costs
~0.4 s CPU per change, far too slow to do every frame, so the moving view is built from
cheap per-frame steps steered by sendcmd: yaw is a horizontal scroll of the equirectangular
frame (a crop of two copies side by side); pitch, when it moves, is the same scroll on a
copy turned on its side (so the pitch axis is the pole); zoom renders the widest fov of the
shot on a larger canvas and scales it (a narrower rectilinear view is the centre of a wider
one); roll is a rotate of a slightly larger frame. v360 itself never changes mid-shot.

`--fit track` (flat footage): the 9:16 crop window follows the motion centroid of the shot,
panning linearly from where the action is in the first third to where it is in the last third.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, replace
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
EQ_SIZE = (3840, 1920)    # equirect working size for keyframed views (default)
EQ_MAX_W = 5760
ZOOM_MAX = 2.0            # zoom canvas at most this much larger (narrower views upsample)
SIDECAR = ".view.json"
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
    roll: float = 0.0
    keys: Optional[List["Keyframe"]] = None   # animated view (keyframes in source seconds)
    start: float = 0.0                        # source time of the shot's first frame
    cmds: Optional[Path] = None               # sendcmd file for the animated view (render)
    canvas: Optional[Tuple[int, int]] = None  # animated view: v360 size before the roll rotate
    tilt: bool = False                        # animated view: pitch moves (sideways pano)
    zoom: Optional[Tuple[float, float]] = None   # animated view: (widest, narrowest) fov
    eq: Tuple[int, int] = EQ_SIZE             # animated view: equirect working size

    @property
    def is360(self) -> bool:
        return self.kind in ("equirect", "dfisheye")

    @property
    def animated(self) -> bool:
        return bool(self.keys) and len(self.keys) > 1


@dataclass
class Keyframe:
    t: float
    yaw: float = 0.0
    pitch: float = 0.0
    fov: float = H_FOV
    roll: float = 0.0


# ---------------------------------------------------------------- keyframe files
def sidecar_path(path: Path) -> Path:
    return Path(path).with_name(Path(path).name + SIDECAR)


def parse_keyframes(data) -> List[Keyframe]:
    if not isinstance(data, dict) or data.get("version") != 1 or \
            not isinstance(data.get("keyframes"), list):
        raise ClipsError("not a clips view file (expected version 1 with keyframes)")
    keys = []
    for k in data["keyframes"]:
        try:
            kf = Keyframe(float(k["t"]), float(k.get("yaw", 0)), float(k.get("pitch", 0)),
                          float(k.get("fov", H_FOV)), float(k.get("roll", 0)))
        except (KeyError, TypeError, ValueError):
            raise ClipsError(f"bad keyframe {k!r}") from None
        if not (0 < kf.fov < 180) or abs(kf.pitch) > 90 or kf.t < 0:
            raise ClipsError(f"keyframe out of range: {k!r}")
        keys.append(kf)
    return sorted(keys, key=lambda k: k.t)


def load_keyframes(path: Path) -> Optional[List[Keyframe]]:
    """Keyframes from the clip's view file, None when there is none (or it is empty)."""
    f = sidecar_path(path)
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClipsError(f"cannot read {f.name}: {exc}") from None
    try:
        return parse_keyframes(data) or None
    except ClipsError as exc:
        raise ClipsError(f"{f.name}: {exc}") from None


def keyframes_json(keys: Sequence[Keyframe]) -> dict:
    return {"version": 1, "keyframes": [
        {"t": round(k.t, 3), "yaw": round(k.yaw, 2), "pitch": round(k.pitch, 2),
         "fov": round(k.fov, 2), "roll": round(k.roll, 2)} for k in sorted(keys, key=lambda k: k.t)]}


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def interpolate(keys: Sequence[Keyframe], t: float) -> Tuple[float, float, float, float]:
    """(yaw, pitch, fov, roll) at source time t. Same maths as the browser tool."""
    if not keys:
        return 0.0, 0.0, H_FOV, 0.0
    if t <= keys[0].t:
        k = keys[0]
        return _wrap(k.yaw), k.pitch, k.fov, k.roll
    for a, b in zip(keys, keys[1:]):
        if t < b.t:
            u = (t - a.t) / max(b.t - a.t, 1e-9)
            e = 0.5 - 0.5 * math.cos(math.pi * u)
            lerp = lambda x, y: x + (y - x) * e       # noqa: E731
            return (_wrap(a.yaw + _wrap(b.yaw - a.yaw) * e), lerp(a.pitch, b.pitch),
                    lerp(a.fov, b.fov), lerp(a.roll, b.roll))
    k = keys[-1]
    return _wrap(k.yaw), k.pitch, k.fov, k.roll


def keyed_view(view: View, keys: Sequence[Keyframe], start: float) -> View:
    """The keyframed view of a shot starting at source time `start` (its yaw/pitch/fov/roll
    are the values at that time; the render animates it from there)."""
    y, p, f, r = interpolate(keys, start)
    return replace(view, yaw=round(y, 2), pitch=round(p, 2), fov=round(f, 2), roll=round(r, 2),
                   keys=list(keys), start=start)


def still(view: View) -> View:
    """The view frozen at its first frame (no commands)."""
    return replace(view, keys=None, cmds=None, canvas=None, tilt=False, zoom=None)


def eq_size(view: View, width: int, height: int) -> Tuple[int, int]:
    """Equirect working size for an animated view of a source of width x height: about the
    source's own resolution (a lens circle spans ~180 deg), even, 1920..EQ_MAX_W wide."""
    if not width:
        return EQ_SIZE
    w = width if view.kind == "equirect" or view.lenses == "sbs" else 2 * width
    w = int(min(max(w, 1920), EQ_MAX_W)) // 4 * 4
    return w, w // 2


def d_fov(h_fov: float, size: Tuple[int, int], canvas: Optional[Tuple[int, int]] = None) -> float:
    """Diagonal fov of a flat view `canvas` (default `size`) px big with the focal length of
    an h_fov-wide view of `size`."""
    w, h = size
    cw, ch = canvas or size
    focal = (w / 2) / math.tan(math.radians(h_fov) / 2)
    return math.degrees(2 * math.atan(math.hypot(cw, ch) / 2 / focal))


def roll_canvas(size: Tuple[int, int], roll: float) -> Tuple[int, int]:
    """Smallest even canvas that still covers `size` after rotating it by +-roll deg."""
    w, h = size
    r = math.radians(min(abs(roll), 90.0))
    c, s = math.cos(r), math.sin(r)
    return (int(math.ceil((w * c + h * s) / 2 - 1e-9)) * 2,
            int(math.ceil((w * s + h * c) / 2 - 1e-9)) * 2)


def _track(view: View, frames: int, fps: int, size: Tuple[int, int]):
    vals = [interpolate(view.keys or [], view.start + i / fps) for i in range(frames)]
    return vals, max((abs(v[3]) for v in vals), default=0.0)


def _even_up(x: float) -> int:
    return int(math.ceil(x / 2 - 1e-9)) * 2


def zoom_canvas(canvas: Tuple[int, int], zoom: Tuple[float, float]) -> Tuple[int, int]:
    """v360 size for a zooming shot: the widest fov, big enough that the narrowest still
    gets full resolution (at most ZOOM_MAX times the canvas)."""
    k = min(ZOOM_MAX, math.tan(math.radians(zoom[0]) / 2) / math.tan(math.radians(zoom[1]) / 2))
    return _even_up(canvas[0] * k), _even_up(canvas[1] * k)


def zoom_scale(canvas: Tuple[int, int], widest: float, fov: float) -> Tuple[int, int]:
    """Size to scale the zoom canvas to so that its centre canvas-sized crop is `fov` wide."""
    s = math.tan(math.radians(fov) / 2) / math.tan(math.radians(widest) / 2)
    return _even_up(canvas[0] / s), _even_up(canvas[1] / s)


def animate(view: View, frames: int, fps: int, size: Tuple[int, int],
            name: str = "f") -> Tuple[str, View]:
    """(sendcmd script, view to render with) for an animated view over `frames` output frames
    (see view_graph). Times are relative to the shot start; only changes are sent. An empty
    script means the view does not move in this shot (render it still)."""
    vals = [interpolate(view.keys or [], view.start + i / fps) for i in range(frames)]
    max_roll = max((abs(v[3]) for v in vals), default=0.0)
    canvas = roll_canvas(size, max_roll) if max_roll >= 0.01 else None
    tilt = len({round(v[1], 2) for v in vals}) > 1
    fovs = [round(v[2], 2) for v in vals]
    zoom = (max(fovs), min(fovs)) if len(set(fovs)) > 1 else None
    ew = view.eq[0]
    lines = []
    last = None
    for i, (y, p, f, r) in enumerate(vals):
        z = zoom_scale(canvas or size, zoom[0], f) if zoom else None
        cur = (_scroll(y, ew), _scroll(-p, ew), z, round(r, 2))
        if last is not None:
            cmds = []
            if cur[0] != last[0]:
                cmds.append(f"crop@{name}k x {cur[0]}")
            if cur[1] != last[1] and tilt:
                cmds.append(f"crop@{name}p x {cur[1]}")
            if cur[2] != last[2]:
                cmds += [f"scale@{name}z w {z[0]}", f"scale@{name}z h {z[1]}"]
            if cur[3] != last[3]:
                cmds.append(f"rotate@{name}r a {-math.radians(cur[3]):.6f}")
            if cmds:
                lines.append(f"{max(0.0, (i - 0.5) / fps):.6f} " + ", ".join(cmds) + ";")
        last = cur
    return ("".join(l + "\n" for l in lines),
            replace(view, canvas=canvas, tilt=tilt, zoom=zoom))


def _scroll(angle: float, w: int) -> int:
    """Crop offset into two side-by-side equirect copies (w wide) that turns the view by
    `angle` deg about the pole (v360 yaw a looks at column (0.5 + a/360) * w)."""
    return int(round(angle / 360.0 * w)) % w


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
    """360 source -> flat 9:16 view [out] of exactly `size`. An animated view (view.cmds set)
    is unwrapped to equirect, scrolled for yaw and steered by the sendcmd file."""
    w, h = size
    pre, opts = unwrap_graph(view, inp, inp2, out + "u").split("|")
    if view.cmds is None:
        return (f"{pre};[{out}u]fps={fps},v360={opts}:output=flat:h_fov={view.fov:.3f}:"
                f"v_fov={v_fov(view.fov, size):.3f}:yaw={view.yaw:.2f}:pitch={view.pitch:.2f}:"
                f"roll={view.roll:.2f}:w={w}:h={h}:interp=linear[{out}]")
    from .subs import escape_filter_path
    ew, eh = view.eq
    cw, ch = view.canvas or size
    to_eq = (f"scale={ew}:{eh}" if view.kind == "equirect" else
             f"v360={opts}:output=e:w={ew}:h={eh}:interp=linear")
    g = (f"{pre};[{out}u]fps={fps},sendcmd=f={escape_filter_path(view.cmds)},{to_eq},"
         f"split[{out}s0][{out}s1];[{out}s0][{out}s1]hstack=inputs=2,"
         f"crop@{out}k=w={ew}:h={eh}:x={_scroll(view.yaw, ew)}:y=0,")
    if view.tilt:      # on its side: pitch becomes a scroll too, turned back upright below
        g += (f"v360=input=e:output=e:roll=90:w={ew}:h={eh}:interp=linear,"
              f"split[{out}t0][{out}t1];[{out}t0][{out}t1]hstack=inputs=2,"
              f"crop@{out}p=w={ew}:h={eh}:x={_scroll(-view.pitch, ew)}:y=0,")
        orient = "roll=-90"
    else:
        orient = f"pitch={view.pitch:.2f}"
    rot = (f",rotate@{out}r=a={-math.radians(view.roll):.6f}:ow={w}:oh={h}"
           if view.canvas else "")
    if view.zoom is None:
        return g + (f"v360=input=e:output=flat:d_fov={d_fov(view.fov, size, view.canvas):.2f}:"
                    f"{orient}:w={cw}:h={ch}:interp=linear{rot}[{out}]")
    zw, zh = zoom_canvas((cw, ch), view.zoom)
    sw, sh = zoom_scale((cw, ch), view.zoom[0], view.fov)
    return g + (f"v360=input=e:output=flat:d_fov={d_fov(view.zoom[0], size, view.canvas):.3f}:"
                f"{orient}:w={zw}:h={zh}:interp=linear,scale@{out}z=w={sw}:h={sh}:eval=frame[{out}z];"
                f"color=black:s={cw}x{ch}:r={fps}[{out}b];[{out}b][{out}z]overlay="
                f"x='2*trunc((W-w)/4)':y='2*trunc((H-h)/4)':eval=frame:shortest=1{rot}[{out}]")


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
