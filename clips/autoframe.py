"""Auto framing: a camera operator for 360 clips that frames like a learned creator.

The whole clip is unwrapped to a small equirectangular strip (reframe.strip_frames) and the
action is located every half second (most motion per longitude, as in the per-shot auto
view). A virtual camera then follows it the way the creator's own videos move (the style
profile's `framing`, see style.py):

- `moving` (share of the creator's shots with camera movement) sets a dead zone: a creator
  who rarely moves the camera only re-aims when the action leaves the middle of the frame;
  one who always moves keeps the action centred.
- `pan_speed` (frame widths per second) caps how fast the camera turns.
- `push` (zoom per second) narrows the view slowly while the camera holds still, as a slow
  push-in; it relaxes back when the camera moves.
- `subject_y` (where the action sits vertically in their frames) offsets the pitch so the
  action lands at the same height.

The path is written as keyframes (`<clip>.view.json`, the same file `clips frame` edits), so
the result can be opened in the framing tool and adjusted by hand. Existing keyframe files
are never overwritten unless asked (`clips autoframe --force`).

Flat footage with `--fit track` gets the same treatment in a simpler form: the crop follows
the action only for creators who move the camera, and places it at their `subject_y`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import reframe as RF
from .ffmpeg import ClipsError

STEP = 0.5                # s between camera samples
WINDOW = 1.0              # s of motion looked at for each sample
MIN_SPEED = 10.0          # deg/s: even a "still" creator's camera can catch up eventually
FOLLOW = 2.0              # 1/s: how quickly the camera closes the gap (before the speed cap)
PUSH_MAX = 0.75           # a push-in narrows the view to at most this share of the fov
RELAX = 0.4               # 1/s: how quickly the fov returns to normal while moving
TOL_ANGLE = 1.5           # deg: keyframes are dropped while the eased path stays this close
TOL_FOV = 1.0
DEFAULT = {"moving": 0.5, "pan_speed": 0.15, "push_share": 0.0, "push": 0.0, "subject_y": 0.5}


def framing_of(profile: Optional[dict]) -> dict:
    f = dict(DEFAULT)
    f.update({k: v for k, v in ((profile or {}).get("framing") or {}).items() if k in DEFAULT})
    return f


# ---------------------------------------------------------------- targets
def motion_targets(frames: np.ndarray, seams: bool, fps: float = RF.AFPS,
                   step: float = STEP, window: float = WINDOW) -> List[Tuple[float, float, float]]:
    """(t, yaw, pitch) of the action every `step` s; samples with no motion are left out."""
    out = []
    n = len(frames)
    w = max(2, int(round(window * fps)))
    t = 0.0
    while True:
        i0 = int(round(t * fps))
        if i0 >= n - 1:
            break
        tgt = RF.motion_target(frames[i0:i0 + w], seams)
        if tgt is not None:
            out.append((round(t + min(w, n - i0) / fps / 2, 3), tgt[0], tgt[1]))
        t += step
    return out


# ---------------------------------------------------------------- camera
def camera_path(targets: Sequence[Tuple[float, float, float]], duration: float, framing: dict,
                fov: float = RF.H_FOV, size: Tuple[int, int] = (1080, 1920),
                dt: float = STEP) -> List[RF.Keyframe]:
    """Simulated camera, one sample every dt s from 0 to duration."""
    vf = RF.v_fov(fov, size)
    dead = fov * (0.35 - 0.3 * min(max(framing["moving"], 0.0), 1.0))   # deg from centre
    vmax = max(MIN_SPEED, framing["pan_speed"] * fov)                   # deg/s
    push = max(0.0, framing["push"]) if framing.get("push_share", 0) > 0 else 0.0
    offset = (framing["subject_y"] - 0.5) * vf      # look higher when the action sits low
    ts = np.array([t for t, _, _ in targets])
    if not len(ts):
        return [RF.Keyframe(0.0, 0.0, 0.0, fov, 0.0)]

    def target(t):        # nearest earlier target (the camera reacts, it does not predict)
        i = int(np.searchsorted(ts, t + 1e-9)) - 1
        _, y, p = targets[max(i, 0)]
        return y, float(np.clip(p + offset, -RF.MAX_PITCH, RF.MAX_PITCH))

    yaw, pitch = target(0.0)
    f = fov
    keys = []
    moving = False
    for k in range(int(math.floor(duration / dt + 1e-9)) + 1):
        t = k * dt
        if k == 0:
            keys.append(RF.Keyframe(0.0, round(yaw, 2), round(pitch, 2), round(f, 2), 0.0))
            continue
        ty, tp = target(t)
        ey, ep = RF._wrap(ty - yaw), tp - pitch
        gap = math.hypot(ey, ep)
        if gap > dead or (moving and gap > dead / 3):
            moving = True
            mv = min(gap * (1 - math.exp(-FOLLOW * dt)), vmax * dt)
            yaw = RF._wrap(yaw + ey / gap * mv)
            pitch += ep / gap * mv
            f += (fov - f) * (1 - math.exp(-RELAX * dt))
        else:
            moving = False
            f = max(fov * PUSH_MAX, f * math.exp(-push * dt))
        keys.append(RF.Keyframe(round(t, 3), round(yaw, 2), round(pitch, 2), round(f, 2), 0.0))
    return keys


def simplify(keys: Sequence[RF.Keyframe]) -> List[RF.Keyframe]:
    """Fewest keyframes whose eased interpolation stays within TOL of every sample."""
    keys = list(keys)
    if len(keys) <= 2:
        return keys
    out = [keys[0]]
    i = 0
    while i < len(keys) - 1:
        j = i + 1
        while j + 1 < len(keys) and _fits(out[-1], keys[j + 1], keys[i + 1:j + 1]):
            j += 1
        out.append(keys[j])
        i = j
    return out


def _fits(a: RF.Keyframe, b: RF.Keyframe, mid: Sequence[RF.Keyframe]) -> bool:
    for k in mid:
        y, p, f, _ = RF.interpolate([a, b], k.t)
        if (abs(RF._wrap(y - k.yaw)) > TOL_ANGLE or abs(p - k.pitch) > TOL_ANGLE
                or abs(f - k.fov) > TOL_FOV):
            return False
    return True


# ---------------------------------------------------------------- clips
def auto_keyframes(path: Path, view: RF.View, duration: float, framing: dict,
                   fov: float = RF.H_FOV) -> List[RF.Keyframe]:
    frames = RF.strip_frames(path, view, 0.0, duration)
    tg = motion_targets(frames, view.kind == "dfisheye")
    return simplify(camera_path(tg, duration, framing, fov))


def write_keyframes(path: Path, keys: Sequence[RF.Keyframe], force: bool = False) -> bool:
    """Write the clip's keyframe file; False (untouched) if one exists and not force."""
    side = RF.sidecar_path(path)
    if side.exists() and not force:
        return False
    side.write_text(json.dumps(RF.keyframes_json(keys), indent=1) + "\n", encoding="utf-8")
    return True


def flat_track(track: Tuple[float, float, float, float], framing: dict
               ) -> Tuple[Tuple[float, float, float, float], float]:
    """(track, subject height) for a flat --fit track crop: hold still (mean position) for
    creators who rarely move the camera."""
    x0, y0, x1, y1 = track
    if framing["moving"] < 0.3:
        x0 = x1 = round((x0 + x1) / 2, 4)
        y0 = y1 = round((y0 + y1) / 2, 4)
    return (x0, y0, x1, y1), float(np.clip(framing["subject_y"], 0.2, 0.8))


def run_autoframe(args) -> int:
    import sys
    from .ffmpeg import probe
    from .sources import discover
    from . import style as style_mod
    prof = style_mod.load_profile(args.style) if args.style else None
    fr = framing_of(prof)
    if args.fov <= 0 or args.fov >= 180:
        raise ClipsError("--fov must be between 0 and 180")
    done = 0
    for src in discover(Path(args.input)):
        info = src.info or probe(src.path)
        view = RF.detect_view(src.path, info, args.mode360)
        if not view.is360:
            continue
        if RF.sidecar_path(src.path).exists() and not args.force:
            print(f"{src.path.name}: keeping existing {RF.sidecar_path(src.path).name} "
                  "(--force to replace)", file=sys.stderr)
            continue
        keys = auto_keyframes(src.path, view, src.duration, fr, args.fov)
        write_keyframes(src.path, keys, force=True)
        done += 1
        print(f"{src.path.name}: {len(keys)} keyframes -> {RF.sidecar_path(src.path).name}",
              file=sys.stderr)
    if not done:
        print("no keyframe files written", file=sys.stderr)
    return 0
