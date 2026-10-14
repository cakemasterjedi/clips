"""Footage-driven cut timing for edits without a song (--no-music --cuts footage).

Without a song there is no beat to cut on, so the footage sets the rhythm. Every clip is read
small (AFPS frames per second, 128x128 gray) and its own sound is decoded. Per analysis step:

- event (0..1) = how much happens right then: the rise of motion above its level over the
  previous half second, and the clip's sound onsets (librosa onset strength), each scaled by its
  95th percentile over all clips and averaged (motion only for clips without sound). A hard cut
  inside the clip (block diff >= footage.CUT_DIFF) counts as an event of 1.
- activity (0..1) = motion and loudness, smoothed over a second. It sets the shot length the
  same way song energy does: max_shot when calm, down to FAST_SHOT when busy.

Shots walk through the clips in order (shuffled with --shuffle/--seed, like the song mode), each
continuing where that clip left off. A shot ends TAIL after the strongest event near its target
length, so the action or sound lands inside the shot; right before a hard cut in the clip, so a
shot never contains a cut of its own; or at the target when nothing happens. A single clip skips
footage between shots, so every cut is a visible jump. The event a shot ended on is the next
shot's hit strength, which drives --punch, --flash and --ramp as beat hits do with a song.
"""
from __future__ import annotations

import random
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import audio as A
from .ffmpeg import cpu_count, probe
from .footage import CUT_DIFF, frame_features, read_small
from .sources import Assignment, Source, _usable

AFPS = 10                 # analysis steps per second
STEP = 1.0 / AFPS
FAST_SHOT = 0.6           # target shot length (s) at full activity
TAIL = 0.2                # s kept after the event a shot ends on
MIN_EVENT = 0.15          # weaker than this = nothing happens, cut at the target length
NEAR = 0.35               # how much an event loses per target length of distance
RISE_WIN = 5              # steps of motion history an event must rise above


@dataclass
class ClipEvents:
    event: np.ndarray     # per step, 0..1
    activity: np.ndarray  # per step, 0..1
    cut: np.ndarray       # per step, True = hard cut between step i-1 and i


def _bin(x: np.ndarray, times: np.ndarray, n: int, how: str) -> np.ndarray:
    idx = np.clip((times * AFPS).astype(int), 0, max(n - 1, 0))
    out = np.zeros(n)
    if how == "max":
        np.maximum.at(out, idx, x)
        return out
    cnt = np.zeros(n)
    np.add.at(out, idx, x)
    np.add.at(cnt, idx, 1)
    return out / np.maximum(cnt, 1)


def raw_features(src: Source) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """(motion, sound onsets, loudness) per step; the sound ones are None for silent clips."""
    import librosa

    sys.stderr.write(f"analysing {src.path.name}...\n")
    sys.stderr.flush()
    motion, _, _ = frame_features(read_small(src.path, AFPS))
    n = motion.size
    if n == 0 or not (src.info or probe(src.path)).has_audio:
        return motion, None, None
    y = A.decode(src.path)
    if y.size < A.HOP * 2:
        return motion, None, None
    env = librosa.onset.onset_strength(y=y, sr=A.SR, hop_length=A.HOP)
    rms = librosa.feature.rms(y=y, hop_length=A.HOP)[0]
    t_env = librosa.frames_to_time(np.arange(env.size), sr=A.SR, hop_length=A.HOP)
    t_rms = librosa.frames_to_time(np.arange(rms.size), sr=A.SR, hop_length=A.HOP)
    return motion, _bin(env, t_env, n, "max"), _bin(rms, t_rms, n, "mean")


def _p95(parts: Sequence[np.ndarray]) -> float:
    x = np.concatenate([p for p in parts if p is not None and p.size] or [np.zeros(1)])
    return max(float(np.percentile(x, 95)), 1e-6)


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    if x.size == 0 or k <= 1:
        return x
    pad = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def clip_events(raws) -> List[ClipEvents]:
    """Event and activity curves from raw_features() of every clip, scaled together so clips
    are comparable."""
    cuts, calm, rises = [], [], []
    for m, _, _ in raws:
        cut = m >= CUT_DIFF
        mm = np.where(cut, 0.0, m)
        hist = _smooth(mm, RISE_WIN)
        prev = np.concatenate([[mm[0]], hist[:-1]]) if mm.size else mm
        cuts.append(cut)
        calm.append(mm)
        rises.append(np.maximum(mm - prev, 0.0))
    pr, pm = _p95(rises), _p95(calm)
    po, pl = _p95([o for _, o, _ in raws]), _p95([r for _, _, r in raws])
    out = []
    for (m, onset, loud), cut, mm, rise in zip(raws, cuts, calm, rises):
        vis = np.clip(rise / pr, 0, 1)
        mov = np.clip(mm / pm, 0, 1)
        if onset is None:
            ev, act = vis, mov
        else:
            ev = 0.5 * vis + 0.5 * np.clip(onset / po, 0, 1)
            act = 0.6 * mov + 0.4 * np.clip(loud / pl, 0, 1)
        ev = ev.copy()
        ev[cut] = 1.0
        out.append(ClipEvents(ev, _smooth(act, AFPS), cut))
    return out


def analyse(sources: Sequence[Source]) -> List[ClipEvents]:
    workers = max(1, min(len(sources), cpu_count() // 2))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        raws = list(ex.map(raw_features, sources))
    return clip_events(raws)


def target_length(activity: float, max_shot: float) -> float:
    return FAST_SHOT + (max_shot - FAST_SHOT) * (1.0 - float(np.clip(activity, 0, 1))) ** 1.5


def mean_between(x: np.ndarray, a: float, b: float) -> float:
    i, j = int(a * AFPS), max(int(a * AFPS) + 1, int(np.ceil(b * AFPS)))
    seg = x[i:j]
    return float(seg.mean()) if seg.size else (float(x[-1]) if x.size else 0.0)


def pick_length(ce: ClipEvents, start: float, target: float,
                room: float) -> Tuple[float, float, Optional[float]]:
    """(shot length, strength of the event it ends on, where the clip should resume) for a shot
    starting at `start` in a clip with `room` seconds left: the best event in
    [target/2, 1.6 x target], scored by strength and closeness to the target; the target itself
    (strength 0) when nothing happens there. Resume is the time of a hard cut the shot stopped
    in front of (so the clip's next shot starts on the new scene), else None."""
    lo = max(A.MIN_SHOT, 0.5 * target)
    hi = min(1.6 * target, A.MAX_SHOT, room)
    if hi < lo:
        return max(min(target, room), A.MIN_SHOT), 0.0, None
    best = None
    first = max(int(np.floor((start + lo - TAIL) * AFPS)), 1)
    last = min(int(np.ceil((start + hi) * AFPS)), ce.event.size - 1)
    for i in range(first, last + 1):
        t = i * STEP
        ln = (t - STEP - start) if ce.cut[i] else (t + TAIL - start)
        if not lo - 1e-9 <= ln <= hi + 1e-9:
            continue
        ev = float(ce.event[i])
        score = ev * (1.0 - NEAR * abs(ln - target) / target)
        if best is None or score > best[0]:
            best = (score, ln, ev, t if ce.cut[i] else None)
    if best is None or best[2] < MIN_EVENT:
        return float(np.clip(target, lo, hi)), 0.0, None
    return best[1], best[2], best[3]


def plan_shots(events: Sequence[ClipEvents], sources: Sequence[Source], length: float, fps: int,
               max_shot: float = A.AUTO_MAX_SHOT, shuffle: bool = False,
               seed: Optional[int] = None) -> List[Tuple[int, Assignment, float, float]]:
    """[(frames, assignment, hit, energy)] covering exactly round(length * fps) frames."""
    total = int(round(length * fps))
    min_f = int(np.ceil(A.MIN_SHOT * fps - 1e-9))
    n = len(sources)
    order = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(order)
    usable = [_usable(s.duration) for s in sources]
    longest = max(range(n), key=lambda i: usable[i])
    cursors = [0.0] * n
    gap = 0.0
    if n == 1:          # one clip: spread the shots over it so every cut jumps
        est = max(1.0, length / target_length(float(events[0].activity.mean())
                                              if events[0].activity.size else 0.0, max_shot))
        gap = max(0.0, usable[0] - length) / est
    out: List[Tuple[int, Assignment, float, float]] = []
    pos, nxt, hit = 0, 0, 0.0
    while pos < total:
        remaining = total - pos
        ci = None
        for k in range(n):
            c = order[(nxt + k) % n]
            if usable[c] >= A.MIN_SHOT:
                ci, nxt = c, (nxt + k + 1) % n
                break
        if ci is None:                       # every clip is shorter than a shot: loop one
            out.append((remaining, Assignment(longest, 0.0, True), hit, 0.0))
            break
        if cursors[ci] + A.MIN_SHOT > usable[ci]:
            cursors[ci] = 0.0
        st = cursors[ci]
        ce = events[ci]
        energy = mean_between(ce.activity, st, st + max_shot)
        ln, ev, resume = pick_length(ce, st, target_length(energy, max_shot), usable[ci] - st)
        f = min(max(min_f, int(round(ln * fps))), remaining)
        if remaining - f < min_f:
            f = remaining                    # never leave a sliver at the end
        out.append((f, Assignment(ci, round(st, 6), f / fps > usable[ci] - st + 1e-6),
                    hit, round(energy, 3)))
        hit = ev
        cursors[ci] = max(st + f / fps, resume or 0.0) + gap
        pos += f
    return out


def footage_plan(sources: Sequence[Source], length: float, fps: int,
                 max_shot: float = A.AUTO_MAX_SHOT, shuffle: bool = False,
                 seed: Optional[int] = None, punch: bool = False, flash: bool = False,
                 punch_frac: float = A.PUNCH_FRAC, flash_frac: float = A.FLASH_FRAC,
                 ramp: bool = False, events: Optional[Sequence[ClipEvents]] = None
                 ) -> Tuple[A.Plan, List[Assignment]]:
    """Plan + assignments for an edit without a song, cut on what happens in the footage."""
    events = events if events is not None else analyse(sources)
    rows = plan_shots(events, sources, length, fps, max_shot, shuffle, seed)
    shots, pos = [], 0
    for i, (f, _, h, e) in enumerate(rows):
        shots.append(A.Shot(i, pos, f, fps, e, round(h, 3)))
        pos += f
    pu, fl = A.select_fx([s.hit for s in shots], punch_frac, flash_frac)
    for sh, p, q in zip(shots, pu, fl):
        sh.punch, sh.flash = bool(punch and p), bool(flash and q)
    if ramp:
        for sh, r in zip(shots, A.select_ramps([s.hit for s in shots], [s.energy for s in shots],
                                               [s.length for s in shots])):
            sh.ramp = r
    return A.Plan(pos / fps, fps, "footage", 0.0, shots), [a for _, a, _, _ in rows]
