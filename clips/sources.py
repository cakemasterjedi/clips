"""Input discovery and assignment of shots to source windows."""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .ffmpeg import ClipsError, probe

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".insv"}
END_MARGIN = 0.04  # seconds kept free at the end of a source so -frames:v never runs dry


@dataclass
class Source:
    path: Path
    duration: float
    width: int = 0
    height: int = 0
    info: object = None       # ProbeInfo, for 360 detection


@dataclass
class Assignment:
    source: int          # index into the sources list
    start: float
    loop: bool = False


def discover(inp) -> List[Source]:
    p = Path(inp)
    if p.is_dir():
        from .reframe import is_pair_secondary

        # an Insta360 _10_ lens file is read together with its _00_ partner, not on its own
        files = sorted((f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTS
                        and not is_pair_secondary(f)), key=lambda f: f.name)
        if not files:
            raise ClipsError(f"no video files ({', '.join(sorted(VIDEO_EXTS))}) in {p}")
    elif p.is_file():
        files = [p]
    else:
        raise ClipsError(f"input not found: {p}")
    out = []
    for f in files:
        info = probe(f)
        if not info.has_video or not info.duration or info.duration <= 0:
            print(f"warning: skipping {f.name}: no usable video stream", file=sys.stderr)
            continue
        out.append(Source(f.resolve(), info.duration, info.width or 0, info.height or 0, info))
    if not out:
        raise ClipsError(f"no usable video in {p}")
    return out


def _usable(d: float) -> float:
    return max(d - END_MARGIN, 0.0)


def assign_directory(shot_lens: Sequence[float], durations: Sequence[float],
                     shuffle: bool = False, seed: Optional[int] = None,
                     start_at: Optional[Sequence[float]] = None) -> List[Assignment]:
    """start_at[i]: initial cursor for clip i (e.g. end of the dialogue intro taken from it),
    so the music section doesn't replay footage the intro just showed."""
    n = len(durations)
    order = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(order)
    cursors = [0.0] * n
    if start_at:
        cursors = [float(c) if c < _usable(d) - 0.25 else 0.0 for c, d in zip(start_at, durations)]
    longest = max(range(n), key=lambda i: durations[i])
    nxt = 0
    if start_at and any(c > 0 for c in start_at) and n > 1:
        # open the music section on a different clip than the intro, so the music hit is a visible cut
        last_intro = max(range(n), key=lambda i: start_at[i])
        nxt = (order.index(last_intro) + 1) % n
    res: List[Assignment] = []
    for ln in shot_lens:
        chosen = None
        for k in range(n):
            pos = (nxt + k) % n
            ci = order[pos]
            if _usable(durations[ci]) + 1e-9 >= ln:
                chosen, nxt = ci, pos + 1
                break
        if chosen is None:
            res.append(Assignment(longest, 0.0, True))
            nxt = (nxt + 1) % n
            continue
        if cursors[chosen] + ln > _usable(durations[chosen]) + 1e-9:
            cursors[chosen] = 0.0
        res.append(Assignment(chosen, round(cursors[chosen], 6)))
        cursors[chosen] += ln
    return res


def assign_single(shot_lens: Sequence[float], duration: float,
                  shuffle: bool = False, seed: Optional[int] = None,
                  start_at: float = 0.0) -> List[Assignment]:
    """start_at: skip footage before this time when enough remains (sequential mode only)."""
    usable = _usable(duration)
    total = sum(shot_lens)
    rng = random.Random(seed)
    if shuffle:
        if usable >= total:
            idx = list(range(len(shot_lens)))
            rng.shuffle(idx)
            cuts = sorted(rng.uniform(0, usable - total) for _ in idx)
            res = [None] * len(shot_lens)
            pos = 0.0
            prev_gap = 0.0
            for j, si in enumerate(idx):
                pos += cuts[j] - prev_gap
                prev_gap = cuts[j]
                res[si] = Assignment(0, round(pos, 6))
                pos += shot_lens[si]
            return res
        return [Assignment(0, round(rng.uniform(0, usable - ln), 6), False) if usable >= ln
                else Assignment(0, 0.0, True) for ln in shot_lens]
    res = []
    off = start_at if usable - start_at >= total else 0.0
    if usable - off >= total:
        ratio = (usable - off) / total if total > 0 else 1.0
        cum = 0.0
        for ln in shot_lens:
            res.append(Assignment(0, round(off + min(cum * ratio, usable - off - ln), 6)))
            cum += ln
        return res
    cursor = 0.0
    for ln in shot_lens:
        if ln > usable + 1e-9:
            res.append(Assignment(0, 0.0, True))
            continue
        if cursor + ln > usable + 1e-9:
            cursor = 0.0
        res.append(Assignment(0, round(cursor, 6)))
        cursor += ln
    return res


def assign(shot_lens: Sequence[float], sources: Sequence[Source], shuffle=False, seed=None,
           start_at: Optional[Sequence[float]] = None):
    if len(sources) == 1:
        return assign_single(shot_lens, sources[0].duration, shuffle, seed,
                             start_at[0] if start_at else 0.0)
    return assign_directory(shot_lens, [s.duration for s in sources], shuffle, seed, start_at)


def first_source_index(n: int, shuffle: bool = False, seed: Optional[int] = None) -> int:
    """Index of the clip the music section would use first (mirrors assign_directory)."""
    order = list(range(n))
    if shuffle:
        random.Random(seed).shuffle(order)
    return order[0]
