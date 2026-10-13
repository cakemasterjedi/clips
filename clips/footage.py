"""Smart footage picking: score every clip over time and give the best moments to the shots
that need them most (the opening shot, then the strongest beat hits).

Per analysis frame (5 fps, 128x128 gray): motion = block-averaged frame difference, sharpness =
mean gradient. Frames that are near black or nearly featureless score -1, and so do frames on a
hard cut inside the clip (a shot should not contain a cut of its own). The rest score
0.6 * motion + 0.4 * sharpness, both scaled by their 95th percentile over all clips so clips are
comparable. Windows never overlap and keep MIN_GAP apart within a clip, and neighbouring shots
come from different clips when there is more than one, so every cut is visible.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg import ClipsError, ffmpeg_exe
from .sources import Assignment, Source, _usable

SFPS = 5                  # analysis frames per second
SIZE = 128
DARK = 15.0               # mean luma below this = unusable
FLAT = 1.0                # mean gradient below this = blank / heavily blurred
CUT_DIFF = 40.0           # block diff above this = a hard cut inside the clip
MIN_GAP = 0.5             # s kept between two windows taken from the same clip
STEP = 1.0 / SFPS


@dataclass
class ClipScore:
    times: np.ndarray     # frame times (s)
    score: np.ndarray     # per-frame score, -1 = unusable


def read_small(path) -> np.ndarray:
    vf = f"fps={SFPS},scale={SIZE}:{SIZE},format=gray"
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
           "-an", "-vf", vf, "-f", "rawvideo", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ClipsError(f"cannot read video {path}: {msg[-1] if msg else 'ffmpeg failed'}")
    buf = np.frombuffer(proc.stdout, dtype=np.uint8)
    n = buf.size // (SIZE * SIZE)
    return buf[: n * SIZE * SIZE].reshape(n, SIZE, SIZE).astype(np.float32)


def frame_features(frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(motion, sharpness, luma) per frame; motion of frame 0 copies frame 1."""
    n = len(frames)
    if n == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    k = 8
    b = frames.reshape(n, SIZE // k, k, SIZE // k, k).mean(axis=(2, 4))
    d = np.abs(np.diff(b, axis=0)).mean(axis=(1, 2)) if n > 1 else np.zeros(0)
    motion = np.concatenate([[d[0] if d.size else 0.0], d])
    sharp = (np.abs(np.diff(frames, axis=1)).mean(axis=(1, 2))
             + np.abs(np.diff(frames, axis=2)).mean(axis=(1, 2)))
    return motion, sharp, frames.mean(axis=(1, 2))


def score_clips(feats: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> List[ClipScore]:
    good_m = np.concatenate([m[m < CUT_DIFF] for m, _, _ in feats] or [np.zeros(0)])
    all_s = np.concatenate([s for _, s, _ in feats] or [np.zeros(0)])
    pm = max(float(np.percentile(good_m, 95)), 1e-3) if good_m.size else 1.0
    ps = max(float(np.percentile(all_s, 95)), 1e-3) if all_s.size else 1.0
    out = []
    for m, s, lum in feats:
        sc = 0.6 * np.clip(m / pm, 0, 1) + 0.4 * np.clip(s / ps, 0, 1)
        sc[(lum < DARK) | (s < FLAT) | (m >= CUT_DIFF)] = -1.0
        if sc.size and m.size and m[0] >= CUT_DIFF:   # frame 0 copied frame 1's diff
            sc[0] = 0.4 * np.clip(s[0] / ps, 0, 1)
        out.append(ClipScore(np.arange(sc.size) * STEP, sc))
    return out


def analyse(sources: Sequence[Source]) -> List[ClipScore]:
    return score_clips([frame_features(read_small(s.path)) for s in sources])


def window_score(cs: ClipScore, start: float, length: float) -> float:
    m = (cs.times >= start - 1e-9) & (cs.times < start + length - 1e-9)
    if not m.any():
        i = int(np.clip(round(start / STEP), 0, max(cs.score.size - 1, 0)))
        return float(cs.score[i]) if cs.score.size else 0.0
    v = cs.score[m]
    return -1.0 if (v < 0).mean() > 0.25 else float(v[v >= 0].mean()) if (v >= 0).any() else -1.0


def _free(used: List[Tuple[float, float]], a: float, b: float) -> bool:
    return all(b + MIN_GAP <= u0 + 1e-9 or a >= u1 + MIN_GAP - 1e-9 for u0, u1 in used)


def assign_smart(shot_lens: Sequence[float], hits: Sequence[float], sources: Sequence[Source],
                 scores: Sequence[ClipScore],
                 reserved: Optional[Dict[int, List[Tuple[float, float]]]] = None
                 ) -> List[Assignment]:
    """Best non-overlapping source window per shot. Shot 0 (the hook) picks first, then shots
    by hit strength (ties: earlier shot). `reserved` = windows already used (e.g. the intro)."""
    n = len(shot_lens)
    used: Dict[int, List[Tuple[float, float]]] = {i: list((reserved or {}).get(i, []))
                                                  for i in range(len(sources))}
    order = [0] + sorted(range(1, n), key=lambda i: (-hits[i], i)) if n else []
    res: List[Optional[Assignment]] = [None] * n
    longest = max(range(len(sources)), key=lambda i: sources[i].duration)
    for si in order:
        ln = shot_lens[si]
        neighbours = {res[j].source for j in (si - 1, si + 1)
                      if 0 <= j < n and res[j] is not None and not res[j].loop}
        best = None
        for ci, (src, cs) in enumerate(zip(sources, scores)):
            # same clip as a neighbouring shot only when nothing else is free
            penalty = 2.0 if len(sources) > 1 and ci in neighbours else 0.0
            last = _usable(src.duration) - ln
            for t in np.arange(0.0, last + 1e-9, STEP):
                if not _free(used[ci], t, t + ln):
                    continue
                key = (window_score(cs, t, ln) - penalty, -ci, -t)
                if best is None or key > best[0]:
                    best = (key, ci, float(t))
        if best is None:     # no free window long enough anywhere: loop the longest clip
            res[si] = Assignment(longest, 0.0, True)
            continue
        _, ci, t = best
        used[ci].append((t, t + ln))
        res[si] = Assignment(ci, round(t, 6))
    return res  # type: ignore[return-value]
