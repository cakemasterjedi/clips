"""Heuristic local speech detection (DSP only, no models).

Per-frame speech score combines voice-band energy ratio, wideband flatness, loudness relative
to the clip, syllabic (3-8 Hz) envelope modulation and a beat-regularity penalty. Frames over a
threshold are merged into regions whose edges are snapped to low-RMS pauses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

SR = 16000
WIN = 400           # 25 ms
HOP = 160           # 10 ms
FPS = SR / HOP      # 100 frames/s
THRESHOLD = 0.5
BRIDGE = 0.35
MIN_REGION = 1.5
PAD = 0.1
GRID = 0.25         # feature grid (s) for the windowed features


@dataclass
class Region:
    source: Optional[Path]
    start: float
    end: float
    score: float
    pauses: List[float] = field(default_factory=list, repr=False)  # pause centres (abs seconds)

    @property
    def length(self) -> float:
        return self.end - self.start

    def to_dict(self):
        return {"path": str(self.source) if self.source else None, "start": round(self.start, 3),
                "end": round(self.end, 3), "score": round(self.score, 3)}


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _windowed(env: np.ndarray, centers: np.ndarray, half: int, fn) -> np.ndarray:
    out = np.zeros(centers.size)
    for k, c in enumerate(centers):
        seg = env[max(0, c - half): c + half]
        out[k] = fn(seg)
    return out


def _modulation(seg: np.ndarray) -> float:
    """Syllabic modulation score of an envelope window (~1 s)."""
    if seg.size < 50:
        return 0.0
    m = seg.mean()
    if m <= 1e-9:
        return 0.0
    x = seg / m - 1.0
    depth = float(np.std(x))
    nfft = 256
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size), nfft)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / FPS)
    tot = spec[(f >= 0.5) & (f <= 20)].sum()
    if tot <= 0:
        return 0.0
    ratio = spec[(f >= 3) & (f <= 8)].sum() / tot
    return float(_clip01((ratio - 0.15) / 0.35) * _clip01((depth - 0.1) / 0.3))


def _regularity(seg: np.ndarray) -> float:
    """Max normalized autocorrelation of the envelope at beat-like lags (0.3-1.2 s)."""
    if seg.size < 200:
        return 0.0
    x = seg - seg.mean()
    e0 = float(np.dot(x, x))
    if e0 <= 1e-12:
        return 0.0
    n = 1 << int(np.ceil(np.log2(2 * x.size)))
    ac = np.fft.irfft(np.abs(np.fft.rfft(x, n)) ** 2, n)[: x.size] / e0
    lo, hi = int(0.3 * FPS), min(int(1.2 * FPS), x.size - 1)
    return float(ac[lo:hi].max())


def score_frames(y: np.ndarray, sr: int = SR):
    """Return (score[n_frames], rms_db[n_frames]) at 100 frames/s."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, win_length=WIN, center=True)) ** 2
    n = S.shape[1]
    f = np.fft.rfftfreq(512, 1.0 / sr)
    tot = S.sum(0) + 1e-12
    band = S[(f >= 300) & (f <= 3400)].sum(0) / tot
    wide = S[(f >= 80) & (f <= 7500)] + 1e-12
    flat = np.exp(np.log(wide).mean(0)) / wide.mean(0)
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP, center=True)[0][:n]
    db = 20 * np.log10(rms + 1e-9)
    ref = np.percentile(db, 90)
    s_band = _clip01((band - 0.5) / 0.4)
    s_flat = 1.0 - _clip01((flat - 0.3) / 0.4)
    s_rms = _clip01((db - ref + 30) / 20)
    env = rms.astype(float)
    centers = np.arange(0, n, int(GRID * FPS))
    mod = _windowed(env, centers, int(0.5 * FPS), _modulation)
    reg = _windowed(env, centers, int(1.5 * FPS), _regularity)
    idx = np.arange(n)
    s_mod = np.interp(idx, centers, mod)
    regularity = np.interp(idx, centers, reg)
    s_beat = 1.0 - _clip01((regularity - 0.45) / 0.35)
    base = 0.25 * s_band + 0.10 * s_flat + 0.15 * s_rms + 0.50 * s_mod
    score = base * (0.2 + 0.8 * s_beat)
    score[db < -55] = 0.0
    return score, db


def _runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    if mask.size == 0:
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _pause_centres(db: np.ndarray, lo: int, hi: int, med_db: float) -> List[Tuple[int, int]]:
    """(start, end) frame runs of genuine low-RMS pauses inside [lo, hi)."""
    k = 5
    sm = np.convolve(np.pad(db, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")
    seg = sm[lo:hi]
    if seg.size == 0:
        return []
    thr = min(np.percentile(seg, 12) + 3.0, med_db - 6.0)
    return [(lo + a, lo + b) for a, b in _runs(seg < thr) if b - a >= 6]


def _snap(t_frame: int, runs: List[Tuple[int, int]], db: np.ndarray, side: str,
          reach: int, lo: int, hi: int) -> int:
    """Snap a boundary to the nearest pause (start -> just before the word, end -> just after)."""
    best, best_d = None, None
    for a, b in runs:
        c = (a + b) // 2
        d = abs(c - t_frame)
        if d <= reach and (best_d is None or d < best_d):
            best, best_d = (a, b), d
    if best is not None:
        a, b = best
        pad = int(PAD * FPS)
        if side == "start":
            return b - pad if b - a > 2 * pad else (a + b) // 2
        return a + pad if b - a > 2 * pad else (a + b) // 2
    a, b = max(lo, t_frame - int(0.3 * FPS)), min(hi, t_frame + int(0.3 * FPS))
    return a + int(np.argmin(db[a:b])) if b > a else t_frame


def detect_regions(y: np.ndarray, sr: int = SR, threshold: float = THRESHOLD,
                   source: Optional[Path] = None) -> List[Region]:
    if y.size < sr:
        return []
    score, db = score_frames(y, sr)
    n = score.size
    active = score >= threshold
    runs = _runs(active)
    merged: List[List[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] < BRIDGE * FPS:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out: List[Region] = []
    for a, b in merged:
        if (b - a) / FPS < MIN_REGION:
            continue
        med = float(np.median(db[a:b]))
        ctx_lo, ctx_hi = max(0, a - int(1.0 * FPS)), min(n, b + int(1.0 * FPS))
        pr = _pause_centres(db, ctx_lo, ctx_hi, med)
        reach = int(0.7 * FPS)
        s0 = _snap(a, pr, db, "start", reach, ctx_lo, ctx_hi)
        e0 = _snap(b, pr, db, "end", reach, ctx_lo, ctx_hi)
        s0, e0 = max(0, s0), min(n, e0)
        if (e0 - s0) / FPS < MIN_REGION:
            continue
        sc = float(score[s0:e0].mean())
        out.append(Region(source, s0 / FPS, e0 / FPS, sc,
                          [((p0 + p1) // 2) / FPS for p0, p1 in pr if s0 <= (p0 + p1) // 2 <= e0]))
    return out


def detect_source(path, decode_fn, duration: Optional[float] = None) -> List[Region]:
    y = decode_fn(path, 0.0, duration, SR)
    return detect_regions(y, SR, source=Path(path))


def trim_region(r: Region, max_len: float, min_len: float = MIN_REGION) -> Optional[Region]:
    """Trim the end of a region to <= max_len at the last pause that keeps >= min_len."""
    if r.length <= max_len + 1e-9:
        return r
    if max_len < min_len:
        return None
    ok = [p for p in r.pauses if min_len <= p - r.start <= max_len]
    end = max(ok) if ok else r.start + max_len   # no pause available: hard cut at the limit
    return Region(r.source, r.start, end, r.score, [p for p in r.pauses if p <= end])


def choose_intro(cands: Sequence[Region], intro_max: float,
                 preferred: Optional[Path] = None, bonus: float = 1.15) -> List[Region]:
    """Best region (trimmed to fit), plus a second one when it still fits in intro_max."""
    def rank(r: Region) -> float:
        return r.score * (bonus if preferred is not None and r.source == preferred else 1.0)

    ordered = sorted(cands, key=lambda r: (-rank(r), str(r.source), r.start))
    chosen: List[Region] = []
    remaining = intro_max
    for r in ordered:
        if remaining < MIN_REGION:
            break
        t = trim_region(r, remaining)
        if t is None:
            continue
        chosen.append(t)
        remaining -= t.length
        if len(chosen) == 2:
            break
    return _join_close(chosen, intro_max)


MERGE_GAP = 1.0


def _join_close(chosen: List[Region], intro_max: float) -> List[Region]:
    """Two regions from the same clip play in source order; if under MERGE_GAP apart they become
    one continuous region (avoids a jump cut + audio skip mid-conversation)."""
    if len(chosen) != 2 or chosen[0].source != chosen[1].source:
        return chosen
    a, b = sorted(chosen, key=lambda r: r.start)
    if 0 <= b.start - a.end < MERGE_GAP and b.end - a.start <= intro_max + 1e-9:
        return [Region(a.source, a.start, b.end, max(a.score, b.score), a.pauses + b.pauses)]
    return [a, b]
