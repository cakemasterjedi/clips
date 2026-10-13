"""Cut planning. Pure planning functions plus a thin librosa analysis layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from .ffmpeg import ClipsError, probe

SR = 22050
HOP = 512
MIN_SHOT = 0.25
MAX_SHOT = 4.0
AUTO_MAX_SHOT = 3.5      # target shot length at energy 0 (a scene may breathe)
HOLD_FRAC = 0.12         # top share of cut candidates (by hit strength) that trigger a hold
HOLD_MAX = 2.5           # post-hit hold cap (s)
HOLD_MIN_GAP = 8.0       # at most one post-hit hold per this many seconds
HIGH_ENERGY = 0.7
TEMPO_GUARD_BPM = 150.0
BEAT_TOL = 0.03          # seconds of jitter tolerated when comparing to a target
FALLBACK_GRID = 1.0
FALLBACK_ONSET_GAP = 0.6


@dataclass
class Analysis:
    length: float
    tempo: float
    beats: np.ndarray
    strong_onsets: np.ndarray
    rms: np.ndarray
    rms_times: np.ndarray
    onset_env: Optional[np.ndarray] = None    # onset strength envelope
    onset_env_times: Optional[np.ndarray] = None


@dataclass
class Shot:
    index: int
    start_frame: int
    frames: int
    fps: int
    energy: float = 0.0
    hit: float = 0.0
    punch: bool = False
    flash: bool = False
    section: str = "music"
    ramp: bool = False        # slow-motion tail leading into the next (hit) shot

    @property
    def start(self) -> float:
        return self.start_frame / self.fps

    @property
    def end(self) -> float:
        return (self.start_frame + self.frames) / self.fps

    @property
    def length(self) -> float:
        return self.frames / self.fps


@dataclass
class Plan:
    output_len: float
    fps: int
    mode: str
    tempo: float
    shots: List[Shot] = field(default_factory=list)
    t_intro: float = 0.0


# ---------------------------------------------------------------- analysis
def decode(path, offset: float = 0.0, duration: Optional[float] = None, sr: int = SR) -> np.ndarray:
    """Decode any ffmpeg-readable audio (mp3, m4a, aac, wav, video files...) to mono float32 at SR."""
    import subprocess

    from .ffmpeg import ffmpeg_exe

    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           "-ss", f"{offset:.6f}", "-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise ClipsError(f"cannot decode audio {path}: {msg[-1] if msg else 'ffmpeg failed'}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def audio_length(path) -> float:
    info = probe(path)
    if not info.has_audio:
        raise ClipsError(f"no audio stream in {path}")
    if info.duration:
        return float(info.duration)
    return decode(path).size / SR


def output_length(audio_len: float, audio_start: float, duration: Optional[float]) -> float:
    avail = audio_len - audio_start
    if avail <= 0:
        raise ClipsError(f"--audio-start {audio_start}s is beyond the end of the audio ({audio_len:.2f}s)")
    return min(avail, duration) if duration is not None else avail


def analyze(path, audio_start: float, length: float) -> Analysis:
    import librosa

    y, sr = decode(path, audio_start, length), SR
    if y.size == 0:
        raise ClipsError("audio contains no samples in the requested range")
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    onset_frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=HOP)
    if env.size and onset_frames.size:
        thr = np.percentile(env, 85)
        onset_frames = onset_frames[env[onset_frames] >= thr]
    strong = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_times = librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=HOP)
    return Analysis(length=length, tempo=tempo, beats=np.asarray(beats, float),
                    strong_onsets=np.asarray(strong, float), rms=rms, rms_times=rms_times,
                    onset_env=np.asarray(env, float),
                    onset_env_times=librosa.frames_to_time(np.arange(env.size), sr=sr, hop_length=HOP))


HIT_WINDOW = 0.04
NO_MUSIC_BPM = 120.0


def grid_analysis(length: float, bpm: float = NO_MUSIC_BPM) -> Analysis:
    """Stand-in analysis when there is no song (--no-music): a steady beat grid at `bpm`
    with flat energy, so auto pacing gives even shots (about 1.6 s at 120 BPM)."""
    if bpm <= 0:
        raise ClipsError("--bpm must be > 0")
    beats = np.arange(0.0, length, 60.0 / bpm)
    t = np.arange(0.0, length + HOP / SR, HOP / SR)
    return Analysis(length=length, tempo=float(bpm), beats=beats, strong_onsets=np.zeros(0),
                    rms=np.full(t.size, 0.5), rms_times=t)

# ---------------------------------------------------------------- auto song section
AUTO_WINDOW = 20.0       # scored window when --duration is not given (s)
DROP_PRE = 4.0           # look-back before a candidate start for the drop contrast (s)
DROP_POST = 2.0          # look-ahead after a candidate start for the drop contrast (s)
DROP_WEIGHT = 1.0
ONSET_WEIGHT = 0.5
REPEAT_WEIGHT = 0.5
REPEAT_MIN_LAG = 8        # beats; a repeat must be at least this far away (a chorus, not a loop)


def repetition_scores(chroma: np.ndarray, beat_times: np.ndarray, candidates: Sequence[float],
                      window: float) -> np.ndarray:
    """How strongly the `window` starting at each candidate repeats elsewhere in the song
    (chorus finder, pychorus/msaf idea): cosine self-similarity of beat-synchronous chroma,
    best mean along a diagonal at lag >= max(REPEAT_MIN_LAG, window_beats), scaled to 0..1
    over all beat positions (median -> 0, best -> 1)."""
    n = chroma.shape[1] if chroma.ndim == 2 else 0
    out = np.zeros(len(candidates))
    if n < 2 * REPEAT_MIN_LAG or beat_times.size < 2:
        return out
    period = float(np.median(np.diff(beat_times)))
    k = max(2, int(round(window / period)))
    c = chroma - chroma.mean(axis=1, keepdims=True)   # drop what every beat shares (drums, hum)
    c = c / (np.linalg.norm(c, axis=0, keepdims=True) + 1e-9)
    sim = c.T @ c
    min_lag = max(REPEAT_MIN_LAG, k)              # the repeat must not overlap the window
    best = np.zeros(n)
    for lag in range(min_lag, n - 1):
        d = np.diagonal(sim, offset=lag)            # d[m] = sim[m, m + lag]
        if d.size < k:                                # the repeat must fit in the song
            break
        w = k
        cs = np.concatenate([[0.0], np.cumsum(d)])
        means = (cs[w:] - cs[:-w]) / w                # windows of d starting at m
        m = np.arange(means.size)
        np.maximum.at(best, m, means)                 # window at m repeats later at m + lag
        np.maximum.at(best, m + lag, means)           # ...and the later one repeats m
    lo, hi = float(np.median(best)), float(best.max())
    scaled = np.clip((best - lo) / (hi - lo), 0, 1) if hi - lo > 1e-9 else np.zeros(n)
    for i, t in enumerate(candidates):
        j = int(np.argmin(np.abs(beat_times[:n] - t)))
        if abs(beat_times[j] - t) <= period:         # no beats there (e.g. a beatless intro)
            out[i] = scaled[j]
    return out


def _mean_between(x: np.ndarray, t: np.ndarray, a: float, b: float) -> Optional[float]:
    m = (t >= a) & (t < b)
    return float(x[m].mean()) if m.any() else None


def section_scores(candidates: Sequence[float], window: float, rms: np.ndarray,
                   rms_times: np.ndarray, env: np.ndarray, env_times: np.ndarray,
                   repeat: Optional[np.ndarray] = None) -> np.ndarray:
    """Score each candidate start t for a `window`-second excerpt:
    mean RMS in [t, t+window] + ONSET_WEIGHT * mean onset strength there
    + DROP_WEIGHT * max(0, RMS just after t - RMS just before t)
    + REPEAT_WEIGHT * repeat[i] (how much the window recurs: the chorus).
    RMS and onset strength are normalized by their maxima over the whole song. The drop term
    rewards starting right where the song gets louder; at the song start (nothing before t)
    it is 0."""
    r = rms / rms.max() if rms.size and rms.max() > 0 else np.zeros_like(rms)
    e = env / env.max() if env.size and env.max() > 0 else np.zeros_like(env)
    out = np.zeros(len(candidates))
    for i, t in enumerate(candidates):
        body = _mean_between(r, rms_times, t, t + window) or 0.0
        hits = _mean_between(e, env_times, t, t + window) or 0.0
        pre = _mean_between(r, rms_times, t - DROP_PRE, t)
        post = _mean_between(r, rms_times, t, t + min(DROP_POST, window))
        drop = max(0.0, post - pre) if pre is not None and post is not None else 0.0
        out[i] = body + ONSET_WEIGHT * hits + DROP_WEIGHT * drop
    if repeat is not None:
        out += REPEAT_WEIGHT * np.asarray(repeat, float)
    return out


def pick_section(window: float, song_len: float, beats: np.ndarray, rms: np.ndarray,
                 rms_times: np.ndarray, env: np.ndarray, env_times: np.ndarray,
                 rank: int = 0, chroma: Optional[np.ndarray] = None
                 ) -> Tuple[float, List[Tuple[float, float]]]:
    """Best start (on a detected beat, or 0) for a `window`-second excerpt that fits the song.
    Returns (start, top candidates as (start, score), best first). Ties go to the earlier start.
    The candidates are distinct sections, at least min(window/2, 8 s) apart; `rank` picks the
    rank-th one instead of the best (the last one if there are fewer)."""
    latest = song_len - window
    if latest <= 0:
        return 0.0, [(0.0, 0.0)]
    cands = np.unique(np.concatenate([[0.0], np.asarray(beats, float)]))
    cands = cands[(cands >= 0) & (cands <= latest + 1e-6)]
    rep = (repetition_scores(chroma, np.asarray(beats, float), cands, window)
           if chroma is not None else None)
    scores = section_scores(cands, window, rms, rms_times, env, env_times, rep)
    order = sorted(range(cands.size), key=lambda i: (-round(scores[i], 6), cands[i]))
    sep = min(window / 2, 8.0)
    kept: List[int] = []
    for i in order:
        if all(abs(cands[i] - cands[j]) >= sep - 1e-6 for j in kept):
            kept.append(i)
            if len(kept) == 5:
                break
    top = [(round(float(cands[i]), 3), round(float(scores[i]), 4)) for i in kept]
    return float(cands[kept[min(rank, len(kept) - 1)]]), top


def auto_audio_start(path, window: float, lead: float = 0.0,
                     rank: int = 0) -> Tuple[float, List[Tuple[float, float]]]:
    """Analyse the whole song and pick where the `window`-second excerpt should start.
    lead > 0 (from a style profile): the best drop is found for the part of the window after
    the lead, and the start moves back to the beat closest to `lead` seconds before it."""
    import librosa

    y = decode(path)
    if y.size == 0:
        raise ClipsError(f"audio contains no samples: {path}")
    song_len = y.size / SR
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP)
    beats = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    rms_times = librosa.frames_to_time(np.arange(rms.size), sr=SR, hop_length=HOP)
    env_times = librosa.frames_to_time(np.arange(env.size), sr=SR, hop_length=HOP)
    env = np.asarray(env, float)
    chroma = librosa.util.sync(librosa.feature.chroma_stft(y=y, sr=SR, hop_length=HOP),
                               beat_frames, aggregate=np.median)[:, 1:]   # column j = beat j
    if lead <= 0:
        return pick_section(window, song_len, beats, rms, rms_times, env, env_times, rank, chroma)
    drop, top = pick_section(max(window - lead, 1.0), song_len, beats, rms, rms_times, env,
                             env_times, rank, chroma)
    cands = np.concatenate([[0.0], np.asarray(beats, float)])
    cands = cands[cands <= drop + 1e-6]
    start = float(cands[np.argmin(np.abs(cands - (drop - lead)))])
    return start, top   # top = drop candidates; the start is `lead` before the best one


# ---------------------------------------------------------------- loop ending
LOOP_SEARCH_BEATS = 8     # how far before the requested end the loop point may move


def _beat_features(y: np.ndarray, sr: int, a: float, b: float) -> Optional[np.ndarray]:
    import librosa

    seg = y[int(a * sr):int(b * sr)]
    if seg.size < sr // 8:
        return None
    chroma = librosa.feature.chroma_stft(y=seg, sr=sr, hop_length=HOP).mean(axis=1)
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, hop_length=HOP).mean(axis=1)[1:]
    v = np.concatenate([chroma / (np.linalg.norm(chroma) + 1e-9), mfcc / (np.linalg.norm(mfcc) + 1e-9)])
    return v / (np.linalg.norm(v) + 1e-9)


def pick_loop_end(y: np.ndarray, sr: int, beats: np.ndarray, length: float,
                  search: int = LOOP_SEARCH_BEATS) -> Tuple[float, float]:
    """Music length that ends on a beat whose continuation sounds most like the opening beat,
    Infinite-Jukebox style, so the replay's jump back to the start is hard to hear.
    Candidates are the last `search` beats at or before `length` (never longer). Returns
    (length, similarity); falls back to (length, 0) without usable beats."""
    beats = np.asarray(beats, float)
    beats = beats[(beats > 1.0) & (beats <= length + 1e-6)]
    if beats.size == 0:
        return length, 0.0
    period = float(np.median(np.diff(beats))) if beats.size > 1 else 0.5
    ref = _beat_features(y, sr, 0.0, period)
    if ref is None:
        return length, 0.0
    best = (-2.0, float(beats[-1]))
    cands = beats[-search:]
    for k, e in enumerate(cands[::-1]):
        f = _beat_features(y, sr, e, e + period)          # what would play after the end
        if f is None:
            f = _beat_features(y, sr, e - period, e)
        if f is None:
            continue
        score = float(f @ ref) - 0.01 * k                   # mild preference for the full length
        if score > best[0]:
            best = (score, float(e))
    return best[1], round(best[0], 3)


def loop_length(path, audio_start: float, length: float) -> Tuple[float, float]:
    """Music length for --loop (see pick_loop_end), analysing the song from audio_start."""
    import librosa

    y = decode(path, audio_start, length + 2.0)
    if y.size == 0:
        return length, 0.0
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=HOP)
    _, bf = librosa.beat.beat_track(onset_envelope=env, sr=SR, hop_length=HOP)
    beats = librosa.frames_to_time(bf, sr=SR, hop_length=HOP)
    return pick_loop_end(y, SR, beats, length)


# ---------------------------------------------------------------- energy
def interval_energy(boundaries: Sequence[float], rms: np.ndarray, rms_times: np.ndarray,
                    length: float, smooth: int = 4) -> np.ndarray:
    """Normalized [0,1] mean-RMS energy per interval [b[i], b[i+1]) (last runs to `length`)."""
    b = np.asarray(boundaries, float)
    if b.size == 0:
        return np.zeros(0)
    edges = np.append(b, max(length, b[-1] + 1e-3))
    raw = np.zeros(b.size)
    for i in range(b.size):
        lo = np.searchsorted(rms_times, edges[i], "left")
        hi = np.searchsorted(rms_times, edges[i + 1], "left")
        if hi > lo:
            raw[i] = rms[lo:hi].mean()
        elif lo < rms.size:
            raw[i] = rms[min(lo, rms.size - 1)]
    # reference percentiles come from the whole track (sliding mean at ~one interval), so a
    # half-quiet/half-loud track still spans the full 0..1 range even if beats are missing
    # in the quiet part
    hop_s = float(rms_times[1] - rms_times[0]) if rms_times.size > 1 else 0.023
    period = float(np.median(np.diff(b))) if b.size > 1 else 0.5
    win = int(np.clip(round(period / hop_s), 1, max(1, rms.size)))
    ref = np.convolve(rms, np.ones(win) / win, mode="valid") if rms.size >= win else rms
    p10, p90 = np.percentile(ref, 10), np.percentile(ref, 90)
    if p90 - p10 < 1e-9:
        norm = np.full(b.size, 0.5)
    else:
        norm = np.clip((raw - p10) / (p90 - p10), 0.0, 1.0)
    if smooth > 1 and b.size > 1:
        pad = np.pad(norm, (smooth // 2, smooth - 1 - smooth // 2), mode="edge")
        norm = np.convolve(pad, np.ones(smooth) / smooth, mode="valid")
    return norm


def make_energy_fn(boundaries: Sequence[float], energies: np.ndarray) -> Callable[[float], float]:
    b = np.asarray(boundaries, float)
    if b.size == 0:
        return lambda t: 0.0

    def fn(t: float) -> float:
        i = int(np.clip(np.searchsorted(b, t, "right") - 1, 0, b.size - 1))
        return float(energies[i])

    return fn


# ---------------------------------------------------------------- cut selection
def make_hit_fn(env: Optional[np.ndarray], env_times: Optional[np.ndarray]) -> Callable[[float], float]:
    """hit(t) = onset-envelope max within +-40 ms of t, divided by the envelope max (0..1)."""
    if env is None or env_times is None or len(env) == 0 or float(np.max(env)) <= 0:
        return lambda t: 0.0
    peak = float(np.max(env))

    def fn(t: float) -> float:
        m = np.abs(env_times - t) <= HIT_WINDOW
        v = float(env[m].max()) if m.any() else float(env[int(np.argmin(np.abs(env_times - t)))])
        return float(np.clip(v / peak, 0.0, 1.0))

    return fn


def auto_cuts(beats: np.ndarray, tempo: float, strong_onsets: np.ndarray,
              energy_fn: Callable[[float], float], length: float,
              max_shot: float = AUTO_MAX_SHOT, min_shot: float = MIN_SHOT,
              hit_fn: Optional[Callable[[float], float]] = None,
              onset_cuts: bool = True) -> List[float]:
    """Energy-adaptive cut times (seconds, first is 0.0).

    Target shot length falls from `max_shot` at energy 0 to one beat at energy 1
    (period + (max_shot - period) * (1 - e)^1.5). After a cut whose hit strength is among the
    top HOLD_FRAC of all cut candidates, the next shot is held for at least
    min(4 beats, HOLD_MAX); at most one such hold per HOLD_MIN_GAP seconds.
    onset_cuts=False keeps cuts on beats only (no extra cuts on strong off-beat onsets).
    """
    beats = np.asarray(beats, float)
    period = float(np.median(np.diff(beats))) if beats.size > 1 else 60.0 / max(tempo, 1.0)
    fast = max(tempo, 60.0 / period) > TEMPO_GUARD_BPM
    onset_gap = max(min_shot, 0.5 * period)
    cands = [(float(t), True) for t in beats]
    for t in (np.asarray(strong_onsets, float) if onset_cuts else ()):
        if beats.size == 0 or np.min(np.abs(beats - t)) > 0.06:
            cands.append((float(t), False))
    cands.sort()
    hot = set()
    if hit_fn is not None and cands:
        hits = [(hit_fn(t), t) for t, _ in cands]
        k = max(1, int(round(HOLD_FRAC * len(cands))))
        hot = {t for h, t in sorted(hits, key=lambda x: (-x[0], x[1]))[:k] if h > 0}
    hold_len = min(4 * period, HOLD_MAX)
    cuts = [0.0]
    last = 0.0
    holding = False
    last_hold = -1e9
    for t, is_beat in cands:
        if t <= 1e-6 or t >= length - 1e-6:
            continue
        d = t - last
        if is_beat:
            e = energy_fn(last)
            target = period + (max_shot - period) * (1.0 - e) ** 1.5
            target = float(np.clip(target, min_shot, MAX_SHOT))
            if fast:
                target = max(target, 2 * period)
            if holding:
                target = max(target, hold_len)
            ok = d >= target - BEAT_TOL
        else:
            ok = energy_fn(t) > HIGH_ENERGY and d >= onset_gap and (not holding or d >= hold_len)
        if ok:
            cuts.append(t)
            last = t
            holding = False
            if t in hot and t - last_hold >= HOLD_MIN_GAP:
                holding = True
                last_hold = t
    return cuts


def every_n_beats(beats: np.ndarray, n: int, length: float) -> List[float]:
    beats = np.asarray(beats, float)
    if beats.size < 2:
        step = n * 0.5   # no usable beats: assume 120 BPM grid
        return list(np.arange(0.0, length, step))
    cuts = [0.0] + [float(t) for t in beats[::n] if t > 1e-6 and t < length]
    return cuts


def fallback_cuts(strong_onsets: np.ndarray, length: float) -> Tuple[List[float], str]:
    on = np.asarray(strong_onsets, float)
    if on.size >= 4:
        cuts, last = [0.0], 0.0
        for t in on:
            if 1e-6 < t < length and t - last >= FALLBACK_ONSET_GAP:
                cuts.append(float(t))
                last = float(t)
        if len(cuts) >= 4:
            return cuts, "onsets"
    return list(np.arange(0.0, length, FALLBACK_GRID)), "grid"


# ---------------------------------------------------------------- finalize
def finalize_frames(cuts: Sequence[float], length: float, fps: int,
                    min_shot: float = MIN_SHOT, max_shot: float = MAX_SHOT) -> List[Tuple[int, int]]:
    """Snap to frame grid, merge short shots, split long ones.

    Returns [(start_frame, frames)] summing exactly to round(length*fps). Every shot is
    within [ceil(min_shot*fps), floor(max_shot*fps)] frames, except when the whole output
    is shorter than min_shot (then there is a single shorter shot).
    """
    total = int(round(length * fps))
    if total <= 0:
        raise ClipsError("output length is zero frames")
    min_f = int(np.ceil(min_shot * fps - 1e-9))
    max_f = int(np.floor(max_shot * fps + 1e-9))
    pts = sorted({0} | {int(round(t * fps)) for t in cuts if 0 < round(t * fps) < total})
    lens = [b - a for a, b in zip(pts, pts[1:] + [total])]
    # merge short shots into previous neighbour (or next for the first one)
    i = 0
    while len(lens) > 1 and i < len(lens):
        if lens[i] < min_f:
            if i > 0:
                lens[i - 1] += lens.pop(i)
                i -= 1
            else:
                first = lens.pop(0)
                lens[0] += first
        else:
            i += 1
    out: List[int] = []
    for n in lens:
        k = max(1, -(-n // max_f))
        base, rem = divmod(n, k)
        out.extend(base + (1 if j < rem else 0) for j in range(k))
    res, pos = [], 0
    for n in out:
        res.append((pos, n))
        pos += n
    assert pos == total
    return res


# ---------------------------------------------------------------- beat-hit effects
PUNCH_FRAC = 0.35
FLASH_FRAC = 0.15
RAMP_FRAC = 0.15          # share of shots whose start is a big enough hit to ramp into
RAMP_MIN_SHOT = 0.5       # s; lead-in shots shorter than this are not slowed


def select_ramps(hits: Sequence[float], energies: Sequence[float], lengths: Sequence[float],
                 frac: float = RAMP_FRAC) -> List[bool]:
    """ramp[i] = shot i slows down into shot i+1. The hits ramped into are the top `frac` of
    shots by hit strength plus the energy rise across the cut (ties: earlier); the lead-in shot
    must be at least RAMP_MIN_SHOT long. Shot 0 is never ramped into."""
    n = len(hits)
    key = {h: hits[h] + max(0.0, energies[h] - energies[h - 1]) for h in range(1, n) if hits[h] > 0}
    order = sorted(key, key=lambda h: (-key[h], h))
    out = [False] * n
    for h in order[:int(round(frac * n))]:
        if lengths[h - 1] >= RAMP_MIN_SHOT - 1e-9:
            out[h - 1] = True
    return out


def hit_strengths(shot_starts: Sequence[float], env: Optional[np.ndarray],
                  env_times: Optional[np.ndarray]) -> List[float]:
    """Onset-envelope max within +-40 ms of each shot start, / envelope max, in [0,1].
    Normalizing by the max (not a percentile) avoids clipping, so strong hits don't tie.
    The first shot always gets 0."""
    n = len(shot_starts)
    if env is None or env_times is None or len(env) == 0:
        return [0.0] * n
    peak = float(np.max(env))
    if peak <= 0:
        return [0.0] * n
    out = []
    for i, t in enumerate(shot_starts):
        if i == 0:
            out.append(0.0)
            continue
        m = np.abs(env_times - t) <= HIT_WINDOW
        v = float(env[m].max()) if m.any() else float(env[int(np.argmin(np.abs(env_times - t)))])
        out.append(float(np.clip(v / peak, 0.0, 1.0)))
    return out


def select_fx(hits: Sequence[float], punch_frac: float = PUNCH_FRAC,
              flash_frac: float = FLASH_FRAC) -> Tuple[List[bool], List[bool]]:
    """Pick the top `punch_frac` / `flash_frac` of shots by hit strength.

    Counts are round(frac * n_shots). Shot 0 and shots with hit <= 0 are never chosen; ties
    are broken by earliest shot. Flash shots are always a subset of punch shots.
    """
    n = len(hits)
    order = sorted((i for i in range(1, n) if hits[i] > 0), key=lambda i: (-hits[i], i))
    punch = set(order[:int(round(punch_frac * n))])
    flash = set(order[:int(round(flash_frac * n))])
    return [i in punch for i in range(n)], [i in flash for i in range(n)]


# ---------------------------------------------------------------- top level
def plan_from_analysis(an: Analysis, fps: int, beats_per_cut: Optional[int] = None,
                       punch: bool = False, flash: bool = False,
                       max_shot: float = AUTO_MAX_SHOT, onset_cuts: bool = True,
                       punch_frac: float = PUNCH_FRAC, flash_frac: float = FLASH_FRAC,
                       ramp: bool = False) -> Plan:
    length = an.length
    beats = an.beats[(an.beats >= 0) & (an.beats < length)]
    if beats_per_cut:
        cuts = every_n_beats(beats, beats_per_cut, length)
        mode = f"beats_per_cut={beats_per_cut}"
    elif beats.size >= 4:
        # energy intervals also cover the stretch before the first detected beat
        bounds = beats if beats[0] <= 1e-3 else np.concatenate([[0.0], beats])
        en = interval_energy(bounds, an.rms, an.rms_times, length)
        efn = make_energy_fn(bounds, en)
        cuts = auto_cuts(beats, an.tempo, an.strong_onsets, efn, length,
                         max_shot=max_shot, hit_fn=make_hit_fn(an.onset_env, an.onset_env_times),
                         onset_cuts=onset_cuts)
        mode = "auto"
    else:
        cuts, mode = fallback_cuts(an.strong_onsets, length)
        mode = "fallback-" + mode
    if mode == "auto":
        efn_final = efn
    else:
        b = np.arange(0.0, length, 0.5)
        efn_final = make_energy_fn(b, interval_energy(b, an.rms, an.rms_times, length))
    frames = finalize_frames(cuts, length, fps)
    shots = [Shot(i, s, n, fps, round(efn_final(s / fps), 3)) for i, (s, n) in enumerate(frames)]
    hits = hit_strengths([sh.start for sh in shots], an.onset_env, an.onset_env_times)
    pu, fl = select_fx(hits, punch_frac, flash_frac)
    for sh, h, p, f in zip(shots, hits, pu, fl):
        sh.hit = round(h, 3)
        sh.punch = bool(punch and p)
        sh.flash = bool(flash and f)
    if ramp:
        for sh, r in zip(shots, select_ramps([s.hit for s in shots], [s.energy for s in shots],
                                             [s.length for s in shots])):
            sh.ramp = r
    return Plan(length, fps, mode, an.tempo, shots)


def compose_plan(intro_frames: Sequence[int], music: Plan) -> Plan:
    """Intro shots first (one shot per dialogue region, never split, no FX), then the music
    section shifted by the intro length. Total frames = sum(intro) + music frames."""
    fps = music.fps
    shots: List[Shot] = []
    pos = 0
    for n in intro_frames:
        shots.append(Shot(len(shots), pos, int(n), fps, 0.0, 0.0, False, False, "intro"))
        pos += int(n)
    for sh in music.shots:
        shots.append(Shot(len(shots), pos + sh.start_frame, sh.frames, fps, sh.energy, sh.hit,
                          sh.punch, sh.flash, "music", sh.ramp))
    t_intro = pos / fps
    return Plan(music.output_len + t_intro, fps, music.mode, music.tempo, shots, t_intro)
