"""Dialogue intro: --intro SPEC parsing/validation and automatic speech-based selection."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import speech
from .ffmpeg import ClipsError, probe
from .sources import Source

DEFAULT_INTRO_MAX = 8.0


def parse_time(s: str) -> float:
    """Seconds or [hh:]mm:ss[.ms]."""
    s = s.strip()
    try:
        if ":" in s:
            parts = s.split(":")
            if len(parts) > 3:
                raise ValueError
            total = 0.0
            for p in parts:
                total = total * 60 + float(p)
            return total
        return float(s)
    except ValueError:
        raise ClipsError(f"bad time {s!r} (use seconds or mm:ss[.ms])")


def parse_intro_spec(spec: str) -> Tuple[str, float, float]:
    """'PATH@START-END' (split on the LAST '@') -> (path, start, end)."""
    if "@" not in spec:
        raise ClipsError(f"bad --intro {spec!r}: expected PATH@START-END")
    path, rng = spec.rsplit("@", 1)
    if not path or "-" not in rng:
        raise ClipsError(f"bad --intro {spec!r}: expected PATH@START-END")
    a, b = rng.split("-", 1)
    start, end = parse_time(a), parse_time(b)
    if start < 0 or end <= start:
        raise ClipsError(f"bad --intro range in {spec!r}: end must be greater than start")
    return path, start, end


@dataclass
class IntroRegion:
    source: Path
    start: float
    end: float
    score: Optional[float] = None   # None for user-specified regions

    @property
    def length(self) -> float:
        return self.end - self.start

    def to_dict(self):
        return {"path": str(self.source), "start": round(self.start, 3), "end": round(self.end, 3),
                "score": None if self.score is None else round(self.score, 3)}


def resolve_specs(specs: Sequence[str], input_dir: Optional[Path]) -> List[IntroRegion]:
    out = []
    for spec in specs:
        path, start, end = parse_intro_spec(spec)
        p = Path(path)
        if not p.is_file() and input_dir is not None and (input_dir / path).is_file():
            p = input_dir / path
        if not p.is_file():
            raise ClipsError(f"--intro clip not found: {path}")
        info = probe(p)
        if not info.has_video:
            raise ClipsError(f"--intro clip has no video stream: {p}")
        if not info.has_audio:
            raise ClipsError(f"--intro clip has no audio stream: {p}")
        if info.duration is not None and end > info.duration + 1e-3:
            raise ClipsError(f"--intro range {start}-{end}s is beyond the clip duration "
                             f"({info.duration:.2f}s): {p}")
        out.append(IntroRegion(p.resolve(), start, end))
    return out


def auto_intro(sources: Sequence[Source], intro_max: float, decode_fn,
               preferred: Optional[Path] = None, top: int = 5):
    """Detect dialogue in every source with audio. Returns (chosen regions, top candidates)."""
    cands: List[speech.Region] = []
    for s in sources:
        if not probe(s.path).has_audio:
            continue
        cands.extend(speech.detect_source(s.path, decode_fn, s.duration))
    best = sorted(cands, key=lambda r: (-r.score, str(r.source), r.start))[:top]
    chosen = speech.choose_intro(cands, intro_max, preferred)
    regions = [IntroRegion(r.source, r.start, r.end, r.score) for r in chosen]
    return regions, [IntroRegion(r.source, r.start, r.end, r.score) for r in best]


def intro_frames(regions: Sequence[IntroRegion], fps: int) -> List[int]:
    return [max(1, int(round(r.length * fps))) for r in regions]
