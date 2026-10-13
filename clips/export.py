"""Export the cut plan as a CMX3600 EDL so the edit can be finished in Resolve / Premiere.

One video event per shot (source clip in/out -> record in/out), the song as an audio event
for the music section and, with a dialogue intro, the intro clips' own audio. Effects
(punch, flash, ramps, loop dissolve), fit and captions are not part of an EDL; the clip
names go in `* FROM CLIP NAME:` comments, which both NLEs use to relink media.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio import Plan
from .sources import Assignment, Source


def timecode(seconds: float, fps: int) -> str:
    f = int(round(seconds * fps))
    return f"{f // (3600 * fps):02d}:{f // (60 * fps) % 60:02d}:{f // fps % 60:02d}:{f % fps:02d}"


def _event(n: int, track: str, src_in: float, src_out: float, rec_in: float, rec_out: float,
           fps: int, name: str) -> List[str]:
    return [f"{n:03d}  AX       {track:<5} C        {timecode(src_in, fps)} {timecode(src_out, fps)} "
            f"{timecode(rec_in, fps)} {timecode(rec_out, fps)}",
            f"* FROM CLIP NAME: {name}"]


def build_edl(plan: Plan, sources: Sequence[Source], assigns: Sequence[Assignment],
              audio: Path, audio_start: float,
              intro_audio: Optional[Sequence[Tuple[Path, float, float]]] = None,
              title: str = "clips") -> str:
    fps = plan.fps
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    n = 0
    for sh, a in zip(plan.shots, assigns):
        n += 1
        lines += _event(n, "V", a.start, a.start + sh.length, sh.start, sh.end, fps,
                        Path(sources[a.source].path).name)
        if a.loop:
            lines.append("* NOTE: source shorter than the shot, looped in the render")
        lines.append("")
    rec = 0.0
    for src, st, ln in intro_audio or []:
        n += 1
        lines += _event(n, "A", st, st + ln, rec, rec + ln, fps, Path(src).name) + [""]
        rec += ln
    music = plan.output_len - plan.t_intro
    n += 1
    lines += _event(n, "A", audio_start, audio_start + music, plan.t_intro, plan.output_len, fps,
                    Path(audio).name) + [""]
    return "\n".join(lines)
