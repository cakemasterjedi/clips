"""Two-pass render: per-shot normalized segments, then concat + subs + music."""
from __future__ import annotations

import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio import Plan
from .ffmpeg import run_ffmpeg
from .sources import Assignment, Source
from .subs import escape_filter_path, prepare_subs

FADE = 0.3


FITS = ("crop", "letterbox", "blur")
PUNCH_ZOOM = 0.12
PREVIEW_SIZE = (270, 480)   # --preview renders shots at quarter size, upscaled in the final pass
PREVIEW_X264 = ["-preset", "ultrafast", "-crf", "30"]
FINAL_X264 = ["-preset", "medium", "-crf", "20"]
RAMP_DUR = 0.5            # s of output at the end of a lead-in shot played in slow motion
RAMP_SPEED = 0.4
LOOP_BLEND = 0.25         # s; the last shot dissolves into the opening frame (--loop)
LOOP_FADE = 0.02          # s; audio fade at the loop point (just enough to avoid a click)
PUNCH_DUR = 0.18
FLASH_DUR = 0.1


def fit_graph(fit: str, fps: int, inp: str = "0:v", out: str = "f",
              size: Tuple[int, int] = (1080, 1920)) -> str:
    """Filter graph from [inp] to [out] producing exactly `size` (1080x1920) for the fit mode."""
    W, H = size
    if fit == "crop":
        return (f"[{inp}]fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"crop={W}:{H}[{out}]")
    if fit == "letterbox":
        return (f"[{inp}]fps={fps},scale={W}:{H}:force_original_aspect_ratio=decrease:"
                f"force_divisible_by=2,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black[{out}]")
    if fit == "blur":
        p = "" if out == "f" else out
        return (f"[{inp}]fps={fps},split[{p}a][{p}b];"
                f"[{p}a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
                f"gblur=sigma=30,eq=brightness=-0.08[{p}bg];"
                f"[{p}b]scale={W}:{H}:force_original_aspect_ratio=decrease:force_divisible_by=2[{p}fg];"
                f"[{p}bg][{p}fg]overlay=(W-w)/2:(H-h)/2[{out}]")
    raise ValueError(f"unknown fit mode {fit!r}")


def segment_filter(fit: str, fps: int, punch: bool = False, flash: bool = False,
                   punch_zoom: float = PUNCH_ZOOM, blend_at: Optional[float] = None,
                   size: Tuple[int, int] = (1080, 1920),
                   ramp_len: Optional[float] = None, preview: bool = False) -> str:
    """Complete -filter_complex for one shot, output label [v]. blend_at (loop ending): from
    that time on, the first frame of input 1 (the opening shot) fades in over the shot.
    ramp_len (shot length in s): the last min(RAMP_DUR, len/2) s play at RAMP_SPEED, frame
    blended up to fps (plain frame repeats in preview)."""
    W, H = size
    g = fit_graph(fit, fps, size=size)
    last = "f"
    if ramp_len is not None:
        r = min(RAMP_DUR, ramp_len / 2)
        t1 = ramp_len - r
        smooth = "" if preview else f"minterpolate=fps={fps}:mi_mode=blend,"
        g += (f";[f]split[ra][rb];[ra]trim=end={t1:.6f},setpts=PTS-STARTPTS[rn];"
              f"[rb]trim=start={t1:.6f}:end={t1 + r * RAMP_SPEED:.6f},"
              f"setpts=(PTS-STARTPTS)/{RAMP_SPEED},{smooth}fps={fps}[rs];"
              f"[rn][rs]concat=n=2:v=1:a=0,fps={fps}[rr]")
        last = "rr"
    if blend_at is not None:
        g += (";" + fit_graph(fit, fps, "1:v", "l0", size) +
              f";[l0]trim=end_frame=1,loop=loop=-1:size=1,setpts=N/{fps}/TB,format=yuva420p,"
              f"fade=t=in:st={blend_at:.6f}:d={LOOP_BLEND:.6f}:alpha=1[l1];"
              f"[{last}][l1]overlay=shortest=1:format=auto[fb]")
        last = "fb"
    if punch:
        z = f"(1+{punch_zoom:g}*pow(1-min(t/{PUNCH_DUR},1),2))"   # 1.12 -> 1.0, ease-out
        g += (f";[{last}]split[pb][pz];"
              f"[pz]scale=w='2*trunc({W}*{z}/2)':h='2*trunc({H}*{z}/2)':eval=frame[pzz];"
              f"[pb][pzz]overlay=x='2*trunc((W-w)/4)':y='2*trunc((H-h)/4)':eval=frame:"
              f"shortest=1[p]")
        last = "p"
    tail = "setsar=1,format=yuv420p"
    if flash:
        tail = f"fade=t=in:st=0:d={FLASH_DUR}:color=white," + tail
    return g + f";[{last}]{tail}[v]"


def segment_args(src: Path, a: Assignment, frames: int, fps: int, out: Path,
                 fit: str = "crop", punch: bool = False, flash: bool = False,
                 punch_zoom: float = PUNCH_ZOOM,
                 blend_to: Optional[Tuple[Path, float]] = None, preview: bool = False,
                 ramp: bool = False) -> List[str]:
    args = ["-y"]
    if a.loop:
        args += ["-stream_loop", "-1"]
    args += ["-ss", f"{a.start:.6f}", "-i", str(src)]
    blend_at = None
    if blend_to is not None:
        args += ["-ss", f"{blend_to[1]:.6f}", "-i", str(blend_to[0])]
        blend_at = max(frames / fps - LOOP_BLEND, 0.0)
    args += ["-frames:v", str(frames), "-an",
             "-filter_complex", segment_filter(fit, fps, punch, flash, punch_zoom, blend_at,
                                               PREVIEW_SIZE if preview else (1080, 1920),
                                               frames / fps if ramp else None, preview),
             "-map", "[v]",
             "-c:v", "libx264", *(PREVIEW_X264 if preview else ["-preset", "veryfast", "-crf", "18"]),
             "-r", str(fps), str(out)]
    return args


def concat_line(p: Path) -> str:
    return "file '" + str(p).replace("'", "'\\''") + "'\n"


def _video_chain(ass: Optional[Path], preview: bool = False) -> str:
    vchain = "scale=1080:1920:flags=bilinear," if preview else ""   # preview segments are small
    vchain += f"ass=filename={escape_filter_path(ass)}," if ass else ""
    return f"[0:v]{vchain}setsar=1,format=yuv420p[v]"


def final_args(listfile: Path, audio: Path, audio_start: float, length: float, fps: int,
               ass: Optional[Path], out: Path, fade: float = FADE, preview: bool = False) -> List[str]:
    fin = f"afade=t=in:st=0:d={fade}," if fade < FADE else ""   # loop: no click at the restart
    fc = (f"{_video_chain(ass, preview)};"
          f"[1:a]atrim=duration={length:.6f},asetpts=PTS-STARTPTS,{fin}"
          f"afade=t=out:st={max(length - fade, 0):.6f}:d={fade}[a]")
    return ["-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-ss", f"{audio_start:.6f}", "-i", str(audio),
            "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", *(PREVIEW_X264 if preview else FINAL_X264), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k", "-r", str(fps), "-movflags", "+faststart",
            "-t", f"{length:.6f}", str(out)]


# ---------------------------------------------------------------- dialogue intro audio
RISE_GAIN = 0.35          # music level just before the drop in `rise` mode
INTRO_FADE_IN = 0.03
INTRO_FADE_OUT = 0.15
AUDIO_FMT = "aformat=sample_rates=44100:channel_layouts=stereo"


@dataclass
class IntroAudio:
    regions: List[Tuple[Path, float, float]]   # (source, start, length) in source seconds
    music_in: str = "rise"                       # "rise" | "drop"
    rise: float = 3.0


def effective_rise(rise: float, audio_start: float, t_intro: float, music_in: str):
    """Clamp the rise window to the song material before audio_start and to the intro length.
    Returns (rise_seconds, notes)."""
    if music_in == "drop":
        return 0.0, []
    r, notes = rise, []
    if audio_start < r:
        r = audio_start
        notes.append(f"note: --rise clamped to {r:.2f}s (only {audio_start:.2f}s of song before --audio-start)")
    if t_intro < r:
        r = t_intro
        notes.append(f"note: --rise clamped to {r:.2f}s (intro is only {t_intro:.2f}s long)")
    return max(r, 0.0), notes


def intro_audio_graph(lengths: Sequence[float], total: float, t_intro: float, rise: float,
                      fade: float = FADE) -> str:
    """Audio part of the final filtergraph -> [a]. Input 1 = song (already seeked to
    audio_start - rise), inputs 2.. = one per intro region. rise == 0 means `drop`."""
    parts = []
    for k, L in enumerate(lengths):
        fo = min(INTRO_FADE_OUT, L / 2)
        parts.append(f"[{2 + k}:a]atrim=duration={L:.6f},asetpts=PTS-STARTPTS,{AUDIO_FMT},"
                     f"afade=t=in:st=0:d={INTRO_FADE_IN},afade=t=out:st={L - fo:.6f}:d={fo:.3f}[d{k}]")
    if len(lengths) > 1:
        parts.append("".join(f"[d{k}]" for k in range(len(lengths)))
                     + f"concat=n={len(lengths)}:v=0:a=1[dl]")
        dl = "dl"
    else:
        dl = "d0"
    parts.append(f"[{dl}]apad=whole_dur={total:.6f}[dlg]")
    music_len = total - t_intro
    delay = t_intro - rise
    vol = ""
    if rise > 0:
        vol = f"volume='if(lt(t,{rise:.6f}),{RISE_GAIN}*t/{rise:.6f},1)':eval=frame,"
    parts.append(f"[1:a]atrim=duration={music_len + rise:.6f},asetpts=PTS-STARTPTS,{AUDIO_FMT},"
                 f"{vol}adelay={round(delay * 44100)}S|{round(delay * 44100)}S[mus]")
    parts.append(f"[dlg][mus]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95,"
                 f"afade=t=out:st={max(total - fade, 0):.6f}:d={fade}[a]")
    return ";".join(parts)


def final_args_intro(listfile: Path, audio: Path, audio_start: float, total: float, fps: int,
                     ass: Optional[Path], out: Path, intro: IntroAudio, t_intro: float,
                     fade: float = FADE, preview: bool = False) -> List[str]:
    rise, _ = effective_rise(intro.rise, audio_start, t_intro, intro.music_in)
    lengths = [L for _, _, L in intro.regions]
    fc = _video_chain(ass, preview) + ";" + intro_audio_graph(lengths, total, t_intro, rise, fade)
    args = ["-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-ss", f"{audio_start - rise:.6f}", "-i", str(audio)]
    for src, st, L in intro.regions:
        args += ["-ss", f"{st:.6f}", "-t", f"{L:.6f}", "-i", str(src)]
    args += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", *(PREVIEW_X264 if preview else FINAL_X264), "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-r", str(fps), "-movflags", "+faststart",
             "-t", f"{total:.6f}", str(out)]
    return args


def render(plan: Plan, sources: Sequence[Source], assignments: Sequence[Assignment],
           audio: Path, audio_start: float, subs: Optional[Path], out: Path,
           workers: Optional[int] = None, fit: str = "crop", sub_style: str = "pop",
           word_reveal: bool = False, intro: Optional[IntroAudio] = None,
           punch_zoom: float = PUNCH_ZOOM, loop: bool = False, preview: bool = False) -> None:
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clips_") as tmp:
        tmpd = Path(tmp)
        ass = prepare_subs(subs, tmpd, sub_style, word_reveal) if subs else None
        segs = [tmpd / f"seg_{i + 1:04d}.mp4" for i in range(len(plan.shots))]

        def job(i: int) -> None:
            a = assignments[i]
            sh = plan.shots[i]
            blend = None
            if loop and i == len(segs) - 1 and len(segs) > 1:
                a0 = assignments[0]
                blend = (sources[a0.source].path, a0.start)
            run_ffmpeg(segment_args(sources[a.source].path, a, sh.frames, plan.fps, segs[i],
                                    fit, sh.punch, sh.flash, punch_zoom, blend, preview,
                                    sh.ramp))

        print(f"rendering {len(segs)} shots with {workers} workers...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in ex.map(job, range(len(segs))):
                pass
        lst = tmpd / "list.txt"
        lst.write_text("".join(concat_line(s) for s in segs), encoding="utf-8")
        print("final pass...", file=sys.stderr)
        fade = LOOP_FADE if loop else FADE
        if intro:
            for note in effective_rise(intro.rise, audio_start, plan.t_intro, intro.music_in)[1]:
                print(note, file=sys.stderr)
            run_ffmpeg(final_args_intro(lst, Path(audio).resolve(), audio_start, plan.output_len,
                                        plan.fps, ass, out.resolve(), intro, plan.t_intro,
                                        fade, preview))
        else:
            run_ffmpeg(final_args(lst, Path(audio).resolve(), audio_start, plan.output_len,
                                  plan.fps, ass, out.resolve(), fade, preview))
