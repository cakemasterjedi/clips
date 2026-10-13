"""Two-pass render: per-shot normalized segments, then concat + subs + music."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .audio import Plan
from .ffmpeg import ClipsError, run_ffmpeg
from .gpu import Gpu, global_args, input_args, upload, video_codec
from .sources import Assignment, Source
from .reframe import View, input_labels, animate, still, track_crop, view_graph
from .subs import escape_filter_path, prepare_subs

FADE = 0.3


FITS = ("crop", "letterbox", "blur")
PUNCH_ZOOM = 0.12
PREVIEW_SIZE = (270, 480)   # --preview renders shots at quarter size, upscaled in the final pass
PREVIEW_X264 = ["-preset", "ultrafast", "-crf", "30"]
FINAL_X264 = ["-preset", "medium", "-crf", "20"]
SEGMENT_X264 = ["-preset", "veryfast", "-crf", "18"]
RAMP_DUR = 0.5            # s of output at the end of a lead-in shot played in slow motion
RAMP_SPEED = 0.4
LOOP_BLEND = 0.25         # s; the last shot dissolves into the opening frame (--loop)
LOOP_FADE = 0.02          # s; audio fade at the loop point (just enough to avoid a click)
PUNCH_DUR = 0.18
FLASH_DUR = 0.1


def zoom_graph(z: float, fps: int, inp: str, out: str, size: Tuple[int, int]) -> str:
    """Crop fit at zoom z: the clip scaled to z x the fill size, centred; below 1 the empty
    area shows a blurred, darkened fill of the same clip."""
    W, H = size
    s = f"max({W}/iw,{H}/ih)*{z:g}"
    fg = (f"scale=w='2*trunc(iw*{s}/2)':h='2*trunc(ih*{s}/2)',"
          f"crop='min(iw,{W})':'min(ih,{H})'")
    if z >= 1:
        return f"[{inp}]fps={fps},{fg},scale={W}:{H}[{out}]"
    p = "z" + out
    return (f"[{inp}]fps={fps},split[{p}a][{p}b];"
            f"[{p}a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"gblur=sigma=30,eq=brightness=-0.08[{p}bg];[{p}b]{fg}[{p}fg];"
            f"[{p}bg][{p}fg]overlay=(W-w)/2:(H-h)/2[{out}]")


def fit_graph(fit: str, fps: int, inp: str = "0:v", out: str = "f",
              size: Tuple[int, int] = (1080, 1920), view: Optional[View] = None,
              inp2: Optional[str] = None, length: float = 0.0, zoom: float = 1.0) -> str:
    """Filter graph from [inp] to [out] producing exactly `size` (1080x1920) for the fit mode.
    360 sources (view.is360) are reframed with v360 instead (fit does not apply); `track`
    pans the crop along view.track; inp2 = the second lens of a split dual-fisheye source."""
    W, H = size
    if view is not None and view.is360:
        return view_graph(view, fps, inp, inp2, out, size)
    if fit == "track":
        return (f"[{inp}]fps={fps},scale={W}:{H}:force_original_aspect_ratio=increase,"
                f"{track_crop(view, length, size)}[{out}]")
    if fit == "crop" and zoom != 1:
        return zoom_graph(zoom, fps, inp, out, size)
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
                   ramp_len: Optional[float] = None, preview: bool = False,
                   view: Optional[View] = None, blend_view: Optional[View] = None,
                   length: float = 0.0, zoom: float = 1.0) -> str:
    """Complete -filter_complex for one shot, output label [v]. blend_at (loop ending): from
    that time on, the first frame of input 1 (the opening shot) fades in over the shot.
    ramp_len (shot length in s): the last min(RAMP_DUR, len/2) s play at RAMP_SPEED, frame
    blended up to fps (plain frame repeats in preview)."""
    W, H = size
    inp, inp2 = input_labels(view or View(), 0)
    g = fit_graph(fit, fps, inp, "f", size, view, inp2, length, zoom)
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
        bidx = 2 if view is not None and view.lenses == "pair" else 1
        b1, b2 = input_labels(blend_view or View(), bidx)
        g += (";" + fit_graph(fit, fps, b1, "l0", size, blend_view, b2, LOOP_BLEND, zoom) +
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
                 ramp: bool = False, view: Optional[View] = None,
                 blend_view: Optional[View] = None, zoom: float = 1.0,
                 gpu: Optional[Gpu] = None) -> List[str]:
    args = ["-y", *global_args(gpu)]
    hw = input_args(gpu)
    loop = ["-stream_loop", "-1"] if a.loop else []
    args += loop + hw + ["-ss", f"{a.start:.6f}", "-i", str(src)]
    if view is not None and view.lenses == "pair" and view.pair is not None:
        args += loop + hw + ["-ss", f"{a.start:.6f}", "-i", str(view.pair)]
    blend_at = None
    if blend_to is not None:
        args += hw + ["-ss", f"{blend_to[1]:.6f}", "-i", str(blend_to[0])]
        if blend_view is not None and blend_view.lenses == "pair" and blend_view.pair is not None:
            args += hw + ["-ss", f"{blend_to[1]:.6f}", "-i", str(blend_view.pair)]
        blend_at = max(frames / fps - LOOP_BLEND, 0.0)
    args += ["-frames:v", str(frames), "-an",
             "-filter_complex", upload(segment_filter(fit, fps, punch, flash, punch_zoom, blend_at,
                                                      PREVIEW_SIZE if preview else (1080, 1920),
                                                      frames / fps if ramp else None, preview,
                                                      view, blend_view, frames / fps, zoom), gpu),
             "-map", "[v]",
             *video_codec(gpu, 30 if preview else 18, PREVIEW_X264 if preview else SEGMENT_X264),
             "-r", str(fps), str(out)]
    return args


def concat_line(p: Path) -> str:
    return "file '" + str(p).replace("'", "'\\''") + "'\n"


def _video_chain(ass: Optional[Path], preview: bool = False) -> str:
    vchain = "scale=1080:1920:flags=bilinear," if preview else ""   # preview segments are small
    vchain += f"ass=filename={escape_filter_path(ass)}," if ass else ""
    return f"[0:v]{vchain}setsar=1,format=yuv420p[v]"


def final_args(listfile: Path, audio: Path, audio_start: float, length: float, fps: int,
               ass: Optional[Path], out: Path, fade: float = FADE, preview: bool = False,
               gpu: Optional[Gpu] = None) -> List[str]:
    fin = f"afade=t=in:st=0:d={fade}," if fade < FADE else ""   # loop: no click at the restart
    fc = (f"{upload(_video_chain(ass, preview), gpu)};"
          f"[1:a]atrim=duration={length:.6f},asetpts=PTS-STARTPTS,{fin}"
          f"afade=t=out:st={max(length - fade, 0):.6f}:d={fade}[a]")
    return ["-y", *global_args(gpu), "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-ss", f"{audio_start:.6f}", "-i", str(audio),
            "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
            *video_codec(gpu, 30 if preview else 20, PREVIEW_X264 if preview else FINAL_X264),
            "-c:a", "aac", "-b:a", "192k", "-r", str(fps), "-movflags", "+faststart",
            "-t", f"{length:.6f}", str(out)]


# ---------------------------------------------------------------- loudness
LUFS = -14.0              # TikTok / Reels playback target
TRUE_PEAK = -1.5
LRA = 11.0


def _loudnorm(extra: str = "") -> str:
    return f"loudnorm=I={LUFS}:TP={TRUE_PEAK}:LRA={LRA}{extra}"


def measure_loudness(path: Path) -> Optional[dict]:
    """First loudnorm pass on a finished file; None if silent or unmeasurable."""
    import json
    import re
    import subprocess

    from .ffmpeg import ffmpeg_exe

    proc = subprocess.run([ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path), "-vn",
                           "-af", _loudnorm(":print_format=json"), "-f", "null", "-"],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          errors="replace")
    m = re.findall(r"\{[^{}]*\}", proc.stderr)
    if proc.returncode != 0 or not m:
        return None
    try:
        d = json.loads(m[-1])
        if float(d["input_i"]) < -70:          # silence / -inf
            return None
    except (ValueError, KeyError):
        return None
    return d


def loudnorm_args(src: Path, out: Path, m: dict, duration: float) -> List[str]:
    """Second pass: one fixed gain to reach LUFS (so dynamics and drops are kept, unlike
    loudnorm's dynamic mode), with a true-peak limiter catching the few peaks pushed over
    TRUE_PEAK. Video is copied; audio is cut to the original duration."""
    gain = LUFS - float(m["input_i"])
    limit = 10 ** (TRUE_PEAK / 20)
    return ["-y", "-i", str(src), "-map", "0:v", "-map", "0:a", "-c:v", "copy",
            "-af", f"volume={gain:.2f}dB,alimiter=limit={limit:.4f}:attack=5:release=50:level=0",
            "-c:a", "aac", "-b:a", "192k", "-t", f"{duration:.6f}",
            "-movflags", "+faststart", str(out)]


def normalize_loudness(path: Path, duration: float) -> Optional[float]:
    """Two-pass loudnorm of `path` in place. Returns the measured input LUFS (None = skipped)."""
    m = measure_loudness(path)
    if m is None:
        return None
    tmp = path.with_name(path.stem + ".loudnorm" + path.suffix)
    run_ffmpeg(loudnorm_args(path, tmp, m, duration))
    os.replace(tmp, path)
    return float(m["input_i"])


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
                     fade: float = FADE, preview: bool = False,
                     gpu: Optional[Gpu] = None) -> List[str]:
    rise, _ = effective_rise(intro.rise, audio_start, t_intro, intro.music_in)
    lengths = [L for _, _, L in intro.regions]
    fc = (upload(_video_chain(ass, preview), gpu) + ";" +
          intro_audio_graph(lengths, total, t_intro, rise, fade))
    args = ["-y", *global_args(gpu), "-f", "concat", "-safe", "0", "-i", str(listfile),
            "-ss", f"{audio_start - rise:.6f}", "-i", str(audio)]
    for src, st, L in intro.regions:
        args += ["-ss", f"{st:.6f}", "-t", f"{L:.6f}", "-i", str(src)]
    args += ["-filter_complex", fc, "-map", "[v]", "-map", "[a]",
             *video_codec(gpu, 30 if preview else 20, PREVIEW_X264 if preview else FINAL_X264),
             "-c:a", "aac", "-b:a", "192k", "-r", str(fps), "-movflags", "+faststart",
             "-t", f"{total:.6f}", str(out)]
    return args


def clip_audio(plan: Plan, sources: Sequence[Source], assignments: Sequence[Assignment],
               out: Path, mute: bool = False, rate: int = 48000) -> Path:
    """Sound track for an edit without a song: each shot's own audio (silence for clips
    without sound, or everything with mute), sample-exact per shot, as one WAV."""
    from .ffmpeg import probe

    out = Path(out)
    has = {}
    parts = []
    for i, (sh, a) in enumerate(zip(plan.shots, assignments)):
        n = round(sh.frames / plan.fps * rate)            # samples, so parts add up exactly
        part = out.with_name(f"snd_{i:04d}.wav")
        src = sources[a.source].path
        if not mute and a.source not in has:
            has[a.source] = bool((sources[a.source].info or probe(src)).has_audio)
        if mute or not has[a.source]:
            args = ["-y", "-f", "lavfi", "-i", f"anullsrc=r={rate}:cl=stereo"]
        else:
            args = ["-y", *(["-stream_loop", "-1"] if a.loop else []),
                    "-ss", f"{a.start:.6f}", "-i", str(src), "-map", "0:a:0", "-vn"]
        args += ["-af", f"aresample={rate},apad,atrim=end_sample={n}", "-ac", "2", "-ar", str(rate),
                 "-c:a", "pcm_s16le", str(part)]
        run_ffmpeg(args)
        parts.append(part)
    lst = out.with_name("snd_list.txt")
    lst.write_text("".join(concat_line(p) for p in parts), encoding="utf-8")
    run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)])
    return out


def render(plan: Plan, sources: Sequence[Source], assignments: Sequence[Assignment],
           audio: Path, audio_start: float, subs: Optional[Path], out: Path,
           workers: Optional[int] = None, fit: str = "crop", sub_style: str = "pop",
           word_reveal: bool = False, intro: Optional[IntroAudio] = None,
           punch_zoom: float = PUNCH_ZOOM, loop: bool = False, preview: bool = False,
           loudnorm: bool = True, views: Optional[Sequence[Optional[View]]] = None,
           zoom: float = 1.0, gpu: Optional[Gpu] = None, gpu_required: bool = False) -> None:
    workers = workers or max(1, (os.cpu_count() or 2) // 2)
    hw = {"gpu": gpu}
    lock = threading.Lock()

    def encode(build) -> None:
        """Run build(gpu) on the GPU; if that fails (and the GPU was not required), say so
        once and run it and everything after it on the CPU."""
        g = hw["gpu"]
        if g is not None:
            try:
                run_ffmpeg(build(g))
                return
            except ClipsError as exc:
                if gpu_required:
                    raise
                with lock:
                    if hw["gpu"] is not None:
                        hw["gpu"] = None
                        last = str(exc).strip().splitlines()[-1]
                        print(f"gpu: encode failed ({last}), continuing on the CPU",
                              file=sys.stderr, flush=True)
        run_ffmpeg(build(None))

    if gpu is not None:
        print(f"gpu: decoding and encoding with VAAPI on {gpu.device}", file=sys.stderr)
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
            vw = views[i] if views else None
            if vw is not None and vw.animated:
                txt, anim = animate(vw, sh.frames, plan.fps,
                                    PREVIEW_SIZE if preview else (1080, 1920))
                if txt:
                    cmd = tmpd / f"view_{i + 1:04d}.cmd"
                    cmd.write_text(txt, encoding="utf-8")
                    vw = replace(anim, cmds=cmd)
                else:
                    vw = still(vw)
            bv = still(views[0]) if views and blend and views[0] is not None else None
            encode(lambda g: segment_args(sources[a.source].path, a, sh.frames, plan.fps, segs[i],
                                          fit, sh.punch, sh.flash, punch_zoom, blend, preview,
                                          sh.ramp, vw, bv, zoom, g))

        print(f"rendering {len(segs)} shots with {workers} workers...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for k, _ in enumerate(ex.map(job, range(len(segs))), 1):
                if not sys.stderr.isatty():         # `clips ui` reads this for its progress bar
                    print(f"shot {k}/{len(segs)}", file=sys.stderr, flush=True)
        lst = tmpd / "list.txt"
        lst.write_text("".join(concat_line(s) for s in segs), encoding="utf-8")
        print("final pass...", file=sys.stderr)
        fade = LOOP_FADE if loop else FADE
        if intro:
            for note in effective_rise(intro.rise, audio_start, plan.t_intro, intro.music_in)[1]:
                print(note, file=sys.stderr)
            encode(lambda g: final_args_intro(lst, Path(audio).resolve(), audio_start,
                                              plan.output_len, plan.fps, ass, out.resolve(),
                                              intro, plan.t_intro, fade, preview, g))
        else:
            encode(lambda g: final_args(lst, Path(audio).resolve(), audio_start, plan.output_len,
                                        plan.fps, ass, out.resolve(), fade, preview, g))
    if loudnorm:
        was = normalize_loudness(out.resolve(), plan.output_len)
        if was is not None:
            print(f"loudness {was:.1f} -> {LUFS:.0f} LUFS", file=sys.stderr)
