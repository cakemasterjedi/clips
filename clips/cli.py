"""Command line interface."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import audio as audio_mod
from . import footage as footage_mod
from . import intro as intro_mod
from . import sources as src_mod
from . import style as style_mod
from .ffmpeg import ClipsError, probe
from .subs import check_subs_path


def audio_start_arg(v: str):
    """--audio-start: seconds (>= 0) or 'auto'."""
    if v.strip().lower() == "auto":
        return "auto"
    try:
        return float(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds or 'auto', got {v!r}") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clips", description="Beat-synced 9:16 TikTok edits.")
    p.add_argument("input", help="video file or directory of clips")
    p.add_argument("--audio", required=True, help="music track")
    p.add_argument("--subs", help=".srt or .ass (times relative to the output video)")
    p.add_argument("-o", "--output", help="output .mp4 (required unless --dry-run)")
    p.add_argument("--fps", type=int, choices=(30, 60), default=30)
    p.add_argument("--beats-per-cut", type=int, metavar="N", help="cut every N beats (disables auto)")
    p.add_argument("--duration", type=float, help="max output length in seconds")
    p.add_argument("--audio-start", type=audio_start_arg, default=None, metavar="SECONDS|auto",
                   help="start offset in the song (s), or 'auto' to pick the drop / most energetic part")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--loop", action="store_true",
                   help="seamless replay: end on the beat that sounds most like the start and "
                        "dissolve the last shot into the first frame")
    p.add_argument("--preview", action="store_true",
                   help="fast draft: shots rendered small and upscaled, quick low-quality encode")
    p.add_argument("--smart-pick", action="store_true",
                   help="give the busiest, sharpest footage to the opening shot and strongest hits")
    p.add_argument("--seed", type=int)
    p.add_argument("--fit", choices=("crop", "letterbox", "blur"), default=None,
                   help="how to fit non-9:16 footage (default crop)")
    p.add_argument("--sub-style", choices=("pop", "glow", "top"), default="pop",
                   help="caption style for .srt input (default pop)")
    p.add_argument("--word-reveal", action="store_true", help="reveal captions word by word (.srt only)")
    p.add_argument("--flash", action="store_true", default=None, help="white flash on the strongest beat hits")
    p.add_argument("--punch", action="store_true", default=None, help="zoom punch on strong beat hits")
    p.add_argument("--intro", action="append", default=[], metavar="PATH@START-END",
                   help="dialogue intro region (source audio, no music); repeatable")
    p.add_argument("--no-intro", action="store_true", help="disable the (auto-detected) dialogue intro")
    p.add_argument("--intro-max", type=float, default=None, metavar="SECONDS",
                   help="cap on the auto-detected intro length (default 8)")
    p.add_argument("--music-in", choices=("rise", "drop"), default=None,
                   help="how the music enters after the intro (default rise)")
    p.add_argument("--rise", type=float, default=None, metavar="SECONDS",
                   help="length of the music rise under the dialogue (default 3)")
    p.add_argument("--style", metavar="PROFILE.json",
                   help="edit like a learned creator profile (see `clips learn`); flags still win")
    p.add_argument("--dry-run", action="store_true", help="print cut plan as JSON and exit")
    return p


DEFAULTS = {"audio_start": 0.0, "fit": "crop", "punch": False, "flash": False,
            "intro_max": intro_mod.DEFAULT_INTRO_MAX, "music_in": "rise", "rise": 3.0}


def apply_style(args) -> dict:
    """Fill flags the user did not give from --style, then from DEFAULTS. Returns the
    planning knobs that have no flag of their own."""
    knobs = {"max_shot": audio_mod.AUTO_MAX_SHOT, "onset_cuts": True,
             "punch_frac": audio_mod.PUNCH_FRAC, "flash_frac": audio_mod.FLASH_FRAC,
             "punch_zoom": None, "lead": 0.0}
    if args.style:
        prof = style_mod.load_profile(args.style)
        pace, fx, mus, intro = (prof.get(k, {}) for k in ("pacing", "fx", "music", "intro"))
        if args.beats_per_cut is None and pace.get("beats_per_cut"):
            args.beats_per_cut = int(pace["beats_per_cut"])
        knobs["max_shot"] = float(pace.get("max_shot", knobs["max_shot"]))
        knobs["onset_cuts"] = bool(pace.get("onset_cuts", True))
        if args.punch is None:
            args.punch = fx.get("punch", 0) > 0
        if args.flash is None:
            args.flash = fx.get("flash", 0) > 0
        if fx.get("punch", 0) > 0:
            knobs["punch_frac"] = float(fx["punch"])
            knobs["punch_zoom"] = float(fx.get("punch_zoom") or 0) or None
        if fx.get("flash", 0) > 0:
            knobs["flash_frac"] = float(fx["flash"])
        if args.fit is None:
            args.fit = prof.get("fit")
        if not args.intro and not args.no_intro and not intro.get("use", True):
            args.no_intro = True
        if args.intro_max is None and intro.get("max"):
            args.intro_max = float(intro["max"])
        if args.music_in is None:
            args.music_in = mus.get("music_in")
        if args.rise is None and mus.get("music_in") == "rise":
            args.rise = float(mus.get("rise") or DEFAULTS["rise"])
        if args.audio_start is None:
            args.audio_start = "auto"
            knobs["lead"] = float(mus.get("lead_in") or 0.0)
    for k, v in DEFAULTS.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return knobs


def run(args) -> int:
    knobs = apply_style(args)
    if not args.output and not args.dry_run:
        raise ClipsError("-o/--output is required unless --dry-run")
    if args.beats_per_cut is not None and args.beats_per_cut < 1:
        raise ClipsError("--beats-per-cut must be >= 1")
    if args.duration is not None and args.duration <= 0:
        raise ClipsError("--duration must be > 0")
    if args.audio_start != "auto" and args.audio_start < 0:
        raise ClipsError("--audio-start must be >= 0")
    if not Path(args.audio).is_file():
        raise ClipsError(f"audio file not found: {args.audio}")
    if args.subs:
        sp = check_subs_path(args.subs)
        if sp.suffix.lower() == ".ass" and (args.sub_style != "pop" or args.word_reveal):
            print("warning: --sub-style/--word-reveal apply to .srt only; .ass is used as-is",
                  file=sys.stderr)
    if args.intro_max <= 0:
        raise ClipsError("--intro-max must be > 0")
    if args.rise < 0:
        raise ClipsError("--rise must be >= 0")
    sources = src_mod.discover(args.input)
    alen = audio_mod.audio_length(args.audio)
    inp = Path(args.input)

    # ---- dialogue intro (explicit, auto-detected or none)
    regions, candidates = [], []
    if args.no_intro:
        pass
    elif args.intro:
        regions = intro_mod.resolve_specs(args.intro, inp if inp.is_dir() else None)
    else:
        pref = sources[src_mod.first_source_index(len(sources), args.shuffle, args.seed)].path
        imax = args.intro_max if args.duration is None else min(args.intro_max, args.duration / 2)
        regions, candidates = intro_mod.auto_intro(sources, imax, audio_mod.decode, pref)
        if not regions:
            print("note: no dialogue found in the source audio; skipping the intro", file=sys.stderr)
    fps = args.fps
    ifr = intro_mod.intro_frames(regions, fps)
    t_intro = sum(ifr) / fps
    if args.duration is not None and t_intro >= args.duration - 1e-9:
        raise ClipsError(f"intro ({t_intro:.2f}s) is not shorter than --duration ({args.duration}s)")
    section_cands = []
    if args.audio_start == "auto":
        window = audio_mod.AUTO_WINDOW if args.duration is None else args.duration - t_intro
        audio_start, section_cands = audio_mod.auto_audio_start(args.audio, window, knobs["lead"])
        print(f"note: --audio-start auto picked {audio_start:.2f}s", file=sys.stderr)
    else:
        audio_start = args.audio_start
    music_len = audio_mod.output_length(alen, audio_start,
                                        None if args.duration is None else args.duration - t_intro)
    loop_sim = None
    if args.loop:
        music_len, loop_sim = audio_mod.loop_length(args.audio, audio_start, music_len)
        print(f"note: --loop ends the music at {music_len:.2f}s (match {loop_sim:.2f})",
              file=sys.stderr)
    an = audio_mod.analyze(args.audio, audio_start, music_len)
    plan = audio_mod.plan_from_analysis(an, fps, args.beats_per_cut, args.punch, args.flash,
                                        knobs["max_shot"], knobs["onset_cuts"],
                                        knobs["punch_frac"], knobs["flash_frac"])
    # music section starts after any footage the intro already used from each clip
    start_at = [max([r.end for r in regions if r.source == s.path], default=0.0) for s in sources]
    if args.smart_pick:
        reserved = {i: [(r.start, r.end) for r in regions if r.source == s.path]
                    for i, s in enumerate(sources)}
        music_assigns = footage_mod.assign_smart([s.length for s in plan.shots],
                                                 [s.hit for s in plan.shots], sources,
                                                 footage_mod.analyse(sources), reserved)
    else:
        music_assigns = src_mod.assign([s.length for s in plan.shots], sources, args.shuffle,
                                       args.seed, start_at)
    all_sources = list(sources)
    intro_assigns = []
    if regions:
        for r in regions:
            idx = next((i for i, s in enumerate(all_sources) if s.path == r.source), None)
            if idx is None:
                all_sources.append(src_mod.Source(r.source, probe(r.source).duration or r.end))
                idx = len(all_sources) - 1
            intro_assigns.append(src_mod.Assignment(idx, r.start))
        plan = audio_mod.compose_plan(ifr, plan)
    assigns = intro_assigns + music_assigns
    if args.dry_run:
        shots = [{"index": s.index, "section": s.section, "start": round(s.start, 4),
                  "end": round(s.end, 4), "frames": s.frames,
                  "source": str(all_sources[a.source].path), "source_start": a.start,
                  "energy": s.energy, "loop": a.loop,
                  "hit": s.hit, "punch": s.punch, "flash": s.flash}
                 for s, a in zip(plan.shots, assigns)]
        print(json.dumps({"intro": [r.to_dict() for r in regions],
                          "candidates": [r.to_dict() for r in candidates],
                          "audio_start": round(audio_start, 4),
                          "audio_start_lead": knobs["lead"],
                          "style": args.style, "loop_match": loop_sim, "fit": args.fit,
                          "audio_start_candidates": [{"start": t, "score": sc}
                                                     for t, sc in section_cands],
                          "music_in": args.music_in, "rise": args.rise,
                          "t_intro": round(t_intro, 4), "shots": shots}, indent=2))
        return 0
    from .render import IntroAudio, render

    print(f"mode={plan.mode} tempo={plan.tempo:.1f} shots={len(plan.shots)} "
          f"length={plan.output_len:.2f}s intro={t_intro:.2f}s", file=sys.stderr)
    ia = None
    if regions:
        ia = IntroAudio([(r.source, r.start, n / fps) for r, n in zip(regions, ifr)],
                        args.music_in, args.rise)
    render(plan, all_sources, assigns, Path(args.audio), audio_start, args.subs,
           Path(args.output), fit=args.fit, sub_style=args.sub_style,
           word_reveal=args.word_reveal, intro=ia,
           loop=args.loop, preview=args.preview,
           **({"punch_zoom": knobs["punch_zoom"]} if knobs["punch_zoom"] else {}))
    print(f"wrote {args.output}", file=sys.stderr)
    return 0


def build_learn_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="clips learn",
                                description="Learn a creator's editing style from their videos.")
    p.add_argument("refs", nargs="+", help="links (downloaded with yt-dlp), video files or folders")
    p.add_argument("-o", "--output", required=True, help="profile to write (.json)")
    p.add_argument("--name", help="creator name stored in the profile (default: output file name)")
    p.add_argument("--cache", default=str(style_mod.DEFAULT_CACHE),
                   help=f"download folder for links (default {style_mod.DEFAULT_CACHE})")
    return p


def run_learn(args) -> int:
    prof = style_mod.learn(args.refs, args.name or Path(args.output).stem, Path(args.cache))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prof, indent=2) + "\n", encoding="utf-8")
    pc, fx, mu = prof["pacing"], prof["fx"], prof["music"]
    print(f"wrote {out}: shots median {pc['median_shot']}s, on beat {pc['on_beat']:.0%}, "
          f"punch {fx['punch']:.0%}, flash {fx['flash']:.0%}, fit {prof['fit']}, "
          f"intro {'yes' if prof['intro']['use'] else 'no'}, music {mu['music_in']}",
          file=sys.stderr)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    learning = bool(argv) and argv[0] == "learn"
    args = build_learn_parser().parse_args(argv[1:]) if learning else build_parser().parse_args(argv)
    try:
        return run_learn(args) if learning else run(args)
    except ClipsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
