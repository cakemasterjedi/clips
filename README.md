# clips

Local CLI that cuts source video on the beat of a music track (librosa) and burns in
user-provided subtitles (FFmpeg). Output: 1080x1920, H.264 + AAC, 30 fps (or 60).

    .venv/bin/python -m clips INPUT --audio SONG [--subs FILE.srt|.ass] -o OUT.mp4 \
        [--fps 30|60] [--beats-per-cut N] [--duration S] [--audio-start S|auto] \
        [--shuffle] [--seed N] [--dry-run]

- INPUT: a video file, or a directory of clips (non-recursive; mp4 mov mkv webm avi m4v).
- Default cut density is automatic and energy-adaptive (louder = faster cuts, up to every
  beat plus strong onsets; quiet = ~1.6 s shots). `--beats-per-cut N` cuts every N beats.
- Shots are 0.25 s to 2.0 s. Cuts are snapped to the frame grid; frame counts sum exactly.
- `--audio-start auto` picks where the song starts: the beat that opens the most energetic
  window of the output's music length (`--duration` minus the intro; 20 s if no `--duration`),
  with a bonus for starting right on a drop (RMS jump vs the 4 s before). Local librosa only.
  The pick is printed on stderr; `--dry-run` reports `audio_start` and the top 5
  `audio_start_candidates`. With `--music-in rise` the rise plays the build-up before the drop.
- `--variations N` (2-10): renders `OUT_v1.mp4` .. `OUT_vN.mp4`. v1 is exactly what you asked for; the others reshuffle the footage (seed + k) and, when the song start is automatic, use the next best distinct song sections. With `--dry-run` it prints `{"variations": [plan, ...]}`. Combine with `--preview` for quick comparisons. (With `--smart-pick` the footage choice is deterministic, so only the song section changes.)
- `--audio-start auto` candidates are distinct sections at least min(window/2, 8 s) apart.
- `--dry-run` prints the cut plan as JSON (shots: index, section, start, end, frames, source, source_start, energy, hit, punch, flash; see the intro section below).
- Subtitles: `.ass` is burned as-is; `.srt` is converted to a styled ASS with a pop-in effect.
  **Subtitle times are relative to the output video and are NOT shifted by `--audio-start`.**
- FFmpeg: system `ffmpeg` if on PATH, else the build bundled with `imageio-ffmpeg`.

Tests: `.venv/bin/python -m pytest -q` (offline).

## Look options (all off/default unless given)

- `--fit crop|letterbox|blur`: how non-9:16 footage fills 1080x1920 (default crop; blur = darkened blurred backdrop).
- `--sub-style pop|glow|top` (.srt only): `pop` default; `glow` = centered uppercase serif with an orange glow; `top` = plain white caption above a letterboxed 16:9 frame (use with `--fit letterbox`).
- `--word-reveal` (.srt only): words appear one by one inside each cue.
- `--smart-pick`: instead of walking each clip in order, score all footage (motion + sharpness, local DSP) and give the best windows to the opening shot first, then to the strongest beat hits. Black, blank/blurry, frozen footage and windows containing a cut inside the clip are avoided; windows never overlap and neighbouring shots come from different clips when possible.
- `--loop`: seamless replay. The music ends on the beat (within the last 8 beats before the requested end, never later) whose continuation sounds most like the opening beat (chroma + MFCC similarity, Infinite-Jukebox style), the audio only gets a 20 ms fade at both ends, and the last 0.25 s of video dissolves into the opening frame. `--dry-run` reports `loop_match` (similarity).
- `--preview`: fast draft. Shots are rendered at 270x480 with an ultrafast encode and upscaled to 1080x1920 in the final pass (subtitles are burned at full size), so the output spec is unchanged but it looks soft. Use it to check cuts before the real render.
- `--punch`: zoom punch (1.12 -> 1.0 in 0.18 s) on the strongest ~35% of beat hits.
- `--flash`: white flash on the strongest ~15% of beat hits.
- `--ramp`: speed ramps on the biggest hits (top ~15% by hit strength plus the energy lift across the cut). The shot before the hit plays its last 0.5 s (at most half the shot) at 0.4x speed, frame-blended, and the hit shot snaps back to normal speed. Lead-in shots under 0.5 s are skipped; `--preview` uses plain frame repeats. `--dry-run` reports `ramp` per shot.
- `.ass` input ignores `--sub-style`/`--word-reveal` (with a warning). `--dry-run` JSON also reports `hit`, `punch`, `flash`.

## Pacing and dialogue intro

- Auto pacing lets quiet passages breathe (up to ~3.5 s targets, hard max 4 s) and holds the shot after a strong hit (at most one hold per 8 s). `--beats-per-cut` is unchanged.
- Intro: the timeline is `[dialogue intro: source video + its own audio, no music]` then `[music section]`.
  - Default: dialogue is auto-detected locally (heuristic DSP, no models). If none is found a note is printed and no intro is used.
  - `--intro PATH@START-END` (repeatable; times as seconds or mm:ss[.ms]; split on the last `@`; PATH as given, else relative to the input folder) overrides detection. `--no-intro` disables it. `--intro-max SECONDS` (default 8) caps the auto intro.
  - Each intro region is one uncut shot (no punch/flash); `--fit` applies.
  - `--music-in rise|drop` (default rise) and `--rise SECONDS` (default 3): `drop` = music at full volume exactly when the intro ends; `rise` = music fades in from silence to 0.35 under the last `--rise` seconds of the intro, then jumps to full. If `--audio-start` < `--rise` the rise is clamped (note on stderr).
  - `--duration` is the TOTAL length including the intro (error if the intro is not shorter). Cuts are analysed on the song from `--audio-start` for (total - intro).
- `--dry-run` now prints an object: `{intro, candidates (top 5 detected regions with scores), music_in, rise, t_intro, shots}`; each shot has `section` = `intro` or `music`. Copy a candidate into `--intro`.

## Learn a creator's style

    .venv/bin/python -m clips learn LINK|VIDEO|FOLDER ... -o style.json [--name NAME] [--cache DIR]
    .venv/bin/python -m clips INPUT --audio SONG --style style.json -o OUT.mp4

- `learn` downloads links with `yt-dlp` into `--cache` (default `~/.cache/clips/refs`, outside
  the repo; already downloaded videos are reused) and analyses local files in place. Give it
  several videos from one creator; one profile per creator.
- What it measures (local FFmpeg + librosa, no models): cuts (frame-difference spikes), how many
  land on the beat, a fixed beats-per-cut if the creator uses one, how shot length follows
  loudness (fitted to the auto pacing curve), share of extra off-beat cuts, punch zooms (zoom-out
  over the first frames of a shot) and their size, white flashes, fit (letterbox / blurred
  backdrop / crop), a dialogue intro and its length, whether the music rises under it or drops
  in (and for how long), and how long after the music starts the big drop lands (`lead_in`).
- `--style` fills every flag you did not give: `--beats-per-cut`, `--punch`/`--flash` (with the
  learned share of shots and zoom size), `--fit`, intro on/off and `--intro-max`, `--music-in`,
  `--rise`, and `--audio-start auto` shifted so the song's drop lands `lead_in` seconds into the
  music, like in the references. Flags you pass always win.
- The profile is plain JSON (`pacing`, `fx`, `fit`, `intro`, `music`, plus per-video `refs`);
  edit it by hand if a value looks off. Captions are not learned.
- Reference videos are third-party media: keep them out of the repo.

