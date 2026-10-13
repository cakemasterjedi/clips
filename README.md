# clips

Local CLI that cuts source video on the beat of a music track (librosa) and burns in
user-provided subtitles (FFmpeg). Output: 1080x1920, H.264 + AAC, 30 fps (or 60).

    .venv/bin/python -m clips INPUT --audio SONG [--subs FILE.srt|.ass] -o OUT.mp4 \
        [--fps 30|60] [--beats-per-cut N] [--duration S] [--audio-start S] \
        [--shuffle] [--seed N] [--dry-run]

- INPUT: a video file, or a directory of clips (non-recursive; mp4 mov mkv webm avi m4v).
- Default cut density is automatic and energy-adaptive (louder = faster cuts, up to every
  beat plus strong onsets; quiet = ~1.6 s shots). `--beats-per-cut N` cuts every N beats.
- Shots are 0.25 s to 2.0 s. Cuts are snapped to the frame grid; frame counts sum exactly.
- `--dry-run` prints the cut plan as JSON (shots: index, section, start, end, frames, source, source_start, energy, hit, punch, flash; see the intro section below).
- Subtitles: `.ass` is burned as-is; `.srt` is converted to a styled ASS with a pop-in effect.
  **Subtitle times are relative to the output video and are NOT shifted by `--audio-start`.**
- FFmpeg: system `ffmpeg` if on PATH, else the build bundled with `imageio-ffmpeg`.

Tests: `.venv/bin/python -m pytest -q` (offline).

## Look options (all off/default unless given)

- `--fit crop|letterbox|blur`: how non-9:16 footage fills 1080x1920 (default crop; blur = darkened blurred backdrop).
- `--sub-style pop|glow|top` (.srt only): `pop` default; `glow` = centered uppercase serif with an orange glow; `top` = plain white caption above a letterboxed 16:9 frame (use with `--fit letterbox`).
- `--word-reveal` (.srt only): words appear one by one inside each cue.
- `--punch`: zoom punch (1.12 -> 1.0 in 0.18 s) on the strongest ~35% of beat hits.
- `--flash`: white flash on the strongest ~15% of beat hits.
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
