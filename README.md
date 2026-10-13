# clips

Local CLI that cuts source video on the beat of a music track (librosa) and burns in
user-provided subtitles (FFmpeg). Output: 1080x1920, H.264 + AAC, 30 fps (or 60).
Everything runs on your machine; no paid APIs.

## Install (Linux)

    git clone https://github.com/cakemasterjedi/clips && cd clips
    ./install.sh              # ffmpeg check, .venv, requirements, `clips` command in ~/.local/bin
    ./install.sh --vision     # optional: --subject (local object detector, CPU)

Needs Python 3.10+. If ffmpeg is missing it tries your package manager (may ask for your
password) and otherwise falls back to a bundled build. Re-running is safe.

## Quick test

Put a few phone clips in a folder and pick a song (mp3/m4a/wav):

    clips ~/test/footage --audio ~/test/song.mp3 --duration 15 --audio-start auto --dry-run   # cut plan only, no render
    clips ~/test/footage --audio ~/test/song.mp3 --duration 15 --audio-start auto --preview -o ~/test/draft.mp4
    clips ~/test/footage --audio ~/test/song.mp3 --duration 15 --audio-start auto \
        --smart-pick --punch --flash -o ~/test/final.mp4

Add `--subs captions.srt` for burned-in captions and `--variations 3` for three versions.
Keep footage and renders outside the repo folder.

## Usage

    clips INPUT --audio SONG [--subs FILE.srt|.ass] -o OUT.mp4 \
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
- `--audio-start auto` also rewards sections that repeat elsewhere in the song (beat-synchronous chroma self-similarity, the pychorus/msaf idea), so a repeated chorus beats a loud one-off intro.
- `--audio-start auto` candidates are distinct sections at least min(window/2, 8 s) apart.
- `--export-edl FILE.edl`: also write the cut plan as a CMX3600 EDL (one video event per shot, the song and any intro dialogue as audio events, clip names in `* FROM CLIP NAME:` comments) to finish the edit in DaVinci Resolve or Premiere. Works with `--dry-run`. Effects, fit and captions are not carried over.
- `--dry-run` prints the cut plan as JSON (shots: index, section, start, end, frames, source, source_start, energy, hit, punch, flash; see the intro section below).
- Subtitles: `.ass` is burned as-is; `.srt` is converted to a styled ASS with a pop-in effect.
  **Subtitle times are relative to the output video and are NOT shifted by `--audio-start`.**
- FFmpeg: system `ffmpeg` if on PATH, else the build bundled with `imageio-ffmpeg`.

Tests: `.venv/bin/python -m pytest -q` (offline).

Audio is loudness-normalized to -14 LUFS (true peak -1.5 dB) in two passes after the final pass (measure with `loudnorm`, then one fixed gain plus a peak limiter, so drops keep their punch) (video stream copied, not re-encoded). Silent output is left alone. `--no-loudnorm` skips it.

## Look options (all off/default unless given)

- `--fit crop|letterbox|blur`: how non-9:16 footage fills 1080x1920 (default crop; blur = darkened blurred backdrop).
- `--fit track`: the 9:16 crop follows the action (motion centroid of each shot, linear pan from where it is in the first third to where it is in the last third) instead of the centre.
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
- `--dry-run` prints an object: `{intro, candidates (top 5 detected regions with scores), music_in, rise, t_intro, shots}`; each shot has `section` = `intro` or `music`. Copy a candidate into `--intro`.

## Learn a creator's style

    clips learn LINK|VIDEO|FOLDER ... -o style.json [--name NAME] [--cache DIR]
    clips INPUT --audio SONG --style style.json -o OUT.mp4

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
- It also learns how the creator frames: how many of their shots have a moving camera and how
  fast it pans (frame widths per second), slow push-ins (zoom per second) and how high in the
  frame the action sits (`framing` in the profile). See auto framing below.
- The profile is plain JSON (`pacing`, `fx`, `fit`, `intro`, `music`, `framing`, plus per-video `refs`);
  edit it by hand if a value looks off. Captions are not learned.
- Reference videos are third-party media: keep them out of the repo.

## 360 / Insta360 footage

- Native Insta360 `.insv` files work directly: put them in the input folder (or pass one). All three layouts are handled: both lenses side by side in one stream, one stream per lens, or one file per lens (`VID_..._00_NNN.insv` + `VID_..._10_NNN.insv`; the `_10_` file is read together with its `_00_` partner). Exported 360 MP4s (equirectangular, detected from their 360 metadata or a 2:1 frame of at least 3840 wide) work too.
- Each shot is reframed to a flat 1080x1920 view with FFmpeg `v360`. The view follows the motion in that shot (yaw from where the most movement is, pitch capped at +-30 deg). `--yaw DEG [--pitch DEG]` fixes the view, `--fov DEG` sets the horizontal field of view (default 60). `--360 auto|off|equirect|dfisheye` overrides detection.
- Quality vs Insta360 Studio: `.insv` is stitched with a generic 200 deg fisheye model, with no per-camera lens calibration, no seam blending, no FlowState stabilization and no horizon levelling (those need Insta360's software and gyro data). The auto view keeps away from the lens seams (yaw +-90), but a view pointed at a seam can show a visible line. For best quality, export a 360 MP4 from Insta360 Studio and use that.
- `--dry-run` reports `view` per shot (kind, lenses, yaw, pitch, fov, roll, the number of keyframes when a keyframe file is used, or the track for `--fit track`).

### Auto framing like a creator

```
clips autoframe CLIPS_FOLDER --style creator.json      # write keyframe files only
clips CLIPS_FOLDER --audio song.mp3 --style creator.json --auto-frame -o out.mp4
```

A virtual camera operator follows the action through each 360 clip the way the creator's own camera moves: a creator who rarely moves the camera gets a wide dead zone (the view only re-aims when the action drifts out of the middle), a busy one keeps the action centred; turns are capped at their pan speed; the view slowly pushes in while the camera holds, if they push in; and the pitch puts the action at the height it sits in their frames. The result is written as the clip's keyframe file (`<clip>.view.json`), so `clips frame` opens it for hand adjustment. Clips that already have a keyframe file are left alone (`clips autoframe --force` replaces them); `--dry-run` writes nothing. Without `--style`, neutral defaults are used.

With `--style` and `--fit track`, flat footage follows the same profile: the crop holds still for creators who rarely move the camera and places the action at their height in the frame.

### Following a subject (`--subject`, optional local AI)

```
./install.sh --vision                                     # once (CPU); see below for GPU
clips autoframe CLIPS_FOLDER --subject person              # 360: keyframes follow the person
clips CLIPS_FOLDER --audio song.mp3 --subject dog -o out.mp4   # 360 view / flat crop follow the dog
```

Instead of the area with the most motion, the framer follows a subject found by a local object detector (YOLOX, Apache-2.0, runs on your machine; the model file is downloaded once into your cache, about 35 MB for `--detector s`, 20 MB `tiny`, 4 MB `nano`). `--subject` takes `auto` (people first, else the most prominent thing), `biggest`, or a class name such as `person`, `dog`, `cat`, `car`, `bicycle`, `skateboard`, `horse`, `boat` (any of the 80 COCO classes, comma-separated for several). The subject is the most prominent track of that class over the clip; when it disappears behind something, a later track with similar colours takes over, and while it is not found the framer falls back to motion. 360 frames are cut into 5 normal-looking views around the horizon before detection, so people near the seam or stretched by the projection are still found. On flat footage `--subject` implies `--fit track`. Works with `--auto-frame` and `--style`, and the result stays editable in `clips frame`.

**Which one, in words** (`--subject-desc`): when several people are in view, describe the one to follow and a local vision model in [Ollama](https://ollama.com) picks it:

```
ollama pull qwen2.5vl                                      # once, in your own Ollama install; ~6 GB
clips autoframe CLIPS_FOLDER --subject-desc "the guy in the red jacket"
clips CLIPS_FOLDER --audio song.mp3 --subject dog --subject-desc "the small white dog" -o out.mp4
```

The detector and tracker run as usual; then the up to 6 most prominent candidates are each shown to the model once as a close-up (everything but the candidate darkened) and scored 0-10 against the description. The best match (5 or more) becomes the subject. If Ollama is not running, the model is not pulled, or nothing matches, a warning is printed and the most prominent candidate is used. Answers are cached per clip and description. `--ollama-model` picks another vision model (e.g. `llava`, `moondream` for low memory); `OLLAMA_HOST` sets where Ollama listens. Ollama runs on AMD GPUs on Linux through its own ROCm build (RX 6700 XT needs `HSA_OVERRIDE_GFX_VERSION=10.3.0` set for the Ollama service).

Speed: about one minute of CPU per minute of 360 footage with `--detector s` (5 tiles every 0.5 s), roughly 3x faster with `tiny`; results are cached per clip, so variations and re-renders are instant. `--device auto|cpu|gpu` (default auto uses a GPU when a GPU build of ONNX Runtime is installed):

- Windows, any GPU including AMD Radeon (e.g. RX 6700 XT): `.venv/bin/pip uninstall onnxruntime` then `.venv/bin/pip install onnxruntime-directml`.
- Linux with an AMD GPU: install ROCm and MIGraphX, then `.venv/bin/pip install onnxruntime-migraphx`. RX 6000 cards that ROCm does not list (6700 XT = gfx1031) need `HSA_OVERRIDE_GFX_VERSION=10.3.0` in the environment.
- NVIDIA: `onnxruntime-gpu`. macOS: the standard package already uses CoreML.

### Framing by hand: `clips frame`

```
clips frame VID_20261013_120000_00_001.insv
```

Opens a local page (http://127.0.0.1:8360, nothing leaves the machine) that plays the clip as a 360 view in the 9:16 frame of the final video:

- Drag to aim, mouse wheel or `+`/`-` to zoom, `Q`/`E` to roll (`Shift` for 5 deg steps, `R` levels the horizon), or click the panorama to look there.
- `K` sets a keyframe at the playhead, `Del` removes it, `[`/`]` jump between keyframes, `Space` plays with the view following the keyframes, arrow keys step a frame (`Shift`: 1 s).
- `Ctrl+S` (or Save) writes `<clip>.view.json` next to the clip, e.g. `VID_..._00_001.insv.view.json`.

Every render then uses that file for the clip: whichever part of the clip a shot uses, the view follows the keyframes in clip time (eased between keyframes, held before the first and after the last). The keyframe file beats the automatic view; `--yaw` beats both. The file is plain JSON (`{"version": 1, "keyframes": [{"t": 1.5, "yaw": 40, "pitch": -5, "fov": 60, "roll": 0}]}`, `t` in seconds from the clip start), so it can be written by hand too.

The page plays a lightweight equirectangular preview, built once per clip and cached in `~/.cache/clips/proxies` (`--cache` to move it, `--proxy-width` to change its size). Options: `--port`, `--no-browser`, `--360 equirect|dfisheye` to force the layout. Moving views cost render time: an 8 s shot that pans, tilts, zooms and rolls the whole way renders in roughly 2.5x the time of a fixed view; a pan alone costs little.

