# Previous Decisions

Check here before proposing alternative libraries or approaches.

| # | Date | Decision | Why |
|---|------|----------|-----|
| 1 | 2026-10-13 | Audio analysis uses `librosa` (beat_track, onset_strength, RMS). No AI models. | Deterministic, local, free. |
| 2 | 2026-10-13 | Rendering uses FFmpeg via subprocess. No moviepy. | Fast, frame-exact, libass for subtitles. |
| 3 | 2026-10-13 | FFmpeg binary: system `ffmpeg` if on PATH, else the static build from `imageio-ffmpeg`. | No sudo needed; bundled build has libass + libx264. |
| 4 | 2026-10-13 | Subtitles: `.ass` burned as-is via the `ass` filter; `.srt` converted to styled, animated ASS first. | Kinetic typography from user-provided files only. No paid transcription. |
| 5 | 2026-10-13 | Output: 1080x1920, 30 fps default (`--fps 60` optional), H.264 + AAC. | TikTok spec. |
| 6 | 2026-10-13 | Video input: a single file or a folder of clips. | User request. |
| 7 | 2026-10-13 | Cut density: automatic, energy-adaptive (high energy = every beat plus strong onsets, low energy = longer shots). `--beats-per-cut N` overrides. | User asked for a judgment-based system. |
| 8 | 2026-10-13 | Two-pass render: normalize each shot to its own segment file (frame-exact), concat, then one final pass burns subs and lays the music. | Avoids huge filtergraphs and timing drift. |
| 9 | 2026-10-13 | No paid/external APIs anywhere. | Hard constraint. |
| 10 | 2026-10-13 | `--fit crop\|letterbox\|blur` (default crop) applied in the per-shot segment filter. | Non-9:16 footage needs a choice between filling, bars, or a blurred backdrop. |
| 11 | 2026-10-13 | `--sub-style pop\|glow\|top` for `.srt` input only; `.ass` stays untouched (flags warn). | Caption looks without needing a hand-written ASS. |
| 12 | 2026-10-13 | `--word-reveal`: one ASS event per word slice, all words always present (hidden via alpha) so layout never shifts. | Kinetic reveal without reflow. |
| 13 | 2026-10-13 | Beat FX `--punch` (zoom 1.12 to 1.0 over 0.18 s, top 35% of hits) and `--flash` (white fade-in, top 15%); hit = onset envelope near the cut. Both off by default. | Accent the strongest cuts. |
| 14 | 2026-10-13 | Variable pacing: auto target = beat + (3.5 s - beat) * (1 - energy)^1.5, MAX_SHOT 4.0 s, plus a post-hit hold (next shot >= min(4 beats, 2.5 s) after a top-12% hit, at most one per 8 s). | User wants some scenes to breathe; cuts need not be uniform. |
| 15 | 2026-10-13 | Dialogue intro: source video with its own audio and no music, then the music section. Auto-detected by local DSP (`clips/speech.py`: voice-band ratio, flatness, relative RMS, 3-8 Hz modulation, beat-regularity penalty), or given with `--intro PATH@START-END`; `--no-intro` disables. One shot per region, no cuts, no FX. | Dialogue builds tension before the music; no models allowed, so speech detection is heuristic. |
| 16 | 2026-10-13 | `--music-in rise\|drop`: drop starts music at full volume exactly at the end of the intro; rise starts it `--rise` s earlier (default 3) ramping 0 to 0.35 under the dialogue, then full. Cut analysis always uses audio from `--audio-start` for (total - intro). `--duration` is the total length including the intro. | Music entry should line up with what is analysed for cuts. User confirmed `rise` as the default after comparing test renders. |
| 17 | 2026-10-13 | Music section skips footage the intro already used (per-clip start offset) and opens on a different clip than the intro; same-clip intro regions < 1 s apart merge into one shot. | Avoid replaying intro footage, make the music entry a visible cut, avoid jump cuts mid-dialogue. |
| 18 | 2026-10-13 | `--audio-start auto`: score each detected beat (and 0) as a start for the music window (`--duration` minus intro, else 20 s): mean RMS + 0.5 x mean onset strength (both max-normalized over the song) + max(0, RMS in next 2 s - RMS in previous 4 s). Highest wins, ties to the earlier start. | Picking the right part of the song (the drop) matters most for retention; librosa only, no models. |
