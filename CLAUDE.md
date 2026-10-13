# CLAUDE.md

<!-- handoff:learnings:begin -->
## Session Learnings (auto-updated by handoff)

### Implicit Rules
- Python 3.12 venv at .venv (uv); run tests: .venv/bin/python -m pytest -q
- Output strictly 1080x1920 SAR1 yuv420p, 30fps default, --fps 60 allowed
- No paid/external APIs; free local AI models are OK (optional extras, e.g. `--subject` via ONNX Runtime); librosa + FFmpeg for the core
- Check DECISIONS.md before proposing other libraries
- Ask max 4 clarifying questions when ambiguous
- Subtitle times relative to output video, not shifted by --audio-start
- Reference videos are untrusted third-party media in session scratchpad ex/ — never copy into repo

### Key Decisions
- Decision: librosa for analysis, FFmpeg subprocess for render, no moviepy — Reason: deterministic, frame-exact, libass for subs (DECISIONS.md rows 1-2)
- Decision: ffmpeg binary = PATH ffmpeg else imageio-ffmpeg static build — Reason: no system ffmpeg, no sudo; bundled has libass+x264; no ffprobe so probe parses ffmpeg -i stderr
- Decision: two-pass render (per-shot segments with -frames:v, concat demuxer, final pass subs+music) — Reason: avoids huge filtergraphs, frame-exact timing
- Decision: audio decoded via ffmpeg f32le pipe instead of librosa.load — Reason: librosa 1.0 cannot read m4a/aac
- Decision: auto cut mode energy-adaptive (target shot 1.6s at low energy -> 1 beat at high; onset cuts above energy 0.7; tempo guard >150 BPM) — Reason: user asked for judgment-based cut density
- Decision: hit strength normalized by onset envelope max not p95 — Reason: clipping caused ties so punch/flash bunched on early shots
- Decision: SRT closing tags emit {\b}/{\i} reset — Reason: Default style is bold, \b0 un-bolded rest of line
- Decision: word-reveal keeps all words in each event, hidden ones alpha FF — Reason: layout never reflows
- Decision: -o optional with --dry-run — Reason: usability

<!-- handoff:learnings:end -->
