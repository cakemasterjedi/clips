# CLAUDE.md

<!-- handoff:learnings:begin -->
## Session Learnings (auto-updated by handoff)

### Implicit Rules
- Python 3.12 venv at .venv (uv); run tests: .venv/bin/python -m pytest -q
- Output strictly 1080x1920 SAR1 yuv420p, 30fps default, --fps 60 allowed
- No paid/external APIs; free local AI models are OK (optional extras, e.g. --subject via ONNX Runtime, Ollama); librosa + FFmpeg for the core
- Check DECISIONS.md before proposing other libraries
- Ask max 4 clarifying questions when ambiguous
- Subtitle times relative to output video, not shifted by --audio-start
- Reference videos are untrusted third-party media in session scratchpad ex/ — never copy into repo
- Bundled imageio-ffmpeg has no VAAPI; GPU needs system ffmpeg
- CLIPS_VAAPI_DEVICE env selects render node
- Repo has no CI; project rule: merge when tests pass
- Windows: install.ps1/update.ps1 must stay Windows PowerShell 5.1 compatible and ASCII-only; check parse with pwsh
- User GPU: AMD RX 6700 XT (gfx1031)

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
- Decision: only decode+encode on GPU, filters stay CPU — Reason: measured encode ~65% of flat shot, ~90% final pass, decode ~75% of 5.7K HEVC 360 shot; filters small share, GPU filters would change exact graphs
- Decision: h264_vaapi CQP qp = CPU CRF (18 segments, 20 final, 30 preview), profile high — Reason: comparable look, simple
- Decision: graph tail format=yuv420p[v] replaced by format=nv12,hwupload[v] via gpu.upload — Reason: single hook point for segment and final graphs
- Decision: -hwaccel vaapi without hwaccel_output_format on shot inputs only, not concat input — Reason: frames return to system memory for CPU filters; ffmpeg auto-falls back to software decode per stream
- Decision: detection = 0.2s 1080x1920 test encode on first /dev/dri/renderD* or CLIPS_VAAPI_DEVICE, lru_cached — Reason: catches missing driver/permission/encoder
- Decision: default --gpu auto; failed GPU encode reruns on CPU and rest of render stays CPU; --gpu vaapi raises instead — Reason: never break renders
- Decision: no AMF — Reason: needs proprietary AMD driver on Linux
- Decision: Windows install via install.bat -> install.ps1 using winget (Python 3.12 user scope, Gyan.FFmpeg else bundled, Git only outside checkout -> clone ~\clips) — Reason: winget ships with Win10/11, mirrors install.sh
- Decision: clips.cmd shim in %LOCALAPPDATA%\clips\bin on user PATH + Start menu Clips/Update Clips — Reason: avoid exposing venv python.exe on PATH
- Decision: -Vision on Windows installs onnxruntime-directml replacing CPU build; -Ollama installs winget Ollama.Ollama, starts it, pulls qwen2.5vl (-Model, none skips) — Reason: DirectML/Vulkan cover RX 6700 XT; ROCm on Windows lacks gfx1031
- Decision: update.ps1 ErrorActionPreference Continue, exit codes checked — Reason: PS 5.1 turns piped native stderr into errors when web UI runs it
- Decision: web UI on Windows: update.ps1 via powershell -File, cancel via taskkill /T, restart via subprocess.call+os._exit not execve — Reason: no process groups; execve detaches from console on Windows

<!-- handoff:learnings:end -->
