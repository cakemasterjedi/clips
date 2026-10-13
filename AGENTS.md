# AGENTS.md

<!-- handoff:learnings:begin -->
## Session Learnings (auto-updated by handoff)

### Implicit Rules
- Bundled imageio-ffmpeg has no VAAPI; GPU needs system ffmpeg
- CLIPS_VAAPI_DEVICE env selects render node
- Repo has no CI; project rule: merge when tests pass
- venv recreated with uv: uv venv -p python3.12 .venv; uv pip install -r requirements.txt pytest -e .

### Key Decisions
- Decision: only decode+encode on GPU, filters stay CPU — Reason: measured encode ~65% of flat shot, ~90% final pass, decode ~75% of 5.7K HEVC 360 shot; filters small share, GPU filters would change exact graphs
- Decision: h264_vaapi CQP qp = CPU CRF (18 segments, 20 final, 30 preview), profile high — Reason: comparable look, simple
- Decision: graph tail format=yuv420p[v] replaced by format=nv12,hwupload[v] via gpu.upload — Reason: single hook point for segment and final graphs
- Decision: -hwaccel vaapi without hwaccel_output_format on shot inputs only, not concat input — Reason: frames return to system memory for CPU filters; ffmpeg auto-falls back to software decode per stream
- Decision: detection = 0.2s 1080x1920 test encode on first /dev/dri/renderD* or CLIPS_VAAPI_DEVICE, lru_cached — Reason: catches missing driver/permission/encoder
- Decision: default --gpu auto; failed GPU encode reruns on CPU and rest of render stays CPU; --gpu vaapi raises instead — Reason: never break renders
- Decision: no AMF — Reason: needs proprietary AMD driver on Linux

<!-- handoff:learnings:end -->
