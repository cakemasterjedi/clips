"""FFmpeg binary resolution, subprocess helper and a stderr-based probe."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence


class ClipsError(Exception):
    """User-facing error."""


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover
        raise ClipsError(f"no ffmpeg found on PATH or via imageio-ffmpeg: {exc}")


def cpu_count() -> int:
    """CPUs this process may really use: the affinity mask (container cpuset) capped by a
    cgroup v2 CPU quota (e.g. a Proxmox/LXC or Docker CPU limit), where os.cpu_count() would
    report every core of the host."""
    try:
        n = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        n = os.cpu_count() or 2
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        if quota != "max":
            n = min(n, max(1, int(-(-int(quota) // int(period)))))
    except (OSError, ValueError):
        pass
    return max(1, n)


def verbose() -> bool:
    return os.environ.get("CLIPS_VERBOSE", "") not in ("", "0")


def vlog(msg: str) -> None:
    """Detail line for `clips ui`'s advanced log (CLIPS_VERBOSE=1); silent otherwise."""
    if verbose():
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def run_ffmpeg(args: Sequence[str], tail_lines: int = 25) -> subprocess.CompletedProcess:
    """Run ffmpeg with args; raise ClipsError with the stderr tail on failure."""
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *map(str, args)]
    t0 = time.time()
    if verbose():
        ins = [os.path.basename(str(args[i + 1])) for i, a in enumerate(args[:-1]) if a == "-i"]
        vlog(f"ffmpeg start: {', '.join(ins) or '-'} -> {os.path.basename(str(args[-1]))}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          errors="replace")
    if verbose():
        vlog(f"ffmpeg done in {time.time() - t0:.1f}s (exit {proc.returncode}): "
             f"{os.path.basename(str(args[-1]))}")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-tail_lines:])
        raise ClipsError(f"ffmpeg failed (exit {proc.returncode}): {' '.join(cmd)}\n{tail}")
    return proc


@dataclass
class ProbeInfo:
    duration: Optional[float]
    has_video: bool
    has_audio: bool
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float] = None
    video_streams: int = 0
    spherical: bool = False     # 360 metadata (e.g. "spherical: equirectangular" side data)


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RES = re.compile(r"(?<![\w])(\d{2,5})x(\d{2,5})(?=[,\s\[])")
_FPS = re.compile(r"(\d+(?:\.\d+)?)\s*fps")


def parse_probe(stderr: str) -> ProbeInfo:
    duration = None
    m = _DUR.search(stderr)
    if m:
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    has_video = has_audio = False
    width = height = fps = None
    nvid = 0
    for line in stderr.splitlines():
        s = line.strip()
        if "Stream #" not in s:
            continue
        if ": Video:" in s and "attached pic" not in s:
            nvid += 1
        if ": Video:" in s and "attached pic" not in s and not has_video:
            has_video = True
            body = s.split(": Video:", 1)[1]
            r = _RES.search(body)
            if r:
                width, height = int(r.group(1)), int(r.group(2))
            f = _FPS.search(body)
            if f:
                fps = float(f.group(1))
        elif ": Audio:" in s:
            has_audio = True
    return ProbeInfo(duration, has_video, has_audio, width, height, fps, nvid,
                     "spherical" in stderr.lower())


def probe(path) -> ProbeInfo:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          errors="replace", stdin=subprocess.DEVNULL)
    err = proc.stderr
    if "Invalid data found" in err or "No such file" in err:
        raise ClipsError(f"cannot read media file {path}: {err.strip().splitlines()[-1]}")
    return parse_probe(err)
