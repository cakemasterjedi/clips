"""FFmpeg binary resolution, subprocess helper and a stderr-based probe."""
from __future__ import annotations

import re
import shutil
import subprocess
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


def run_ffmpeg(args: Sequence[str], tail_lines: int = 25) -> subprocess.CompletedProcess:
    """Run ffmpeg with args; raise ClipsError with the stderr tail on failure."""
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", *map(str, args)]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          errors="replace")
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
    for line in stderr.splitlines():
        s = line.strip()
        if "Stream #" not in s:
            continue
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
    return ProbeInfo(duration, has_video, has_audio, width, height, fps)


def probe(path) -> ProbeInfo:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                          errors="replace", stdin=subprocess.DEVNULL)
    err = proc.stderr
    if "Invalid data found" in err or "No such file" in err:
        raise ClipsError(f"cannot read media file {path}: {err.strip().splitlines()[-1]}")
    return parse_probe(err)
