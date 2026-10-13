"""Optional GPU encode/decode through VAAPI (AMD/Intel on Linux, via Mesa or intel-media).

Only decoding and H.264 encoding move to the GPU; every filter (crop, scale, v360, blur,
captions) still runs on the CPU, where the graphs are exact and portable. Measured on the
test footage, encoding is ~60-90% of a flat shot's render time and decoding dominates 5.7K
360 shots, so this is where the time goes.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Tuple

from .ffmpeg import ClipsError, ffmpeg_exe

MODES = ("auto", "vaapi", "off")
DEVICE_ENV = "CLIPS_VAAPI_DEVICE"


@dataclass(frozen=True)
class Gpu:
    device: str                 # VAAPI render node, e.g. /dev/dri/renderD128


def render_nodes() -> List[str]:
    env = os.environ.get(DEVICE_ENV, "").strip()
    return [env] if env else sorted(glob.glob("/dev/dri/renderD*"))


@lru_cache(maxsize=None)
def _encoders() -> str:
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-encoders"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, text=True, errors="replace",
                          stdin=subprocess.DEVNULL)
    return proc.stdout


@lru_cache(maxsize=None)
def probe_device(device: str) -> Optional[str]:
    """None if a short 1080x1920 h264_vaapi encode works on device, else why not."""
    if not os.path.exists(device):
        return f"{device} does not exist"
    if not os.access(device, os.R_OK | os.W_OK):
        return f"no permission to use {device} (add your user to the 'render' group)"
    cmd = [ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
           *global_args(Gpu(device)), "-f", "lavfi", "-i", "color=black:s=1080x1920:r=30:d=0.2",
           "-vf", "format=nv12,hwupload", *video_codec(Gpu(device), 20), "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                              errors="replace", timeout=30)
    except subprocess.TimeoutExpired:
        return f"test encode on {device} timed out"
    if proc.returncode != 0:
        lines = proc.stderr.strip().splitlines()
        return f"test encode on {device} failed: {lines[-1] if lines else proc.returncode}"
    return None


def detect() -> Tuple[Optional[Gpu], str]:
    """First working VAAPI device, and a one-line reason when there is none."""
    if " h264_vaapi " not in _encoders():
        return None, f"this ffmpeg ({ffmpeg_exe()}) has no h264_vaapi encoder"
    nodes = render_nodes()
    if not nodes:
        return None, "no GPU render device (/dev/dri/renderD*) found"
    why = ""
    for dev in nodes:
        why = probe_device(dev) or ""
        if not why:
            return Gpu(dev), f"VAAPI on {dev}"
    return None, why


def resolve(mode: str) -> Optional[Gpu]:
    """--gpu: off = CPU; vaapi = GPU or a ClipsError; auto = GPU when it works, else CPU."""
    if mode == "off":
        return None
    gpu, why = detect()
    if gpu is None and mode == "vaapi":
        raise ClipsError(f"--gpu vaapi: {why}")
    if gpu is None:
        print(f"gpu: not used ({why}), encoding on the CPU", file=sys.stderr, flush=True)
    return gpu


_analysis: Optional[Gpu] = None      # GPU for the analysis decodes of the current render


def use_for_analysis(gpu: Optional[Gpu]) -> None:
    global _analysis
    _analysis = gpu


def analysis_input_args() -> List[str]:
    """Before an -i of an analysis decode (360 motion, tracking, subject frames): decode on the
    GPU picked for this render; software decoding when there is none or the stream needs it."""
    return ["-hwaccel", "vaapi", "-hwaccel_device", _analysis.device] if _analysis else []


def global_args(gpu: Optional[Gpu]) -> List[str]:
    return ["-init_hw_device", f"vaapi=va:{gpu.device}", "-filter_hw_device", "va"] if gpu else []


def input_args(gpu: Optional[Gpu]) -> List[str]:
    """Before an -i: decode on the GPU; ffmpeg falls back to software decoding by itself for
    streams the GPU cannot decode. Frames come back to system memory for the CPU filters."""
    return ["-hwaccel", "vaapi", "-hwaccel_device", "va"] if gpu else []


def upload(graph: str, gpu: Optional[Gpu]) -> str:
    """Filter graph ending in `format=yuv420p[v]`: hand the frames to the GPU encoder instead."""
    if not gpu:
        return graph
    head, sep, tail = graph.rpartition("format=yuv420p[v]")
    if not sep or tail:
        raise ValueError("graph must end with format=yuv420p[v]")
    return head + "format=nv12,hwupload[v]"


def video_codec(gpu: Optional[Gpu], quality: int, x264: Optional[List[str]] = None) -> List[str]:
    """Encoder args: libx264 with the given preset/crf args, or h264_vaapi at a constant QP
    of `quality` (the CRF the CPU path would use)."""
    if gpu:
        return ["-c:v", "h264_vaapi", "-rc_mode", "CQP", "-qp", str(quality), "-profile:v", "high"]
    return ["-c:v", "libx264", *(x264 or ["-crf", str(quality)]), "-pix_fmt", "yuv420p"]
