import subprocess

import numpy as np
import pytest
import soundfile as sf

from clips import audio as A
from clips.ffmpeg import ffmpeg_exe

SR = 22050


def clicks(times, amps, length, sr=SR):
    y = np.zeros(int(length * sr))
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    burst = np.sin(2 * np.pi * 1500 * t) * np.exp(-t * 150)
    for tt, a in zip(times, amps):
        i = int(tt * sr)
        seg = burst[: max(0, min(n, y.size - i))]
        y[i:i + seg.size] += a * seg
    return y


def make_video(path, size, fps, secs, pattern="testsrc"):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"{pattern}=size={size}:rate={fps}:duration={secs}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


@pytest.fixture(scope="session")
def click120(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "click120.wav"
    times = np.arange(0.25, 20, 0.5)
    sf.write(p, clicks(times, np.full(times.size, 0.8), 20), SR)
    return p


@pytest.fixture(scope="session")
def quiet_loud(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "ql.wav"
    q = np.arange(0.25, 12, 0.5)
    l = np.arange(12.0, 24, 0.25)
    sf.write(p, clicks(np.concatenate([q, l]),
                       np.concatenate([np.full(q.size, 0.06), np.full(l.size, 0.9)]), 24), SR)
    return p


@pytest.fixture(scope="session")
def an120(click120):
    return A.analyze(click120, 0.0, 20.0)


@pytest.fixture(scope="session")
def an_ql(quiet_loud):
    return A.analyze(quiet_loud, 0.0, 24.0)


@pytest.fixture(scope="session")
def silence(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "sil.wav"
    sf.write(p, np.zeros(SR * 8), SR)
    return p


@pytest.fixture(scope="session")
def noise(tmp_path_factory):
    p = tmp_path_factory.mktemp("aud") / "noise.wav"
    sf.write(p, np.random.default_rng(0).normal(0, 0.1, SR * 8), SR)
    return p


def speech_like(dur, sr=16000, seed=0):
    """Band-limited noise + voiced harmonics, amplitude-modulated at syllable rate with
    irregular syllable gaps, word gaps and sentence pauses."""
    rng = np.random.default_rng(seed)
    n = int(dur * sr)
    env = np.zeros(n)
    t = 0.2
    while t < dur - 0.3:
        for _ in range(rng.integers(3, 6)):
            for _ in range(rng.integers(2, 5)):
                L = rng.uniform(0.10, 0.25)
                i, m = int(t * sr), int(L * sr)
                if i + m >= n:
                    break
                env[i:i + m] += np.hanning(m) * rng.uniform(0.5, 1.0)
                t += L + rng.uniform(0.03, 0.10)
            t += rng.uniform(0.1, 0.28)
        t += rng.uniform(0.5, 0.9)
    x = rng.normal(size=n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / sr)
    X[(f < 300) | (f > 3000)] = 0
    x = np.fft.irfft(X, n)
    x /= np.abs(x).max()
    ph = np.cumsum(2 * np.pi * (120 + 20 * np.sin(2 * np.pi * 0.7 * np.arange(n) / sr)) / sr)
    v = sum(np.sin(k * ph) / k for k in range(1, 12))
    y = env * (0.6 * x + 0.3 * v / np.abs(v).max())
    y = 0.5 * y / np.abs(y).max() + 0.0005 * rng.normal(size=n)
    return y.astype(np.float32)


def music_like(dur, sr=16000):
    """120 BPM click track plus sustained tones."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    nn = int(0.03 * sr)
    tt = np.arange(nn) / sr
    burst = np.sin(2 * np.pi * 1500 * tt) * np.exp(-tt * 150)
    for c in np.arange(0.25, dur, 0.5):
        i = int(c * sr)
        y[i:i + nn] += 0.8 * burst[:max(0, min(nn, n - i))]
    y += 0.2 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 660 * t)
    return y.astype(np.float32)
