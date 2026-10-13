import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import render as R
from conftest import SR, clicks, make_video


def test_loudnorm_args():
    m = {"input_i": "-20.1", "input_tp": "-3.0", "input_lra": "5.0", "input_thresh": "-30.4",
         "target_offset": "0.2"}
    a = R.loudnorm_args(R.Path("a.mp4"), R.Path("b.mp4"), m, 6.0)
    af = a[a.index("-af") + 1]
    assert af.startswith("volume=6.10dB,alimiter=limit=0.8414") and a[a.index("-t") + 1] == "6.000000"
    assert a[a.index("-c:v") + 1] == "copy"


def test_render_is_normalized(tmp_path):
    v = tmp_path / "v.mp4"
    make_video(v, "640x360", 30, 6)
    t = np.arange(0.25, 6, 0.25)
    song = tmp_path / "s.wav"
    sf.write(song, 0.05 * clicks(t, np.full(t.size, 1.0), 6) + 0.02 * np.sin(np.arange(6 * SR) / SR * 2 * np.pi * 220), SR)
    out = tmp_path / "o.mp4"
    assert cli.main([str(v), "--audio", str(song), "--no-intro", "-o", str(out)]) == 0
    m = R.measure_loudness(out)
    assert float(m["input_i"]) == pytest.approx(-14.0, abs=1.5)
    raw = tmp_path / "raw.mp4"
    assert cli.main([str(v), "--audio", str(song), "--no-intro", "--no-loudnorm", "-o", str(raw)]) == 0
    assert float(R.measure_loudness(raw)["input_i"]) < -20


def test_silence_is_left_alone(tmp_path, silence):
    v = tmp_path / "v.mp4"
    make_video(v, "640x360", 30, 9)
    out = tmp_path / "o.mp4"
    assert cli.main([str(v), "--audio", str(silence), "--no-intro", "-o", str(out)]) == 0
    assert R.measure_loudness(out) is None
