import json

import numpy as np
import pytest
import soundfile as sf

from clips import audio as A
from clips import cli
from clips import render as R
from clips.ffmpeg import probe
from conftest import SR, clicks, make_video


def test_select_ramps_prefers_hits_after_a_lift():
    hits = [0.0, 0.5, 0.9, 0.5, 0.5, 0.9, 0.5, 0.5, 0.5, 0.5]
    en = [0.2, 0.2, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9, 0.9]       # big lift into shot 5
    lens = [1.0] * 10
    r = A.select_ramps(hits, en, lens, frac=0.1)                  # one ramp
    assert r == [i == 4 for i in range(10)]                       # shot 4 slows into shot 5
    lens[4] = 0.3
    assert not any(A.select_ramps(hits, en, lens, frac=0.1))      # lead-in too short


def test_ramp_filter():
    f = R.segment_filter("crop", 30, size=(1080, 1920), ramp_len=1.5)
    assert "trim=end=1.000000" in f and "trim=start=1.000000:end=1.200000" in f
    assert "setpts=(PTS-STARTPTS)/0.4" in f and "minterpolate=fps=30:mi_mode=blend" in f
    p = R.segment_filter("crop", 30, size=(270, 480), ramp_len=1.5, preview=True)
    assert "minterpolate" not in p                                # skipped in preview


def test_ramp_e2e(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    make_video(d / "a.mp4", "640x360", 30, 6, "testsrc")
    make_video(d / "b.mp4", "640x360", 30, 6, "smptebars")
    t = np.arange(0.25, 8, 0.5)
    amps = np.where(t < 4, 0.1, 0.9)
    song = tmp_path / "s.wav"
    sf.write(song, clicks(t, amps, 8), SR)
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--ramp", "--dry-run"]) == 0
    shots = json.loads(capsys.readouterr().out)["shots"]
    assert any(s["ramp"] for s in shots)
    out = tmp_path / "o.mp4"
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--ramp", "-o", str(out)]) == 0
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920) and i.duration == pytest.approx(8.0, abs=0.06)
