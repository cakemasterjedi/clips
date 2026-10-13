import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import render as R
from clips.ffmpeg import probe
from conftest import SR, clicks, make_video


def test_preview_args():
    from clips.sources import Assignment

    a = R.segment_args(R.Path("a.mp4"), Assignment(0, 0.0), 30, 30, R.Path("o.mp4"),
                       punch=True, preview=True)
    fc = a[a.index("-filter_complex") + 1]
    assert "crop=270:480" in fc and "270*" in fc and "ultrafast" in a
    f = R.final_args(R.Path("l.txt"), R.Path("s.wav"), 0.0, 6.0, 30, None, R.Path("o.mp4"),
                     preview=True)
    assert "scale=1080:1920" in f[f.index("-filter_complex") + 1] and "ultrafast" in f
    full = R.segment_args(R.Path("a.mp4"), Assignment(0, 0.0), 30, 30, R.Path("o.mp4"))
    assert "crop=1080:1920" in full[full.index("-filter_complex") + 1] and "veryfast" in full


def test_preview_e2e_keeps_output_spec(tmp_path):
    v = tmp_path / "v.mp4"
    make_video(v, "1280x720", 30, 6)
    t = np.arange(0.25, 4, 0.5)
    song = tmp_path / "s.wav"
    sf.write(song, clicks(t, np.full(t.size, 0.8), 4), SR)
    srt = tmp_path / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nHello\n", encoding="utf-8")
    out = tmp_path / "o.mp4"
    assert cli.main([str(v), "--audio", str(song), "--no-intro", "--preview", "--punch",
                     "--subs", str(srt), "-o", str(out)]) == 0
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920) and i.fps == pytest.approx(30, abs=0.01)
    assert i.duration == pytest.approx(4.0, abs=0.06)
