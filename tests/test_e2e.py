import json

import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips.ffmpeg import probe
from conftest import SR, clicks, make_video


def test_end_to_end(tmp_path, capsys):
    d = tmp_path / "in clips"
    d.mkdir()
    make_video(d / "a.mp4", "1280x720", 25, 4)
    make_video(d / "b.mp4", "720x1280", 60, 4, "smptebars")
    (d / "notes.txt").write_text("x")
    times = np.arange(0.25, 6, 0.5)
    song = tmp_path / "song.wav"
    sf.write(song, clicks(times, np.full(times.size, 0.8), 6), SR)
    sd = tmp_path / "it's a dir"
    sd.mkdir()
    srt = sd / "s [1], x.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,000\nHello <i>there</i>\n\n"
                   "2\n00:00:03,000 --> 00:00:05,000\nSecond line\n", encoding="utf-8")
    out = tmp_path / "out.mp4"
    rc = cli.main([str(d), "--audio", str(song), "--subs", str(srt), "-o", str(out)])
    assert rc == 0 and out.exists()
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.fps == pytest.approx(30, abs=0.01)
    assert info.has_audio and info.has_video
    assert abs(info.duration - 6.0) < 1 / 30 + 0.06

    capsys.readouterr()
    assert cli.main([str(d), "--audio", str(song), "-o", str(out), "--fps", "60", "--dry-run"]) == 0
    shots = json.loads(capsys.readouterr().out)["shots"]
    assert sum(s["frames"] for s in shots) == 360
    assert shots[0]["start"] == 0 and set(shots[0]) >= {"index", "start", "end", "frames",
                                                          "source", "source_start", "energy"}


def test_bad_subs_extension(tmp_path, capsys):
    assert cli.main([str(tmp_path), "--audio", "x.wav", "--subs", "a.vtt", "-o", "o.mp4"]) == 1
