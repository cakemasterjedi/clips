import json

import numpy as np
import pytest
import soundfile as sf

from clips import audio as A
from clips import cli
from clips import render as R
from clips.ffmpeg import ffmpeg_exe, probe
from conftest import SR, clicks, make_video


def two_chord_song(secs=16, bar=2.0):
    """Clicks every 0.5 s over a tone that alternates A (220 Hz) / B (330 Hz) every `bar` s."""
    t = np.arange(int(secs * SR)) / SR
    f = np.where((t // bar) % 2 == 0, 220.0, 330.0)
    y = 0.3 * np.sin(2 * np.pi * np.cumsum(f) / SR)
    beats = np.arange(0.0, secs, 0.5)
    return y + clicks(beats + 0.001, np.full(beats.size, 0.6), secs), beats


def test_loop_end_lands_where_the_song_sounds_like_the_start():
    y, beats = two_chord_song()
    end, sim = A.pick_loop_end(y, SR, beats, 7.0)          # 6-8 s is a B bar
    assert end <= 7.0
    # beats from 4.0 to 5.5 are followed by A (like the start); the latest one wins
    assert end == pytest.approx(5.5, abs=0.01)
    assert sim > 0.9


def test_loop_end_without_beats_keeps_length():
    assert A.pick_loop_end(np.zeros(SR * 4), SR, np.zeros(0), 3.3) == (3.3, 0.0)


def test_loop_fade_and_blend_args():
    a = R.final_args(R.Path("l.txt"), R.Path("s.wav"), 0.0, 6.0, 30, None, R.Path("o.mp4"),
                     fade=R.LOOP_FADE)
    fc = a[a.index("-filter_complex") + 1]
    assert "afade=t=in:st=0:d=0.02" in fc and "afade=t=out:st=5.980000:d=0.02" in fc
    f = R.segment_filter("blur", 30, blend_at=0.75)
    assert "[1:v]" in f and "fade=t=in:st=0.750000:d=0.250000:alpha=1" in f
    assert "[l0a]" in f                                      # blur labels don't collide


def test_loop_e2e(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    make_video(d / "a.mp4", "640x360", 30, 8, "testsrc")
    make_video(d / "b.mp4", "640x360", 30, 8, "smptebars")
    y, _ = two_chord_song(12)
    song = tmp_path / "s.wav"
    sf.write(song, y, SR)
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--loop", "--duration", "7",
                     "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    end = plan["shots"][-1]["end"]
    assert end <= 7.0 and plan["loop_match"] is not None
    out = tmp_path / "o.mp4"
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--loop", "--duration", "7",
                     "-o", str(out)]) == 0
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.duration == pytest.approx(end, abs=0.06)
    import subprocess

    def frame(t):
        raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-ss", f"{t}", "-i", str(out),
                              "-frames:v", "1", "-vf", "scale=54:96,format=gray", "-f", "rawvideo",
                              "-"], stdout=subprocess.PIPE, check=True).stdout
        return np.frombuffer(raw, np.uint8).astype(float)

    first, last, before = frame(0), frame(info.duration - 0.04), frame(info.duration - 0.5)
    assert np.abs(last - first).mean() < 0.5 * np.abs(before - first).mean()
