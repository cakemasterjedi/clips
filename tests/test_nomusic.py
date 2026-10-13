import json
import subprocess

import pytest

from clips import cli, render as R
from clips.ffmpeg import ffmpeg_exe, probe
from conftest import make_video


@pytest.fixture
def footage(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=1280x720:rate=30:duration=4", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=4", "-shortest", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "a.mp4")], check=True)
    make_video(d / "b.mp4", "720x1280", 30, 3, "smptebars")          # no sound
    return d


def volume(path):
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path), "-af", "volumedetect",
                          "-f", "null", "-"], capture_output=True, text=True).stderr
    return float(err.split("max_volume:")[1].split("dB")[0])


def test_no_music_plan(footage, capsys):
    assert cli.main([str(footage), "--no-music", "--duration", "6", "--bpm", "100", "--dry-run"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["music"] is None and doc["sound"] == "clips"
    assert sum(s["frames"] for s in doc["shots"]) == 180 and len(doc["shots"]) >= 2
    assert cli.main([str(footage), "--duration", "6", "--dry-run"]) == 1     # no song, no flag
    assert "--no-music" in capsys.readouterr().err
    assert cli.main([str(footage), "--no-music", "--loop", "--dry-run"]) == 1


def test_no_music_render_keeps_clip_sound(footage, tmp_path):
    out = tmp_path / "o.mp4"
    assert cli.main([str(footage), "--no-music", "--duration", "4", "--preview", "-o", str(out),
                     "--zoom", "0.6"]) == 0
    info = probe(out)
    assert (info.width, info.height) == (1080, 1920) and info.has_audio
    assert abs(info.duration - 4.0) < 0.1
    assert volume(out) > -30                                    # the sine from clip a is there
    muted = tmp_path / "m.mp4"
    assert cli.main([str(footage), "--mute", "--duration", "3", "--preview", "-o", str(muted)]) == 0
    assert volume(muted) < -80


def test_zoom_graph():
    g = R.fit_graph("crop", 30, zoom=0.6)
    assert "gblur" in g and "overlay" in g and "*0.6" in g
    assert "gblur" not in R.fit_graph("crop", 30, zoom=1.5)
    assert R.fit_graph("crop", 30) == R.fit_graph("crop", 30, zoom=1.0)
    assert cli.main(["x", "--audio", "y", "--zoom", "9", "--dry-run"]) == 1
