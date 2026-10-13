import json

import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips.ffmpeg import ClipsError, probe
from conftest import SR, clicks, make_video


@pytest.fixture
def setup(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for n, p in [("a", "testsrc"), ("b", "smptebars"), ("c", "testsrc2")]:
        make_video(d / f"{n}.mp4", "640x360", 30, 8, p)
    t = np.arange(0.25, 40, 0.5)
    amps = np.where((t >= 10) & (t < 20), 0.9, np.where((t >= 28) & (t < 34), 0.7, 0.1))
    song = tmp_path / "s.wav"
    sf.write(song, clicks(t, amps, 40), SR)
    return d, song


def test_variations_dry_run_differ(setup, capsys):
    d, song = setup
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--duration", "6",
                     "--audio-start", "auto", "--variations", "3", "--dry-run"]) == 0
    docs = json.loads(capsys.readouterr().out)["variations"]
    assert len(docs) == 3
    starts = [doc["audio_start"] for doc in docs]
    assert starts[0] == pytest.approx(10.0, abs=0.3)                 # v1 = the best section
    assert len(set(starts)) == 3 and min(abs(a - b) for a in starts for b in starts if a != b) >= 3
    orders = [[s["source"] for s in doc["shots"]] for doc in docs]
    assert orders[0] != orders[1] or orders[1] != orders[2]


def test_variations_render_files(setup, tmp_path):
    d, song = setup
    out = tmp_path / "o.mp4"
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--duration", "3",
                     "--variations", "2", "--preview", "-o", str(out)]) == 0
    for k in (1, 2):
        i = probe(tmp_path / f"o_v{k}.mp4")
        assert (i.width, i.height) == (1080, 1920) and i.duration == pytest.approx(3.0, abs=0.06)
    assert not out.exists()


def test_variations_range(setup, capsys):
    d, song = setup
    assert cli.main([str(d), "--audio", str(song), "--variations", "11", "--dry-run"]) == 1
