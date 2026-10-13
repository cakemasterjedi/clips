import json

import numpy as np
import pytest
import soundfile as sf

from clips import audio as A
from clips import cli
from conftest import SR, clicks, make_video


@pytest.fixture(scope="session")
def build_drop(tmp_path_factory):
    """120 BPM: quiet 0-20 s, loud 20-40 s (the drop), quiet 40-60 s."""
    p = tmp_path_factory.mktemp("aud") / "drop.wav"
    t = np.arange(0.25, 60, 0.5)
    amps = np.where((t >= 20) & (t < 40), 0.9, 0.08)
    sf.write(p, clicks(t, amps, 60), SR)
    return p


def test_picks_the_drop(build_drop):
    start, top = A.auto_audio_start(build_drop, 15.0)
    # any start in [20, 25] keeps the window fully loud; the drop term wins at 20
    assert start == pytest.approx(20.0, abs=0.3)
    assert top[0][0] == pytest.approx(start, abs=1e-3)
    assert 3 <= len(top) <= 5 and top[0][1] >= top[1][1]
    starts = [t for t, _ in top]                     # distinct sections, >= 7.5 s apart
    assert min(abs(a - b) for a in starts for b in starts if a != b) >= 7.5 - 1e-6


def test_window_must_fit(build_drop):
    start, _ = A.auto_audio_start(build_drop, 50.0)
    assert 0.0 <= start <= 10.0 + 1e-6


def test_song_shorter_than_window(click120):
    assert A.auto_audio_start(click120, 30.0)[0] == 0.0


def test_no_drop_bonus_at_song_start():
    # flat energy: every start scores the same, so the earliest wins (no fake drop at t=0)
    times = np.arange(0, 30, 0.05)
    rms = np.full(times.size, 0.5)
    start, top = A.pick_section(10.0, 30.0, np.arange(0.5, 30, 0.5), rms, times, rms, times)
    assert start == 0.0
    assert top[0][1] == pytest.approx(top[1][1])


@pytest.mark.parametrize("v,ok", [("auto", "auto"), ("AUTO", "auto"), ("12.5", 12.5)])
def test_arg_parse(v, ok):
    assert cli.build_parser().parse_args(["x", "--audio", "a", "--audio-start", v]).audio_start == ok


def test_arg_parse_rejects_junk():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["x", "--audio", "a", "--audio-start", "chorus"])


def test_dry_run_auto(build_drop, tmp_path, capsys):
    v = tmp_path / "v.mp4"
    make_video(v, "1080x1920", 30, 5)
    assert cli.main([str(v), "--audio", str(build_drop), "--dry-run", "--no-intro",
                     "--duration", "10", "--audio-start", "auto"]) == 0
    out = capsys.readouterr()
    plan = json.loads(out.out)
    assert plan["audio_start"] == pytest.approx(20.0, abs=0.3)
    assert plan["audio_start_candidates"][0]["start"] == pytest.approx(plan["audio_start"], abs=1e-3)
    assert sum(s["frames"] for s in plan["shots"]) == 300
    assert "auto picked" in out.err


def test_dry_run_fixed_start_reports_it(build_drop, tmp_path, capsys):
    v = tmp_path / "v.mp4"
    make_video(v, "1080x1920", 30, 5)
    assert cli.main([str(v), "--audio", str(build_drop), "--dry-run", "--no-intro",
                     "--duration", "5", "--audio-start", "3"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["audio_start"] == 3.0 and plan["audio_start_candidates"] == []
