import numpy as np
import pytest

from clips import audio as A


def check_invariants(plan):
    fps = plan.fps
    total = round(plan.output_len * fps)
    assert plan.shots[0].start_frame == 0
    assert sum(s.frames for s in plan.shots) == total
    pos = 0
    for s in plan.shots:
        assert s.start_frame == pos
        pos += s.frames
        assert round(s.start * fps) == s.start_frame      # on frame grid
        assert np.ceil(A.MIN_SHOT * fps) <= s.frames <= np.floor(A.MAX_SHOT * fps)
    assert plan.shots[-1].end == pytest.approx(total / fps)


def test_beats_are_120bpm(an120):
    iv = np.diff(an120.beats)
    assert np.median(iv) == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("fps", [30, 60])
def test_auto_invariants(an120, an_ql, fps):
    check_invariants(A.plan_from_analysis(an120, fps))
    check_invariants(A.plan_from_analysis(an_ql, fps))


def test_loud_half_has_shorter_shots(an_ql):
    plan = A.plan_from_analysis(an_ql, 30)
    assert plan.mode == "auto"
    q = [s.length for s in plan.shots if s.end <= 11]
    l = [s.length for s in plan.shots if s.start >= 13]
    assert len(q) >= 2 and len(l) >= 4
    assert np.mean(l) < np.mean(q)


def test_beats_per_cut_2(an120):
    plan = A.plan_from_analysis(an120, 30, beats_per_cut=2)
    check_invariants(plan)
    mid = [s.length for s in plan.shots[1:-1]]
    assert np.mean(mid) == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("fixture", ["silence", "noise"])
def test_no_rhythm_fallback(request, fixture):
    path = request.getfixturevalue(fixture)
    an = A.analyze(path, 0.0, 8.0)
    plan = A.plan_from_analysis(an, 30)
    check_invariants(plan)
    plan = A.plan_from_analysis(an, 30, beats_per_cut=4)
    check_invariants(plan)


def test_fallback_grid_pure():
    cuts, mode = A.fallback_cuts(np.array([]), 5.0)
    assert mode == "grid" and cuts == [0, 1, 2, 3, 4]


def test_finalize_merge_split():
    # 0.1s shot merged into previous; 3 s shot split evenly
    r = A.finalize_frames([0, 1.0, 1.1, 4.1], 5.0, 30, max_shot=2.0)
    assert sum(n for _, n in r) == 150
    assert all(8 <= n <= 60 for _, n in r)
    assert r[0] == (0, 33) or r[0][1] <= 60
    # leading short shot merges forward
    r = A.finalize_frames([0, 0.1, 1.0], 2.0, 30)
    assert r[0][1] >= 8 and sum(n for _, n in r) == 60


def test_finalize_tiny_output_exempt():
    assert A.finalize_frames([0], 0.1, 30) == [(0, 3)]


def test_output_length():
    assert A.output_length(10, 2, None) == 8
    assert A.output_length(10, 2, 5) == 5
    assert A.output_length(10, 2, 50) == 8
    with pytest.raises(Exception):
        A.output_length(10, 12, None)


def test_tempo_guard():
    beats = np.arange(0, 10, 0.3)   # 200 BPM
    cuts = A.auto_cuts(beats, 200, np.array([]), lambda t: 1.0, 10)
    assert min(np.diff(cuts)) >= 0.6 - 0.04


def test_onset_cuts_at_high_energy():
    beats = np.arange(0, 10, 0.5)
    on = np.array([2.25, 4.1])
    hi = A.auto_cuts(beats, 120, on, lambda t: 1.0, 10)
    lo = A.auto_cuts(beats, 120, on, lambda t: 0.3, 10)
    assert any(abs(c - 2.25) < 1e-6 for c in hi)
    assert not any(abs(c - 2.25) < 1e-6 for c in lo)


def test_m4a_audio_decodes(click120, tmp_path):
    from clips.ffmpeg import run_ffmpeg

    m4a = tmp_path / "song.m4a"
    run_ffmpeg(["-y", "-i", str(click120), "-c:a", "aac", str(m4a)])
    length = A.audio_length(m4a)
    an = A.analyze(m4a, 0.0, length)
    assert abs(an.tempo - 120) < 6
    check_invariants(A.plan_from_analysis(an, 30))
