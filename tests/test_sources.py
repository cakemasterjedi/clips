import pytest

from clips import sources as S


def test_directory_cycle_and_cursor():
    a = S.assign_directory([1, 1, 1, 1, 1, 1], [10, 10])
    assert [x.source for x in a] == [0, 1, 0, 1, 0, 1]
    assert [x.start for x in a] == [0, 0, 1, 1, 2, 2]


def test_directory_wrap():
    a = S.assign_directory([1.0] * 5, [2.0])   # usable 1.96 -> one window then wrap
    assert [x.start for x in a] == [0, 0, 0, 0, 0]


def test_directory_wrap_longer():
    a = S.assign_directory([1.5] * 4, [4.0])
    assert [x.start for x in a] == [0, 1.5, 0, 1.5]


def test_directory_skips_short_clip():
    a = S.assign_directory([1.5, 1.5, 1.5], [1.0, 5.0, 5.0])
    assert [x.source for x in a] == [1, 2, 1]
    assert not any(x.loop for x in a)


def test_directory_loops_longest():
    a = S.assign_directory([3.0, 3.0], [1.0, 2.0])
    assert all(x.loop and x.source == 1 and x.start == 0 for x in a)


def test_directory_shuffle_seeded():
    d = [5.0] * 6
    a = S.assign_directory([1.0] * 6, d, shuffle=True, seed=3)
    b = S.assign_directory([1.0] * 6, d, shuffle=True, seed=3)
    assert a == b
    assert sorted(x.source for x in a) == list(range(6))


def test_single_stride():
    a = S.assign_single([1.0] * 4, 20.0)
    starts = [x.start for x in a]
    assert starts == sorted(starts) and starts[0] == 0
    assert starts[-1] + 1.0 <= 20.0
    assert starts[-1] > 10   # covers the source


def test_single_wrap():
    a = S.assign_single([1.0] * 5, 2.5)
    assert [x.start for x in a] == [0, 1.0, 0, 1.0, 0]


def test_single_loop_when_too_short():
    a = S.assign_single([2.0], 1.0)
    assert a[0].loop


def test_single_shuffle_non_overlapping():
    lens = [1.0, 0.5, 2.0, 0.7, 1.3]
    a = S.assign_single(lens, 30.0, shuffle=True, seed=1)
    assert a == S.assign_single(lens, 30.0, shuffle=True, seed=1)
    wins = sorted((x.start, x.start + ln) for x, ln in zip(a, lens))
    for (s1, e1), (s2, e2) in zip(wins, wins[1:]):
        assert e1 <= s2 + 1e-5
    assert wins[-1][1] <= 30.0


def test_start_at_skips_intro_footage():
    from clips.sources import assign_directory, assign_single

    res = assign_directory([1.0, 1.0, 1.0], [20.0, 20.0], start_at=[8.0, 0.0])
    assert res[0].source == 1 and res[0].start == 0.0      # music opens on another clip
    assert res[1].source == 0 and res[1].start == 8.0      # intro clip resumes after the intro
    assert res[2].source == 1 and res[2].start == 1.0
    res = assign_single([1.0] * 5, 30.0, start_at=10.0)
    assert all(a.start >= 10.0 for a in res)
    res = assign_single([1.0] * 5, 12.0, start_at=10.0)        # not enough left: ignore offset
    assert res[0].start == 0.0
