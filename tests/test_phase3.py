import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from clips import audio as A
from clips import cli, intro as I, speech as SP
from clips import render as R
from clips.ffmpeg import ClipsError, probe, ffmpeg_exe
from conftest import SR, clicks, make_video, music_like, speech_like
import subprocess


# ------------------------------------------------------------------ A. pacing / holds
def test_quiet_half_breathes_loud_half_stays_fast(an_ql):
    plan = A.plan_from_analysis(an_ql, 30)
    quiet = [s.length for s in plan.shots if s.end <= 12.5]
    loud = [s.length for s in plan.shots if s.start >= 13]
    assert max(quiet) >= 2.5
    assert np.mean(loud) < 0.8


def test_max_shot_is_4s():
    r = A.finalize_frames([0.0], 9.0, 30)
    assert [n for _, n in r] == [90, 90] or all(n <= 120 for _, n in r)
    assert max(n for _, n in r) <= 120


def _hit_fn(big):
    return lambda t: 1.0 if any(abs(t - b) < 1e-6 for b in big) else 0.05


def test_post_hit_hold():
    beats = np.arange(0.5, 40, 0.5)
    base = A.auto_cuts(beats, 120, np.array([]), lambda t: 1.0, 40)
    cuts = A.auto_cuts(beats, 120, np.array([]), lambda t: 1.0, 40, hit_fn=_hit_fn([10.0]))
    assert 10.0 in cuts
    nxt = [c for c in cuts if c > 10.0][0]
    assert nxt - 10.0 >= 2.0 - 1e-6            # min(4 * 0.5 s, 2.5 s)
    assert np.diff(base).max() < 1.0           # without hits: one cut per beat


def test_hold_rate_limit():
    beats = np.arange(0.5, 40, 0.5)
    big = [10.0, 13.0, 15.0, 19.0]            # 12% of 79 candidates -> 9 hot; only some given hits
    fn = lambda t: 1.0 if any(abs(t - b) < 1e-6 for b in big) else 0.0
    cuts = A.auto_cuts(beats, 120, np.array([]), lambda t: 1.0, 40, hit_fn=fn)
    gaps = {c: n - c for c, n in zip(cuts, cuts[1:])}
    assert gaps[10.0] >= 2.0 - 1e-6            # first hit holds
    # 13/15 fall inside the 8 s window after the hold at 10 (cut at 13 happens at 12.0.. then 12.5)
    held = [c for c, g in gaps.items() if g >= 2.0 - 1e-6 and c >= 10.0 and c < 18.0]
    assert held == [10.0]
    assert gaps.get(19.0, 0) >= 2.0 - 1e-6     # 19 >= 10 + 8 -> allowed again


def test_hold_suppresses_onset_cuts():
    beats = np.arange(0.5, 20, 0.5)
    on = np.array([10.8, 11.3])
    cuts = A.auto_cuts(beats, 120, on, lambda t: 1.0, 20, hit_fn=_hit_fn([10.0]))
    assert not any(10.5 < c < 12.0 for c in cuts)


def test_invariants_with_new_max(an120, an_ql):
    for an in (an120, an_ql):
        for fps in (30, 60):
            plan = A.plan_from_analysis(an, fps)
            assert sum(s.frames for s in plan.shots) == round(plan.output_len * fps)
            assert all(np.ceil(A.MIN_SHOT * fps) <= s.frames <= np.floor(A.MAX_SHOT * fps)
                       for s in plan.shots)


# ------------------------------------------------------------------ B. intro spec parsing
def test_parse_time():
    assert I.parse_time("5") == 5.0
    assert I.parse_time("1:02.5") == 62.5
    assert I.parse_time("0:07") == 7.0
    assert I.parse_time("1:00:01") == 3601.0


def test_parse_spec():
    assert I.parse_intro_spec("a.mp4@1.5-4") == ("a.mp4", 1.5, 4.0)
    assert I.parse_intro_spec("clip.mp4@0:10-0:15.5") == ("clip.mp4", 10.0, 15.5)
    assert I.parse_intro_spec("/tmp/we@ird/x@y.mp4@2-3") == ("/tmp/we@ird/x@y.mp4", 2.0, 3.0)


@pytest.mark.parametrize("bad", ["a.mp4", "a.mp4@5", "a.mp4@5-5", "a.mp4@6-5", "@1-2",
                                 "a.mp4@x-3", "a.mp4@1-y"])
def test_parse_spec_errors(bad):
    with pytest.raises(ClipsError):
        I.parse_intro_spec(bad)


@pytest.fixture(scope="module")
def speech_clip(tmp_path_factory):
    d = tmp_path_factory.mktemp("p3")
    wav = d / "sp.wav"
    sf.write(wav, speech_like(5, 22050, seed=2), 22050)
    clip = d / "talk.mp4"
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc=size=640x360:rate=25:duration=5", "-i", str(wav), "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(clip)], check=True)
    mute = d / "mute.mp4"
    make_video(mute, "320x240", 25, 3)
    return d, clip, mute


def test_resolve_specs(speech_clip):
    d, clip, mute = speech_clip
    r = I.resolve_specs([f"{clip}@0.5-3"], None)
    assert r[0].source == clip.resolve() and r[0].length == 2.5
    r = I.resolve_specs(["talk.mp4@0:01-0:02"], d)         # relative to the input folder
    assert r[0].start == 1.0
    with pytest.raises(ClipsError, match="beyond"):
        I.resolve_specs([f"{clip}@1-99"], None)
    with pytest.raises(ClipsError, match="no audio"):
        I.resolve_specs([f"{mute}@0-1"], None)
    with pytest.raises(ClipsError, match="not found"):
        I.resolve_specs(["nope.mp4@0-1"], d)


# ------------------------------------------------------------------ speech detection
def test_speech_detected_music_rejected():
    assert SP.detect_regions(speech_like(10)) != []
    assert SP.detect_regions(music_like(10)) == []


def test_mixed_region_overlaps_speech_with_edges_at_pauses():
    y = np.concatenate([music_like(6), speech_like(4, seed=3)])
    regs = SP.detect_regions(y)
    assert regs
    r = max(regs, key=lambda r: r.score)
    assert r.start >= 5.5 and r.end <= 10.01       # never inside the music part
    assert r.length >= SP.MIN_REGION
    sc, db = SP.score_frames(y)
    for t in (r.start, r.end):
        i = min(int(t * SP.FPS), db.size - 1)
        assert db[max(0, i - 3):i + 4].min() < np.median(db[int(r.start * 100):int(r.end * 100)]) - 6


def test_choose_intro_trim_and_second():
    def reg(a, b, s, pauses=()):
        return SP.Region(Path("x.mp4"), a, b, s, list(pauses))
    a = reg(0, 12, 0.9, [3, 6, 9.5])
    out = SP.choose_intro([a], 8.0)
    assert out[0].end == 6 and out[0].start == 0         # trimmed at the last pause <= 8 s
    out = SP.choose_intro([reg(0, 3, 0.9), reg(10, 13, 0.8), reg(20, 30, 0.7, [24, 26])], 8.0)
    assert len(out) == 2 and out[0].start == 0 and out[1].start == 10
    assert sum(r.length for r in out) <= 8.0
    assert SP.choose_intro([], 8.0) == []
    # preference for the first-used clip
    other = SP.Region(Path("y.mp4"), 0, 3, 0.8)
    pref = SP.Region(Path("x.mp4"), 0, 3, 0.75)
    assert SP.choose_intro([other, pref], 3.0, preferred=Path("x.mp4"))[0].source == Path("x.mp4")


def test_auto_detect_dry_run(speech_clip, tmp_path, capsys):
    d, clip, mute = speech_clip
    song = tmp_path / "s.wav"
    times = np.arange(0.25, 12, 0.5)
    sf.write(song, clicks(times, np.full(times.size, 0.8), 12), SR)
    assert cli.main([str(clip), "--audio", str(song), "--dry-run", "--duration", "9"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intro"] and out["candidates"] and out["t_intro"] > 1.0
    assert out["shots"][0]["section"] == "intro"
    capsys.readouterr()
    assert cli.main([str(clip), "--audio", str(song), "--dry-run", "--no-intro"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["intro"] == [] and out["t_intro"] == 0
    assert all(s["section"] == "music" for s in out["shots"])


# ------------------------------------------------------------------ plan composition
def test_compose_plan(an120):
    music = A.plan_from_analysis(an120, 30, punch=True, flash=True)
    plan = A.compose_plan([150, 90], music)
    assert plan.t_intro == 8.0
    intro = [s for s in plan.shots if s.section == "intro"]
    assert [s.frames for s in intro] == [150, 90]          # unsplit (> 4 s allowed)
    assert not any(s.punch or s.flash or s.hit for s in intro)
    assert plan.shots[0].start_frame == 0 and intro[1].start_frame == 150
    m = [s for s in plan.shots if s.section == "music"]
    assert m[0].start_frame == 240
    assert [s.start_frame - 240 for s in m] == [s.start_frame for s in music.shots]
    assert sum(s.frames for s in plan.shots) == round(plan.output_len * 30)
    assert [s.index for s in plan.shots] == list(range(len(plan.shots)))


# ------------------------------------------------------------------ final-pass audio graph
def test_effective_rise_clamp():
    assert R.effective_rise(3.0, 50.0, 7.0, "rise") == (3.0, [])
    r, notes = R.effective_rise(3.0, 1.0, 7.0, "rise")
    assert r == 1.0 and "clamped" in notes[0]
    assert R.effective_rise(3.0, 50.0, 2.0, "rise")[0] == 2.0
    assert R.effective_rise(3.0, 50.0, 7.0, "drop") == (0.0, [])


def test_audio_graph_drop():
    g = R.intro_audio_graph([4.4], 30.0, 4.4, 0.0)
    assert "[2:a]atrim=duration=4.400000" in g and "afade=t=in:st=0:d=0.03" in g
    assert "afade=t=out:st=4.250000:d=0.150" in g
    assert "volume" not in g
    assert f"adelay={round(4.4 * 44100)}S|{round(4.4 * 44100)}S" in g
    assert "[1:a]atrim=duration=25.600000" in g
    assert "amix=inputs=2:duration=longest:normalize=0,alimiter" in g and g.endswith("[a]")
    assert "afade=t=out:st=29.700000:d=0.3[a]" in g


def test_audio_graph_rise_two_regions():
    g = R.intro_audio_graph([3.0, 4.0], 30.0, 7.0, 3.0)
    assert "[d0][d1]concat=n=2:v=0:a=1[dl]" in g and "[3:a]atrim" in g
    assert "volume='if(lt(t,3.000000),0.35*t/3.000000,1)':eval=frame" in g
    assert "[1:a]atrim=duration=26.000000" in g          # rise + music section
    assert f"adelay={4 * 44100}S|{4 * 44100}S" in g       # music starts at T_intro - rise


def test_final_args_intro_inputs(tmp_path):
    ia = R.IntroAudio([(Path("a.mp4"), 1.0, 2.0)], "rise", 3.0)
    a = R.final_args_intro(Path("l.txt"), Path("s.wav"), 50.0, 30.0, 30, None, Path("o.mp4"), ia, 7.0)
    i = a.index("s.wav")
    assert a[i - 3:i - 1] == ["-ss", "47.000000"]                 # song position audio_start - rise
    assert a[a.index("a.mp4") - 5:a.index("a.mp4") - 1] == ["-ss", "1.000000", "-t", "2.000000"]
    assert a[a.index("-t", a.index("a.mp4") + 1) + 1] == "30.000000"
    ia = R.IntroAudio([(Path("a.mp4"), 1.0, 2.0)], "rise", 3.0)
    a = R.final_args_intro(Path("l.txt"), Path("s.wav"), 1.0, 30.0, 30, None, Path("o.mp4"), ia, 7.0)
    assert a[a.index("s.wav") - 2] == "0.000000"             # clamped to audio_start


def test_no_intro_command_unchanged():
    a = R.final_args(Path("l.txt"), Path("s.wav"), 2.0, 6.0, 30, None, Path("o.mp4"))
    assert "[1:a]atrim=duration=6.000000,asetpts=PTS-STARTPTS,afade=t=out:st=5.700000:d=0.3[a]" in \
        a[a.index("-filter_complex") + 1]
    assert "amix" not in " ".join(a)


# ------------------------------------------------------------------ e2e
def test_e2e_intro(speech_clip, tmp_path):
    d, clip, mute = speech_clip
    song = tmp_path / "s.wav"
    times = np.arange(0.25, 12, 0.5)
    sf.write(song, clicks(times, np.full(times.size, 0.8), 12), SR)
    out = tmp_path / "o.mp4"
    assert cli.main([str(d), "--audio", str(song), "-o", str(out), "--duration", "6",
                     "--intro", "talk.mp4@0.5-3.5", "--music-in", "rise", "--rise", "1"]) == 0
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920) and i.has_audio
    assert abs(i.duration - 6.0) < 0.05
    y = A.decode(out, 0.0, None, 22050)
    rms = lambda a, b: float(np.sqrt(np.mean(y[int(a * 22050):int(b * 22050)] ** 2)))
    assert rms(0.3, 2.5) > 0.005                                  # dialogue audible in the intro
    assert rms(3.5, 5.5) > 0.005                                  # music after the intro
    assert y.size / 22050 == pytest.approx(6.0, abs=0.06)


def test_intro_not_shorter_than_duration(speech_clip, tmp_path):
    d, clip, _ = speech_clip
    song = tmp_path / "s.wav"
    sf.write(song, np.zeros(SR * 5), SR)
    assert cli.main([str(d), "--audio", str(song), "-o", str(tmp_path / "o.mp4"),
                     "--duration", "3", "--intro", "talk.mp4@0-3"]) == 1


def test_choose_intro_joins_close_same_clip_regions():
    from pathlib import Path
    from clips.speech import Region, choose_intro

    src = Path("/x/a.mp4")
    cands = [Region(src, 24.4, 29.4, 0.7), Region(src, 21.8, 24.1, 0.72)]
    out = choose_intro(cands, intro_max=8.0)
    assert len(out) == 1 and out[0].start == 21.8 and out[0].end == 29.4
    far = [Region(src, 30.0, 33.0, 0.7), Region(src, 21.8, 24.1, 0.72)]
    out = choose_intro(far, intro_max=8.0)
    assert [r.start for r in out] == [21.8, 30.0]          # kept apart, source order
