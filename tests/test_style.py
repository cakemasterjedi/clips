import json
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import style as S
from clips.ffmpeg import ClipsError, ffmpeg_exe
from conftest import SR, clicks, make_video, speech_like

RNG = np.random.default_rng(0)


def texture(seed):
    """Smooth random image (FH x FW) with structure at several scales."""
    r = np.random.default_rng(seed)
    img = np.zeros((S.FH, S.FW))
    for k in (4, 8, 16):
        small = r.uniform(0, 255, (S.FH // k, S.FW // k))
        img += np.kron(small, np.ones((k, k)))
    return (img / 3).astype(np.float32)


# ------------------------------------------------------------------ frame detectors
def test_detect_cuts_and_flash():
    a, b, c = texture(1), texture(2), texture(3)
    frames = [a + RNG.normal(0, 1, a.shape) for _ in range(20)]
    frames += [np.full_like(b, 255.0)] + [b * (i / 3) + 255 * (1 - i / 3) for i in (1, 2)]
    frames += [b + RNG.normal(0, 1, b.shape) for _ in range(17)]
    frames += [c for _ in range(15)]
    f = np.stack(frames).astype(np.float32)
    cuts = S.detect_cuts(f)
    assert cuts == [20, 40]                     # the white fade is not a second cut
    assert S.is_flash(f, 20) and not S.is_flash(f, 40)


def test_punch_scale():
    base = texture(4)
    shot = [S.zoom(base, 1 + 0.12 * (1 - min(i / 5.4, 1)) ** 2) for i in range(15)]
    f = np.stack([texture(5)] * 5 + shot).astype(np.float32)
    assert S.punch_scale(f, 5, len(f)) == pytest.approx(1.12, abs=0.03)
    still = np.stack([texture(5)] * 5 + [base] * 15).astype(np.float32)
    assert S.punch_scale(still, 5, len(still)) == 1.0


def test_detect_fit():
    t = texture(6)
    lb = t.copy()
    bh = int(S.FH * 0.3)
    lb[:bh] = 0
    lb[-bh:] = 0
    assert S.detect_fit(np.stack([lb] * 4)) == "letterbox"
    bl = t.copy()
    bl[:bh] = 90
    bl[-bh:] = 100
    assert S.detect_fit(np.stack([bl] * 4)) == "blur"
    assert S.detect_fit(np.stack([t] * 4)) == "crop"


# ------------------------------------------------------------------ audio helpers
def test_fit_max_shot_recovers_curve():
    period, m = 0.5, 3.0
    e = np.linspace(0, 1, 30)
    lens = period + (m - period) * (1 - e) ** 1.5
    assert S.fit_max_shot(lens, e, period) == pytest.approx(3.0, abs=1e-6)
    assert S.fit_max_shot(lens * 3, e, period) == 4.0           # clamped to MAX_SHOT


def test_lead_in():
    t = np.arange(0, 20, 0.02)
    rms = np.where(t < 7.0, 0.2, 1.0)
    assert S.lead_in(rms, t, 4.0, 20.0) == pytest.approx(3.0, abs=0.15)
    assert S.lead_in(np.ones_like(t), t, 4.0, 20.0) == 0.0


def test_music_entry_on_grid_vs_speech():
    et = np.arange(0, 12, 0.01)
    env = np.zeros_like(et)
    beats = np.arange(6.0, 12, 0.5)
    rng = np.random.default_rng(1)
    intro = et < 6.0
    env[intro] = rng.uniform(1, 3, intro.sum())     # speech: dense, irregular onset strength
    for x in np.arange(4.0, 12, 0.5):               # music from 4 s, under the last 2 s of intro
        env[np.argmin(np.abs(et - x))] = 6
    assert S.music_entry(env, et, beats, 6.0) == pytest.approx(2.0, abs=0.5)
    speech_only = env.copy()
    speech_only[intro] = rng.uniform(1, 3, intro.sum())
    assert S.music_entry(speech_only, et, beats, 6.0) < 0.5


# ------------------------------------------------------------------ profile
def stats(**kw):
    base = dict(source="x", duration=10, tempo=120, t_intro=0.0, cuts=[], on_beat=1.0,
                beats_per_cut=2, max_shot=2.0, median_shot=1.0, onset_cuts=0.0, punch=0.4,
                punch_zoom=0.12, flash=0.1, fit="crop", music_in="drop", rise=0.0, lead_in=0.0)
    base.update(kw)
    return S.RefStats(**base)


def test_build_profile_votes_and_medians():
    p = S.build_profile([stats(fit="blur", t_intro=4, music_in="rise", rise=2.0),
                         stats(fit="blur", beats_per_cut=1, t_intro=5, music_in="rise", rise=3.0),
                         stats(fit="crop", max_shot=3.0)], "me")
    assert p["fit"] == "blur" and p["pacing"]["beats_per_cut"] is None
    assert p["intro"] == {"use": True, "max": 4.5}
    assert p["music"]["music_in"] == "rise" and p["music"]["rise"] == 2.5
    assert p["pacing"]["max_shot"] == 2.0
    p = S.build_profile([stats(punch=0.0, punch_zoom=0.0), stats(punch=0.3, punch_zoom=0.1)], "me")
    assert p["fx"]["punch_zoom"] == 0.1          # zoom size only from videos that punch


def test_load_profile_rejects(tmp_path):
    f = tmp_path / "p.json"
    f.write_text('{"version": 99}')
    with pytest.raises(ClipsError, match="not a clips style"):
        S.load_profile(f)
    f.write_text('{"version": 1, "fit": "zoom"}')
    with pytest.raises(ClipsError, match="bad fit"):
        S.load_profile(f)


def test_links_are_downloaded(tmp_path, monkeypatch):
    got = []
    vid = tmp_path / "local.mp4"
    vid.write_bytes(b"x")

    def fake(url, cache):
        got.append((url, cache))
        return vid

    monkeypatch.setattr(S, "download", fake)
    out = S.resolve_refs(["https://www.tiktok.com/@a/video/1", str(vid)], tmp_path / "c")
    assert out == [vid, vid] and got == [("https://www.tiktok.com/@a/video/1", tmp_path / "c")]


def test_download_without_ytdlp(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(ClipsError, match="yt-dlp"):
        S.download("https://example.com/v", tmp_path)


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def footage(tmp_path_factory):
    d = tmp_path_factory.mktemp("style") / "in"
    d.mkdir()
    for n, p in [("a", "testsrc"), ("b", "smptebars"), ("c", "rgbtestsrc"), ("d", "testsrc2")]:
        make_video(d / f"{n}.mp4", "1280x720", 30, 10, p)
    return d


def test_learn_then_apply(footage, tmp_path, capsys):
    t = np.arange(0.25, 16, 0.5)
    song = tmp_path / "song.wav"
    sf.write(song, clicks(t, np.full(t.size, 0.8), 16), SR)
    ref = tmp_path / "ref.mp4"
    assert cli.main([str(footage), "--audio", str(song), "--no-intro", "--beats-per-cut", "2",
                     "--punch", "--flash", "--fit", "letterbox", "-o", str(ref)]) == 0
    prof_path = tmp_path / "creator.json"
    assert cli.main(["learn", str(ref), "-o", str(prof_path)]) == 0
    prof = json.loads(prof_path.read_text())
    assert prof["name"] == "creator"
    assert prof["pacing"]["beats_per_cut"] == 2 and prof["pacing"]["on_beat"] == 1.0
    assert prof["fit"] == "letterbox" and prof["intro"]["use"] is False
    assert prof["fx"]["punch"] == pytest.approx(0.4, abs=0.07)
    assert prof["fx"]["punch_zoom"] == pytest.approx(0.12, abs=0.03)
    assert prof["fx"]["flash"] == pytest.approx(0.133, abs=0.07)

    capsys.readouterr()
    assert cli.main([str(footage), "--audio", str(song), "--style", str(prof_path),
                     "--duration", "8", "--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["fit"] == "letterbox" and plan["intro"] == []
    shots = plan["shots"]
    assert all(abs(s["end"] - s["start"] - 1.0) < 0.05 for s in shots[1:-1])   # 2 beats each
    assert any(s["punch"] for s in shots) and any(s["flash"] for s in shots)
    assert plan["audio_start_candidates"]                                        # auto start
    # explicit flags beat the profile
    assert cli.main([str(footage), "--audio", str(song), "--style", str(prof_path),
                     "--duration", "8", "--dry-run", "--fit", "crop", "--audio-start", "1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["fit"] == "crop" and plan["audio_start"] == 1.0


def test_learns_intro_and_music_entry(footage, tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for f in footage.iterdir():
        (d / f.name).symlink_to(f)
    wav = tmp_path / "sp.wav"
    sf.write(wav, speech_like(6, 22050, seed=2), 22050)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=25:duration=6", "-i", str(wav), "-shortest",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(d / "talk.mp4")],
                   check=True)
    t = np.arange(0.25, 30, 0.5)
    song = tmp_path / "song.wav"
    sf.write(song, clicks(t, np.where(t < 8, 0.5, 0.9), 30), SR)    # louder from 8 s
    ref = tmp_path / "ref.mp4"
    assert cli.main([str(d), "--audio", str(song), "--intro", "talk.mp4@0.3-5", "--music-in",
                     "rise", "--rise", "2", "--audio-start", "6", "--duration", "16",
                     "--fit", "blur", "--no-loudnorm", "-o", str(ref)]) == 0
    # (no loudnorm: on a click-only test track the peak limiter flattens the 0.5 -> 0.9 lift)
    r = S.analyze_reference(ref)
    assert r.t_intro == pytest.approx(4.7, abs=0.1)
    assert r.music_in == "rise" and 1.0 <= r.rise <= 2.5
    assert r.lead_in == pytest.approx(2.0, abs=0.4)      # song gets louder 2 s into the music
    assert r.fit == "blur"
