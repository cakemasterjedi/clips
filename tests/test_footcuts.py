import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from clips import cli, footcuts as FC, render as R, ui
from clips.ffmpeg import ffmpeg_exe, probe
from clips.sources import Source


def events(n, spikes=(), cuts=(), act=0.0):
    ev = np.zeros(n)
    cut = np.zeros(n, bool)
    for i, v in spikes:
        ev[i] = v
    for i in cuts:
        ev[i], cut[i] = 1.0, True
    return FC.ClipEvents(ev, np.full(n, act), cut)


def test_pick_length_ends_after_the_event():
    ce = events(100, spikes=[(20, 0.9)])
    ln, ev, resume = FC.pick_length(ce, 0.0, 2.0, 10.0)
    assert ln == pytest.approx(2.0 + FC.TAIL) and ev == 0.9 and resume is None
    # nothing happens near the target: cut at the target
    ln, ev, _ = FC.pick_length(events(100, spikes=[(90, 1.0)]), 0.0, 2.0, 10.0)
    assert ln == pytest.approx(2.0) and ev == 0.0


def test_pick_length_stops_before_a_hard_cut():
    ce = events(100, cuts=[25])
    ln, ev, resume = FC.pick_length(ce, 0.5, 2.0, 10.0)
    assert ln == pytest.approx(2.4 - 0.5) and resume == pytest.approx(2.5)


def test_plan_shots_cover_length_and_alternate_clips():
    srcs = [Source(__import__("pathlib").Path(f"{k}.mp4"), 10.0) for k in "ab"]
    evs = [events(100, spikes=[(i, 0.8) for i in range(7, 100, 13)], act=0.8),
           events(100, cuts=[30], act=0.2)]
    rows = FC.plan_shots(evs, srcs, 12.0, 30)
    assert sum(f for f, *_ in rows) == 360
    assert all(f >= 8 for f, *_ in rows[:-1])
    assert [a.source for _, a, *_ in rows[:4]] == [0, 1, 0, 1]
    assert rows[0][2] == 0.0                         # the first shot never has a hit
    # busy clip a gets shorter shots than calm clip b
    a_len = [f for f, a, *_ in rows if a.source == 0]
    b_len = [f for f, a, *_ in rows if a.source == 1]
    assert np.mean(a_len) < np.mean(b_len)
    # clip b's second shot starts on the new scene after its hard cut
    b_starts = [a.start for _, a, *_ in rows if a.source == 1]
    assert b_starts[1] >= 3.0 - 1e-6


@pytest.fixture(scope="module")
def footage(tmp_path_factory):
    d = tmp_path_factory.mktemp("fc") / "in"
    d.mkdir()
    ff = ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=640x360:rate=30:duration=8", "-f", "lavfi", "-i",
                    "sine=f=300:d=8,volume='if(lt(mod(t,2.3),0.15),1,0.02)':eval=frame",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(d / "a.mp4")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "smptebars=size=640x360:rate=30:duration=4", "-f", "lavfi", "-i",
                    "color=red:size=640x360:rate=30:duration=4", "-filter_complex",
                    "[0][1]concat=n=2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(d / "b.mp4")], check=True)
    return d


def test_footage_cuts_dry_run(footage, capsys):
    assert cli.main([str(footage), "--no-music", "--cuts", "footage", "--duration", "10",
                     "--punch", "--dry-run"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["cuts"] == "footage" and doc["bpm"] is None
    shots = doc["shots"]
    assert sum(s["frames"] for s in shots) == 300
    # clip a beeps at 2.3 s: its first shot ends just after the beep
    assert shots[0]["source"].endswith("a.mp4")
    assert 2.3 < shots[0]["end"] <= 2.3 + FC.TAIL + 0.15
    # clip b has a hard cut at 4 s: no b shot spans it
    for s in shots:
        if s["source"].endswith("b.mp4"):
            span = (s["source_start"], s["source_start"] + s["frames"] / 30)
            assert not (span[0] < 4.0 - 0.05 and span[1] > 4.0 + 0.05)
    assert any(s["punch"] for s in shots)
    # same seed, same edit
    assert cli.main([str(footage), "--no-music", "--cuts", "footage", "--duration", "10",
                     "--punch", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["shots"] == shots


def test_flag_checks(footage, tmp_path, capsys):
    song = tmp_path / "s.wav"
    sf.write(song, np.zeros(22050 * 3), 22050)
    assert cli.main([str(footage), "--audio", str(song), "--cuts", "footage", "--dry-run"]) == 1
    assert "--cuts footage" in capsys.readouterr().err
    assert cli.main([str(footage), "--no-music", "--source-audio", "mix", "--dry-run"]) == 1
    assert "--source-audio needs a song" in capsys.readouterr().err


def test_clip_sound_graph_and_args(tmp_path):
    cs = R.ClipSound(tmp_path / "c.wav", R.SourceAudio("duck", -6, "strong"))
    a = R.final_args(tmp_path / "l.txt", tmp_path / "s.mp3", 1.0, 5.0, 30, None,
                     tmp_path / "o.mp4", clip_sound=cs)
    fc = a[a.index("-filter_complex") + 1]
    assert str(cs.wav) in a and "sidechaincompress" in fc and "ratio=12" in fc
    assert "[2:a]" in fc and "volume=-6dB" in fc
    mix = R.ClipSound(cs.wav, R.SourceAudio("mix"))
    fc = R.clip_sound_graph("m", 3, 5.0, mix, "o")
    assert "sidechaincompress" not in fc and "[3:a]" in fc and fc.endswith("[o]")
    plain = R.final_args(tmp_path / "l.txt", tmp_path / "s.mp3", 1.0, 5.0, 30, None,
                         tmp_path / "o.mp4")
    assert str(cs.wav) not in plain
    g = R.intro_audio_graph([2.0], 6.0, 2.0, 0.0, clip_sound=cs)
    assert "[3:a]" in g and "[dlg][mus]amix" in g


def band_rms(path, lo, hi, a, b):
    y, sr = sf.read(path)
    y = y.mean(axis=1) if y.ndim > 1 else y
    seg = y[int(a * sr):int(b * sr)]
    spec = np.abs(np.fft.rfft(seg))
    f = np.fft.rfftfreq(seg.size, 1 / sr)
    return float(spec[(f > lo) & (f < hi)].sum())


def test_ducking_lowers_the_song_under_clip_sound(tmp_path):
    """Song = steady 800 Hz; clip sound = 300 Hz from 1.5 to 3 s. With duck the 800 Hz part
    drops while the 300 Hz plays and comes back after; with mix it stays put."""
    ff = ffmpeg_exe()
    song, clip = tmp_path / "song.wav", tmp_path / "clip.wav"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=f=800:d=6:sample_rate=44100", "-af", "volume=0.5", str(song)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=f=300:d=6:sample_rate=48000",
                    "-af", "volume='if(between(t,1.5,3),0.8,0)':eval=frame", "-ac", "2",
                    str(clip)], check=True)
    res = {}
    for mode in ("mix", "duck"):
        cs = R.ClipSound(clip, R.SourceAudio(mode, 0, "strong"))
        out = tmp_path / f"{mode}.wav"
        fc = (f"[0:a]{R.AUDIO_FMT}[m];" + R.clip_sound_graph("m", 1, 6.0, cs, "o"))
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(song), "-i", str(clip),
                        "-filter_complex", fc, "-map", "[o]", str(out)], check=True)
        res[mode] = (band_rms(out, 700, 900, 2.0, 2.8), band_rms(out, 700, 900, 4.5, 5.5),
                     band_rms(out, 250, 350, 2.0, 2.8))
    during, after, clip_level = res["duck"]
    assert during < 0.5 * after                       # song ducked under the clip sound
    assert clip_level > 0                             # and the clip sound is there
    m_during, m_after, _ = res["mix"]
    assert m_during == pytest.approx(m_after, rel=0.15)


def test_render_with_source_audio(footage, tmp_path):
    ff = ffmpeg_exe()
    song = tmp_path / "song.wav"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=f=800:d=10",
                    str(song)], check=True)
    out = tmp_path / "o.mp4"
    assert cli.main([str(footage), "--audio", str(song), "--no-intro", "--source-audio", "duck",
                     "--duration", "3", "--preview", "--gpu", "off", "-o", str(out)]) == 0
    info = probe(out)
    assert info.has_audio and abs(info.duration - 3.0) < 0.1


def test_ui_maps_the_new_controls(tmp_path):
    song = tmp_path / "s.mp3"
    song.write_bytes(b"x")
    base = {"input": str(tmp_path), "name": "e"}
    argv = ui.render_argv({**base, "music": "file", "audio": str(song), "source_audio": "duck",
                           "duck": "strong", "source_volume": "-6", "cuts": "footage",
                           "bpm": "90"}, tmp_path)
    assert argv[argv.index("--source-audio") + 1] == "duck"
    assert argv[argv.index("--duck") + 1] == "strong"
    assert argv[argv.index("--source-volume") + 1] == "-6"
    assert "--cuts" not in argv and "--bpm" not in argv          # no-song options left out
    argv = ui.render_argv({**base, "music": "clips", "cuts": "footage", "bpm": "90",
                           "source_audio": "mix", "duck": "strong"}, tmp_path)
    assert argv[argv.index("--cuts") + 1] == "footage"
    assert "--bpm" not in argv and "--source-audio" not in argv and "--duck" not in argv
    argv = ui.render_argv({**base, "music": "file", "audio": str(song), "source_audio": "",
                           "duck": "strong", "source_volume": "-6"}, tmp_path)
    assert "--duck" not in argv and "--source-volume" not in argv
    flags = {o["flag"] for o in ui.option_list()}
    assert not flags & {"--cuts", "--source-audio", "--source-volume", "--duck"}
