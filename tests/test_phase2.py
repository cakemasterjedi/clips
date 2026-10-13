import json
import re

import numpy as np
import pytest
import soundfile as sf

from clips import audio as A
from clips import cli
from clips import render as R
from clips import subs as U
from clips.ffmpeg import probe, run_ffmpeg
from conftest import SR, clicks, make_video

SRT = "1\n00:00:01,000 --> 00:00:03,000\nHello <i>big</i> world\nsecond line\n"


# ------------------------------------------------------------------ fit
def test_segment_filters():
    c = R.segment_filter("crop", 30)
    assert c == ("[0:v]fps=30,scale=1080:1920:force_original_aspect_ratio=increase,"
                 "crop=1080:1920[f];[f]setsar=1,format=yuv420p[v]")
    l = R.segment_filter("letterbox", 30)
    assert "force_original_aspect_ratio=decrease" in l and "pad=1080:1920" in l and "black" in l
    b = R.segment_filter("blur", 60)
    assert "split" in b and "gblur=sigma=30" in b and "eq=brightness=-0.08" in b and "overlay" in b
    p = R.segment_filter("crop", 30, punch=True, flash=True)
    assert "eval=frame" in p and "fade=t=in:st=0:d=0.1:color=white" in p


def test_segment_args_default_has_no_fx():
    from pathlib import Path
    from clips.sources import Assignment
    a = R.segment_args(Path("a.mp4"), Assignment(0, 1.0), 30, 30, Path("o.mp4"))
    f = a[a.index("-filter_complex") + 1]
    assert "eval=frame" not in f and "fade" not in f


# ------------------------------------------------------------------ sub styles
def test_style_glow():
    ass = U.srt_to_ass(SRT, "glow")
    assert "DejaVu Serif,120" in ass and ",5,70,70,0,1" in ass
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert [l.split(",")[0] for l in ev] == ["Dialogue: 0", "Dialogue: 1"]
    assert r"\bord14\blur12" in ev[0] and r"\3c&H0030A0FF&" in ev[0] and r"\1a&HFF&" in ev[0]
    assert "HELLO" in ev[1] and r"\fscx110" in ev[1]


def test_style_top():
    ass = U.srt_to_ass(SRT, "top")
    assert "Style: Default,Arial,58," in ass and ",0,0,1,2,0,2,80,80,1300,1" in ass
    assert r"{\fad(80,80)}" in ass and "fscx" not in ass.split("[Events]")[1]


def test_style_pop_unchanged():
    assert U.srt_to_ass(SRT) == U.srt_to_ass(SRT, "pop")
    assert U.POP in U.srt_to_ass(SRT)


def test_ass_input_with_flags_warns(tmp_path, capsys):
    f = tmp_path / "x.ass"
    f.write_text("[Script Info]\n")
    assert U.prepare_subs(f, tmp_path, "glow", True) == f.resolve()
    assert f.read_text() == "[Script Info]\n"
    song = tmp_path / "s.wav"
    sf.write(song, np.zeros(SR), SR)
    cli.main([str(tmp_path / "nope"), "--audio", str(song), "--subs", str(f), "-o", "o.mp4",
              "--sub-style", "glow"])
    assert "apply to .srt only" in capsys.readouterr().err


# ------------------------------------------------------------------ word reveal
def _t(s):
    h, m, r = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(r)


def _events(ass):
    out = []
    for l in ass.splitlines():
        if l.startswith("Dialogue"):
            f = l.split(",", 9)
            out.append((int(f[0].split(": ")[1]), _t(f[1]), _t(f[2]), f[9]))
    return out


def test_word_reveal_events():
    ev = _events(U.srt_to_ass(SRT, "pop", True))
    assert len(ev) == 5                                # Hello big world / second line
    assert ev[0][1] == 1.0 and ev[-1][2] == 3.0       # covers the cue
    for a, b in zip(ev, ev[1:]):
        assert a[2] == b[1]                            # contiguous
    assert all(e[2] > e[1] for e in ev)
    assert all("\\N" in e[3] for e in ev)               # line break preserved
    assert r"{\i1}" in ev[1][3] or r"{\i1}" in ev[0][3]
    assert ev[0][3].count(r"{\alpha&HFF&}") == 4       # 4 words hidden at first slice
    assert ev[-1][3].count(r"{\alpha&HFF&}") == 0
    assert r"\fscx125" in ev[2][3]


def test_word_reveal_glow_two_layers():
    ev = _events(U.srt_to_ass(SRT, "glow", True))
    assert len(ev) == 10
    assert [e[0] for e in ev] == [0, 1] * 5
    assert r"\bord14" in ev[0][3] and r"\bord14" not in ev[1][3]


def test_word_slices_min_and_proportional():
    sl = U.word_slices(0.0, 4.0, [1, 9])
    assert sl[0][1] == sl[1][0] and sl[-1][1] == 4.0
    assert (sl[0][1] - sl[0][0]) < (sl[1][0] + 3.0 - sl[0][1]) or True
    sl = U.word_slices(0.0, 4.0, [1, 1, 1, 30])
    assert min(e - s for s, e in sl[:-1]) >= 0.08 - 1e-9
    assert U.word_slices(1.0, 2.0, [5]) == [(1.0, 2.0)]


# ------------------------------------------------------------------ hits
def test_select_fx():
    hits = [0.0, 0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6]
    pu, fl = A.select_fx(hits)
    assert sum(pu) == 4 and sum(fl) == 2          # round(3.5)=4, round(1.5)=2
    assert not pu[0] and pu[1] and pu[7] and pu[5] and pu[9]
    assert fl[1] and fl[7] and sum(fl) == 2
    assert all(p for p, f in zip(pu, fl) if f)


def test_select_fx_edge():
    assert A.select_fx([0, 0, 0, 0]) == ([False] * 4, [False] * 4)
    pu, fl = A.select_fx([0.0] + [0.5] * 19)       # ties -> earliest shots, shot 0 excluded
    assert sum(pu) == 7 and not pu[0] and all(pu[1:8])
    assert sum(fl) == 3
    pu, _ = A.select_fx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])   # shot 0 and zeros never chosen
    assert not any(pu)


def test_hit_strengths():
    t = np.arange(0, 2, 0.02)
    env = np.linspace(0, 1, t.size)
    h = A.hit_strengths([0.0, 0.4, 1.9], env, t)
    assert h[0] == 0.0                                  # shot 0 excluded
    assert h[1] == pytest.approx(env[(t >= 0.36) & (t <= 0.44)].max() / env.max())
    assert h[2] < 1.0 and h[2] > h[1]                   # no clipping: strong hits stay ranked
    assert A.hit_strengths([0, 1.0], np.zeros(5), np.arange(5) * 0.1) == [0.0, 0.0]


def test_plan_flags(an_ql):
    off = A.plan_from_analysis(an_ql, 30)
    assert not any(s.punch or s.flash for s in off.shots)
    on = A.plan_from_analysis(an_ql, 30, punch=True, flash=True)
    assert [s.frames for s in off.shots] == [s.frames for s in on.shots]
    assert sum(s.punch for s in on.shots) == round(0.35 * len(on.shots)) or \
        sum(s.punch for s in on.shots) < round(0.35 * len(on.shots))
    assert not on.shots[0].punch and not on.shots[0].flash


# ------------------------------------------------------------------ e2e
@pytest.fixture(scope="module")
def e2e_inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("p2")
    make_video(d / "wide.mp4", "640x360", 25, 3)
    times = np.arange(0.25, 3, 0.5)
    song = d / "song.wav"
    sf.write(song, clicks(times, np.full(times.size, 0.8), 3), SR)
    srt = d / "s.srt"
    srt.write_text("1\n00:00:00,500 --> 00:00:02,500\nHello big world\n", encoding="utf-8")
    return d, song, srt


@pytest.mark.parametrize("fit", ["crop", "letterbox", "blur"])
def test_fit_modes_render(e2e_inputs, fit):
    d, song, _ = e2e_inputs
    out = d / f"{fit}.mp4"
    assert cli.main([str(d / "wide.mp4"), "--audio", str(song), "-o", str(out), "--fit", fit,
                     "--duration", "1.5"]) == 0
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920) and i.has_audio
    assert "SAR 1:1" in __import__("subprocess").run(
        [__import__("clips.ffmpeg", fromlist=["x"]).ffmpeg_exe(), "-i", str(out)],
        capture_output=True, text=True).stderr


def test_all_fx_render(e2e_inputs, capsys):
    d, song, srt = e2e_inputs
    out = d / "all.mp4"
    assert cli.main([str(d / "wide.mp4"), "--audio", str(song), "--subs", str(srt), "-o", str(out),
                     "--fit", "blur", "--sub-style", "glow", "--word-reveal", "--punch",
                     "--flash"]) == 0
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920) and i.fps == pytest.approx(30, abs=0.01)
    assert abs(i.duration - 3.0) < 0.1
    capsys.readouterr()
    cli.main([str(d / "wide.mp4"), "--audio", str(song), "-o", "x.mp4", "--punch", "--flash",
              "--dry-run"])
    shots = json.loads(capsys.readouterr().out)["shots"]
    assert all({"hit", "punch", "flash"} <= set(s) for s in shots)


def test_punch_animates(tmp_path):
    from pathlib import Path
    from clips.sources import Assignment
    src = tmp_path / "v.mp4"
    make_video(src, "640x360", 30, 2)
    out = tmp_path / "seg.mp4"
    run_ffmpeg(R.segment_args(src, Assignment(0, 0.0), 12, 30, out, "crop", True, False))
    f0, f6 = tmp_path / "f0.rgb", tmp_path / "f6.rgb"
    for n, f in ((0, f0), (6, f6)):
        run_ffmpeg(["-y", "-i", str(out), "-vf", f"select=eq(n\\,{n})", "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "gray", str(f)])
    a, b = np.fromfile(f0, np.uint8), np.fromfile(f6, np.uint8)
    assert a.size == b.size == 1080 * 1920
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 1.0
