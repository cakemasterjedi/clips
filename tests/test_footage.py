import json

import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import footage as F
from clips.sources import Source
import subprocess

from clips.ffmpeg import ffmpeg_exe
from conftest import SR, clicks, make_video


def make_color(path, color, secs=6):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c={color}:size=640x360:rate=30:duration={secs}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)


def clip_score(values):
    v = np.asarray(values, float)
    return F.ClipScore(np.arange(v.size) * F.STEP, v)


def test_score_clips_marks_bad_frames():
    m = np.array([5, 5, 50, 5, 5.0])                 # frame 2 is a hard cut inside the clip
    s = np.array([10, 10, 10, 0.2, 10.0])            # frame 3 is blank
    lum = np.array([100, 5, 100, 100, 100.0])        # frame 1 is black
    sc = F.score_clips([(m, s, lum)])[0].score
    assert sc[1] == sc[2] == sc[3] == -1.0
    assert sc[0] > 0 and sc[4] > 0


def test_hook_and_hits_get_best_windows():
    srcs = [Source("a", 4.0), Source("b", 4.0)]
    a = clip_score([0.1] * 10 + [0.9] * 5 + [0.1] * 5)   # a's best moment is 2-3 s
    b = clip_score([0.5] * 20)
    res = F.assign_smart([1.0, 1.0, 1.0], [0.0, 0.2, 0.9], srcs, [a, b])
    assert (res[0].source, res[0].start) == (0, 2.0)     # the hook takes the best window
    assert res[2].source == 1                            # not next to the hook's clip
    assert not res[1].loop


def test_windows_never_overlap_and_reserved_is_skipped():
    srcs = [Source("a", 10.0)]
    cs = clip_score(np.linspace(0, 1, 50))
    res = F.assign_smart([1.0] * 4, [0, 0.5, 0.4, 0.3], srcs, [cs], {0: [(8.0, 9.96)]})
    spans = sorted((r.start, r.start + 1.0) for r in res)
    assert all(b0 >= a1 + F.MIN_GAP - 1e-9 for (a0, a1), (b0, b1) in zip(spans, spans[1:]))
    assert all(s1 <= 8.0 - F.MIN_GAP + 1e-9 for _, s1 in spans)


def test_bad_windows_avoided_and_too_long_loops():
    srcs = [Source("a", 3.0), Source("b", 3.0)]
    bad = clip_score([-1.0] * 15)
    ok = clip_score([0.3] * 15)
    res = F.assign_smart([1.0], [0.0], srcs, [bad, ok])
    assert res[0].source == 1
    res = F.assign_smart([5.0], [0.0], srcs, [ok, ok])
    assert res[0].loop


def test_smart_pick_e2e_skips_black_and_frozen(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    make_color(d / "a_black.mp4", "black")
    make_color(d / "b_frozen.mp4", "gray")
    make_video(d / "c_moving.mp4", "640x360", 30, 6, "testsrc2")
    make_video(d / "d_moving.mp4", "640x360", 30, 6, "testsrc")
    t = np.arange(0.25, 6, 0.5)
    song = tmp_path / "s.wav"
    sf.write(song, clicks(t, np.full(t.size, 0.8), 6), SR)
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--smart-pick", "--dry-run"]) == 0
    shots = json.loads(capsys.readouterr().out)["shots"]
    names = [s["source"].rsplit("/", 1)[-1] for s in shots]
    assert all(n.startswith(("c_", "d_")) for n in names)
    assert all(a != b for a, b in zip(names, names[1:]))     # every cut switches clip
