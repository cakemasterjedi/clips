import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import reframe as RF
from clips.ffmpeg import ClipsError
from conftest import SR, clicks
from test_reframe import moving_box, white_share


def K(t, yaw=0.0, pitch=0.0, fov=60.0, roll=0.0):
    return RF.Keyframe(t, yaw, pitch, fov, roll)


# ------------------------------------------------------------------ unit
def test_interpolate_eases_holds_and_wraps():
    keys = [K(1, yaw=170, fov=40), K(3, yaw=-170, fov=80, pitch=20)]
    assert RF.interpolate(keys, 0) == (170, 0, 40, 0)                 # hold before
    assert RF.interpolate(keys, 9) == (-170, 20, 80, 0)               # hold after
    y, p, f, r = RF.interpolate(keys, 2)
    assert abs(abs(y) - 180) < 1e-6 and p == pytest.approx(10) and f == pytest.approx(60)
    _, p, _, _ = RF.interpolate(keys, 1.5)                            # cosine ease: slow start
    assert p == pytest.approx(20 * (0.5 - 0.5 * np.cos(np.pi * 0.25)))
    ys = [RF.interpolate(keys, t)[0] for t in np.linspace(1, 3, 21)]
    assert all(abs(y) >= 170 - 1e-6 for y in ys)                      # short way, via 180


def test_sidecar_roundtrip_and_errors(tmp_path):
    clip = tmp_path / "VID_00_001.insv"
    clip.write_bytes(b"")
    assert RF.load_keyframes(clip) is None
    side = RF.sidecar_path(clip)
    assert side.name == "VID_00_001.insv.view.json"
    side.write_text(json.dumps(RF.keyframes_json([K(2, 10), K(0.5, -20, 5, 70, 3)])))
    keys = RF.load_keyframes(clip)
    assert [k.t for k in keys] == [0.5, 2] and keys[0].roll == 3
    side.write_text('{"version": 1, "keyframes": [{"t": 0, "fov": 200}]}')
    with pytest.raises(ClipsError, match="out of range"):
        RF.load_keyframes(clip)
    side.write_text('{"keyframes": []}')
    with pytest.raises(ClipsError, match="not a clips view file"):
        RF.load_keyframes(clip)


def test_animate_sends_only_changes():
    v = RF.keyed_view(RF.View("equirect"),
                      [K(0, yaw=0), K(1, yaw=90), K(2, yaw=90, pitch=30), K(3, yaw=90, pitch=30, fov=90)],
                      0.0)
    txt, anim = RF.animate(v, 120, 30, (1080, 1920))
    assert anim.canvas is None and anim.tilt and "rotate" not in txt
    rows = [(float(l.split()[0]), l) for l in txt.strip().splitlines()]
    assert all("crop@fk x" in l and "crop@fp" not in l for t, l in rows if t < 1.0)
    pitch_x = [int(c.split()[-1]) for _, l in rows for c in l.rstrip(";").split(", ")
               if "crop@fp x" in c]
    assert pitch_x[-1] == RF._scroll(-30, 3840) == 3520               # 30 deg up
    zooms = [l for t, l in rows if "scale@fz" in l]
    assert zooms and all(t >= 2.0 for t, l in rows if "scale@fz" in l) and anim.zoom == (90, 60)
    assert "v360" not in txt                                          # v360 never re-solves
    assert RF._scroll(90, 3840) == 960 and RF._scroll(-90, 3840) == 2880
    assert RF.interpolate(v.keys, 2.0) == (90, 30, 60, 0)
    pan = RF.keyed_view(RF.View("equirect"), [K(0, yaw=0, pitch=10), K(1, yaw=90, pitch=10)], 0.0)
    _, anim = RF.animate(pan, 30, 30, (1080, 1920))
    assert not anim.tilt                                              # pitch fixed: no side pano


def test_eq_size():
    assert RF.eq_size(RF.View("equirect"), 5760, 2880) == (5760, 2880)
    assert RF.eq_size(RF.View("equirect"), 7680, 3840) == (5760, 2880)
    assert RF.eq_size(RF.View("dfisheye", "sbs"), 2880, 1440) == (2880, 1440)
    assert RF.eq_size(RF.View("dfisheye", "pair"), 1440, 1440) == (2880, 1440)
    assert RF.eq_size(RF.View("equirect"), 0, 0) == RF.EQ_SIZE


def test_roll_canvas():
    assert RF.roll_canvas((1080, 1920), 0) == (1080, 1920)
    cw, ch = RF.roll_canvas((1080, 1920), 20)
    assert cw >= 1080 * np.cos(np.radians(20)) + 1920 * np.sin(np.radians(20)) and cw % 2 == 0
    v = RF.keyed_view(RF.View("equirect"), [K(0), K(1, roll=-15)], 0.0)
    txt, anim = RF.animate(v, 45, 30, (1080, 1920))
    assert anim.canvas == RF.roll_canvas((1080, 1920), 15) and "rotate@fr a" in txt
    # the same focal length on the larger canvas: diagonal fov grows
    assert RF.d_fov(60, (1080, 1920), anim.canvas) > RF.d_fov(60, (1080, 1920))


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def pano(tmp_path_factory):
    d = tmp_path_factory.mktemp("kf")
    p = d / "pano.mp4"
    moving_box(p, (1280, 640), 960, 320, secs=8)       # action at yaw +90
    s = d / "s.wav"
    t = np.arange(0.25, 4, 0.5)
    sf.write(s, clicks(t, np.full(t.size, 0.8), 4), SR)
    return p, s


def expect_visible(plan, keys, t):
    """Is the box (yaw +90 on the horizon, +-25 deg with its wobble) in the 60 x 91 deg view
    at output time t, per the keyframes? None near the edge of the view."""
    sh = next(s for s in plan["shots"] if s["start"] <= t < s["end"])
    y, p, _, _ = RF.interpolate(keys, sh["source_start"] + t - sh["start"])
    dy, dp = abs(RF._wrap(y - 90)), abs(p)
    if dy < 20 and dp < 20:
        return True
    if dy > 70 or dp > 75:
        return False
    return None


@pytest.mark.parametrize("keys", [
    [K(0, yaw=-90), K(1, yaw=-90), K(3, yaw=90)],                       # pan (scroll)
    [K(0, yaw=90, pitch=-85), K(1, yaw=90, pitch=-85), K(3, yaw=90)],   # tilt (v360 command)
    [K(0, yaw=90), K(1, yaw=90), K(3.5, yaw=-90)],                      # pan away
    [K(0, yaw=90, pitch=-85, fov=40), K(1, yaw=90, pitch=-85, fov=40),  # tilt + zoom + roll
     K(3, yaw=90, fov=70, roll=20)],
], ids=["pan", "tilt", "pan-away", "tilt-zoom-roll"])
def test_keyframes_drive_render(pano, tmp_path, capsys, keys):
    src, song = pano
    d = tmp_path / "in"
    d.mkdir()
    clip = d / "pano.mp4"
    shutil.copy(src, clip)
    RF.sidecar_path(clip).write_text(json.dumps(RF.keyframes_json(keys)))
    common = [str(d), "--audio", str(song), "--360", "equirect", "--no-loudnorm"]
    assert cli.main(common + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert all(s["view"]["keyframes"] == 3 for s in plan["shots"])
    probes = [t for t in np.arange(0.2, 4, 0.3) if expect_visible(plan, keys, t) is not None]
    want = [expect_visible(plan, keys, t) for t in probes]
    assert any(want) and not all(want)                 # the test can tell the views apart
    out = tmp_path / "o.mp4"
    assert cli.main(common + ["-o", str(out)]) == 0
    assert [white_share(out, t) > 0.002 for t in probes] == want


def test_fov_keyframes_zoom(pano, tmp_path):
    src, song = pano
    d = tmp_path / "in"
    d.mkdir()
    clip = d / "pano.mp4"
    shutil.copy(src, clip)
    RF.sidecar_path(clip).write_text(json.dumps(RF.keyframes_json(
        [K(0, yaw=90, fov=30), K(1, yaw=90, fov=30), K(3, yaw=90, fov=120)])))
    out = tmp_path / "o.mp4"
    assert cli.main([str(d), "--audio", str(song), "--360", "equirect", "--no-loudnorm",
                     "--beats-per-cut", "16", "-o", str(out)]) == 0
    assert white_share(out, 0.4) > 3 * white_share(out, 3.6) > 0


def test_yaw_flag_beats_keyframes(pano, tmp_path, capsys):
    src, song = pano
    d = tmp_path / "in"
    d.mkdir()
    shutil.copy(src, d / "pano.mp4")
    RF.sidecar_path(d / "pano.mp4").write_text(json.dumps(RF.keyframes_json([K(0, 10), K(2, 50)])))
    assert cli.main([str(d), "--audio", str(song), "--360", "equirect", "--yaw", "-30",
                     "--dry-run"]) == 0
    v = json.loads(capsys.readouterr().out)["shots"][0]["view"]
    assert v["yaw"] == -30 and "keyframes" not in v


def test_animated_graph_matches_static_view(tmp_path):
    """After the keyframes settle, the animated graph (scroll + pitch deltas + d_fov + rotate)
    shows exactly what a plain v360 view with those angles shows."""
    import subprocess
    from dataclasses import replace
    from clips.ffmpeg import ffmpeg_exe
    src = tmp_path / "eq.mp4"
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-f", "lavfi", "-i",
                    "testsrc2=size=1920x960:rate=30:duration=2", "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    size = (270, 480)
    v = RF.keyed_view(RF.View("equirect"),
                      [K(0, yaw=-20, pitch=-10, fov=75), K(0.6, yaw=50, pitch=20, fov=50, roll=15)], 0.0)
    txt, av = RF.animate(v, 45, 30, size)
    assert av.tilt and av.canvas and av.zoom == (75, 50)
    cmd = tmp_path / "v.cmd"
    cmd.write_text(txt)
    anim = RF.view_graph(replace(av, cmds=cmd), 30, "0:v", None, "f", size)
    end = RF.View("equirect", yaw=50, pitch=20, fov=50, roll=15)
    static = RF.view_graph(end, 30, "0:v", None, "f", size)

    def frame(graph):
        raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(src),
                              "-filter_complex", graph + ";[f]format=gray[o]", "-map", "[o]",
                              "-frames:v", "45", "-f", "rawvideo", "-"],
                             stdout=subprocess.PIPE, check=True).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, size[1], size[0])[-1].astype(float)

    a, b = frame(anim), frame(static)
    assert np.abs(a - b)[20:-20, 20:-20].mean() < 6           # same view (resampling aside)
    first = RF.view_graph(RF.still(v), 30, "0:v", None, "f", size)
    assert np.abs(frame(first) - b).mean() > 20                # and the start view differs
