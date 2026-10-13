import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import reframe as RF
from clips.ffmpeg import ffmpeg_exe, parse_probe, probe
from conftest import SR, clicks


def ff(*args):
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *map(str, args)], check=True)


def moving_box(path, size, cx, cy, secs=4, box=80):
    """Black frame with a white box wobbling around (cx, cy)."""
    w, h = size
    ff("-f", "lavfi", "-i", f"color=black:s={w}x{h}:r=30:d={secs}",
       "-f", "lavfi", "-i", f"color=white:s={box}x{box}:r=30:d={secs}",
       "-filter_complex", f"[0][1]overlay=x='{cx - box // 2}+50*sin(2*PI*t)':y={cy - box // 2}",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", path)


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    p = tmp_path_factory.mktemp("rf") / "s.wav"
    t = np.arange(0.25, 4, 0.5)
    sf.write(p, clicks(t, np.full(t.size, 0.8), 4), SR)
    return p


@pytest.fixture(scope="module")
def equirect(tmp_path_factory):
    d = tmp_path_factory.mktemp("eq")
    p = d / "pano.mp4"
    moving_box(p, (1280, 640), 960, 320)          # yaw +90, on the horizon
    return p


@pytest.fixture(scope="module")
def fisheye_files(tmp_path_factory, equirect):
    """Insta360-style dual fisheye from a pano whose action sits at yaw +135."""
    d = tmp_path_factory.mktemp("insv")
    pano = d / "pano135.mp4"
    moving_box(pano, (1280, 640), 1120, 320)
    sbs = d / "sbs" / "VID_20261013_120000_00_001.insv"
    sbs.parent.mkdir()
    ff("-i", pano, "-vf", f"v360=input=e:output=dfisheye:h_fov={RF.FISHEYE_FOV}:"
       f"v_fov={RF.FISHEYE_FOV}:w=1280:h=640", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-f", "mp4", sbs)
    streams = d / "streams" / "VID_20261013_120000_00_002.insv"
    streams.parent.mkdir()
    ff("-i", sbs, "-filter_complex", "[0]crop=640:640:0:0[a];[0]crop=640:640:640:0[b]",
       "-map", "[a]", "-map", "[b]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4", streams)
    pair = d / "pair"
    pair.mkdir()
    ff("-i", sbs, "-vf", "crop=640:640:0:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4",
       pair / "VID_20261013_120000_00_003.insv")
    ff("-i", sbs, "-vf", "crop=640:640:640:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-f", "mp4",
       pair / "VID_20261013_120000_10_003.insv")
    return sbs, streams, pair


def white_share(path, t=1.0):
    raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-ss", str(t), "-i", str(path),
                          "-frames:v", "1", "-vf", "scale=108:192,format=gray", "-f", "rawvideo", "-"],
                         stdout=subprocess.PIPE, check=True).stdout
    return float((np.frombuffer(raw, np.uint8) > 200).mean())


# ------------------------------------------------------------------ unit
def test_detection():
    info = parse_probe("Duration: 00:00:04.00\n  Stream #0:0: Video: h264, yuv420p, 5760x2880, 30 fps\n"
                       "    Side data:\n      spherical: equirectangular (0.000000/0.000000/0.000000)\n")
    assert info.spherical and info.video_streams == 1
    assert RF.detect_view(RF.Path("a.mp4"), info).kind == "equirect"
    assert RF.detect_view(RF.Path("a.mp4"), info, "off").kind == "flat"
    two = parse_probe("  Stream #0:0: Video: h264, 2880x2880, 30 fps\n  Stream #0:1: Video: h264, 2880x2880, 30 fps\n")
    assert two.video_streams == 2
    v = RF.detect_view(RF.Path("x.insv"), two)
    assert (v.kind, v.lenses) == ("dfisheye", "streams")
    flat = parse_probe("  Stream #0:0: Video: h264, 1920x1080, 30 fps\n")
    assert RF.detect_view(RF.Path("a.mp4"), flat).kind == "flat"


def test_2to1_movie_without_metadata_stays_flat():
    """A 3840x1920 movie is not 360 unless its metadata says so (user's Lanterns.mkv)."""
    movie = parse_probe("Input #0, matroska,webm, from 'Lanterns spherical.mkv':\n"
                        "  Stream #0:0: Video: hevc (Main 10), yuv420p10le, 3840x1920, 23.98 fps\n")
    assert not movie.spherical and RF.looks_360(movie)
    assert RF.detect_view(RF.Path("Lanterns.mkv"), movie).kind == "flat"
    assert RF.detect_view(RF.Path("Lanterns.mkv"), movie, "on").kind == "equirect"
    assert RF.detect_view(RF.Path("x.insv"), movie, "on").kind == "dfisheye"
    for meta in ("    Side data:\n      Spherical Mapping: equirectangular (0.000000/0.000000/0.000000)\n",
                 "    Side data:\n      spherical: tiled equirectangular [0, 0, 0, 0]\n",
                 "    Metadata:\n      ProjectionType  : equirectangular\n"):
        assert parse_probe("  Stream #0:0: Video: h264, 1920x1080, 30 fps\n" + meta).spherical
    cube = parse_probe("  Stream #0:0: Video: h264, 3840x1920\n    Side data:\n      spherical: cubemap\n")
    assert not cube.spherical


def test_v_fov_and_graph():
    assert RF.v_fov(60, (1080, 1920)) == pytest.approx(91.5, abs=0.2)
    g = RF.view_graph(RF.View("dfisheye", "pair", yaw=30), 30, "0:v", "1:v", "f", (1080, 1920))
    assert "[0:v][1:v]hstack=inputs=2" in g and "input=dfisheye:ih_fov=200" in g
    assert "yaw=30.00" in g and "w=1080:h=1920" in g


def test_yaw_from_motion_avoids_seams():
    f = np.zeros((4, RF.AH, RF.AW), np.float32)
    col = int((90 + 180) / 360 * RF.AW)               # motion right on the +90 seam...
    f[1::2, 60:68, col - 3:col + 3] = 255
    col2 = int((-20 + 180) / 360 * RF.AW)             # ...and weaker motion at -20
    f[1::2, 60:68, col2 - 2:col2 + 2] = 255
    assert RF.yaw_pitch_from_motion(f, seams=False)[0] == pytest.approx(90, abs=4)
    assert RF.yaw_pitch_from_motion(f, seams=True)[0] == pytest.approx(-20, abs=4)


def test_motion_track():
    f = np.zeros((9, 40, 80), np.float32)
    for i in range(9):                                  # box moves from x 10 to x 60
        x = 10 + i * 6
        f[i, 10:20, x:x + 6] = 255
    x0, _, x1, _ = RF.motion_track(f)
    assert x0 < 0.35 and x1 > 0.65


def test_pair_detection(tmp_path):
    a = tmp_path / "VID_1_00_007.insv"
    b = tmp_path / "VID_1_10_007.insv"
    a.write_bytes(b"")
    assert RF.pair_of(a) is None
    b.write_bytes(b"")
    assert RF.pair_of(a) == b and RF.is_pair_secondary(b) and not RF.is_pair_secondary(a)


# ------------------------------------------------------------------ end to end
def test_equirect_auto_view(equirect, song, tmp_path, capsys):
    assert cli.main([str(equirect), "--audio", str(song), "--no-intro", "--360", "equirect",
                     "--dry-run"]) == 0
    shots = json.loads(capsys.readouterr().out)["shots"]
    assert all(s["view"]["kind"] == "equirect" for s in shots)
    assert all(abs(s["view"]["yaw"] - 90) < 20 for s in shots)
    out = tmp_path / "o.mp4"
    assert cli.main([str(equirect), "--audio", str(song), "--no-intro", "--360", "equirect",
                     "--preview", "-o", str(out)]) == 0
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920)
    assert white_share(out) > 0.005                       # the action is in view
    away = tmp_path / "away.mp4"
    assert cli.main([str(equirect), "--audio", str(song), "--no-intro", "--360", "equirect",
                     "--yaw", "-90", "--preview", "-o", str(away)]) == 0
    assert white_share(away) < 0.001                      # looking the other way


@pytest.mark.parametrize("layout", ["sbs", "streams", "pair"])
def test_native_insv_layouts(fisheye_files, song, tmp_path, capsys, layout):
    sbs, streams, pair = fisheye_files
    inp = {"sbs": sbs, "streams": streams, "pair": pair}[layout]
    assert cli.main([str(inp), "--audio", str(song), "--no-intro", "--dry-run"]) == 0
    shots = json.loads(capsys.readouterr().out)["shots"]
    assert {s["view"]["lenses"] for s in shots} == {layout}
    assert len({s["source"] for s in shots}) == 1         # a _10_ file is never its own clip
    assert all(abs(s["view"]["yaw"] - 135) < 25 for s in shots)
    out = tmp_path / f"{layout}.mp4"
    assert cli.main([str(inp), "--audio", str(song), "--no-intro", "--preview", "-o", str(out)]) == 0
    assert (probe(out).width, probe(out).height) == (1080, 1920)
    assert white_share(out) > 0.005


def test_fit_track_follows_action(tmp_path, song):
    v = tmp_path / "wide.mp4"
    moving_box(v, (1280, 720), 1120, 360)
    crop, track = tmp_path / "crop.mp4", tmp_path / "track.mp4"
    assert cli.main([str(v), "--audio", str(song), "--no-intro", "--preview", "-o", str(crop)]) == 0
    assert cli.main([str(v), "--audio", str(song), "--no-intro", "--preview", "--fit", "track",
                     "-o", str(track)]) == 0
    assert white_share(crop) < 0.001 and white_share(track) > 0.005
