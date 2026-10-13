import numpy as np
import pytest
import soundfile as sf

from clips import cli
from clips import gpu as G
from clips import render as R
from clips import ui as U
from clips.ffmpeg import ClipsError, probe
from clips.sources import Assignment

from conftest import SR, clicks, make_video

GPU = G.Gpu("/dev/dri/renderD128")


def test_upload_swaps_the_tail():
    g = "[0:v]scale=2:2[f];[f]setsar=1,format=yuv420p[v]"
    assert G.upload(g, None) == g
    assert G.upload(g, GPU).endswith("setsar=1,format=nv12,hwupload[v]")
    with pytest.raises(ValueError):
        G.upload("[0:v]null[v]", GPU)


def test_segment_args_on_gpu():
    cpu = R.segment_args(R.Path("a.mp4"), Assignment(0, 1.0), 30, 30, R.Path("o.mp4"))
    assert "libx264" in cpu and "h264_vaapi" not in cpu and "-hwaccel" not in cpu
    a = R.segment_args(R.Path("a.mp4"), Assignment(0, 1.0), 30, 30, R.Path("o.mp4"),
                       punch=True, gpu=GPU)
    assert a[1:5] == ["-init_hw_device", "vaapi=va:/dev/dri/renderD128", "-filter_hw_device", "va"]
    assert a.index("-hwaccel") < a.index("-i") and "libx264" not in a
    assert a[a.index("-c:v") + 1] == "h264_vaapi" and a[a.index("-qp") + 1] == "18"
    assert a[a.index("-filter_complex") + 1].endswith("format=nv12,hwupload[v]")


def test_final_args_on_gpu():
    f = R.final_args(R.Path("l.txt"), R.Path("s.wav"), 0.0, 6.0, 30, None, R.Path("o.mp4"),
                     gpu=GPU)
    assert "h264_vaapi" in f and "-pix_fmt" not in f and "-hwaccel" not in f
    assert "hwupload[v]" in f[f.index("-filter_complex") + 1]
    ia = R.IntroAudio([(R.Path("v.mp4"), 1.0, 3.0)], "rise", 3.0)
    f = R.final_args_intro(R.Path("l.txt"), R.Path("s.wav"), 10.0, 20.0, 30, None,
                           R.Path("o.mp4"), ia, 3.0, gpu=GPU)
    assert "h264_vaapi" in f and "hwupload[v]" in f[f.index("-filter_complex") + 1]


def test_resolve_modes(monkeypatch, capsys):
    assert G.resolve("off") is None
    monkeypatch.setattr(G, "detect", lambda: (None, "no GPU render device"))
    assert G.resolve("auto") is None
    assert "no GPU render device" in capsys.readouterr().err
    with pytest.raises(ClipsError, match="--gpu vaapi: no GPU render device"):
        G.resolve("vaapi")
    monkeypatch.setattr(G, "detect", lambda: (GPU, "VAAPI on x"))
    assert G.resolve("auto") == GPU


def test_detect_reasons(monkeypatch):
    monkeypatch.setattr(G, "_encoders", lambda: " V....D libx264  H.264\n")
    assert "no h264_vaapi" in G.detect()[1]
    monkeypatch.setattr(G, "_encoders", lambda: " V....D h264_vaapi  H.264\n")
    monkeypatch.setattr(G, "render_nodes", lambda: [])
    assert G.detect() == (None, "no GPU render device (/dev/dri/renderD*) found")
    monkeypatch.setattr(G, "render_nodes", lambda: ["/nonexistent/renderD128"])
    assert "does not exist" in G.detect()[1]


def test_render_device_env(monkeypatch):
    monkeypatch.setenv(G.DEVICE_ENV, "/dev/dri/renderD129")
    assert G.render_nodes() == ["/dev/dri/renderD129"]


def _song(tmp_path):
    t = np.arange(0.25, 4, 0.5)
    song = tmp_path / "s.wav"
    sf.write(song, clicks(t, np.full(t.size, 0.8), 4), SR)
    return song


def test_broken_gpu_falls_back_to_cpu(tmp_path, monkeypatch, capsys):
    """auto: a GPU that fails mid-render (here a device that does not exist) -> CPU, same spec."""
    v = tmp_path / "v.mp4"
    make_video(v, "640x360", 30, 5)
    monkeypatch.setattr(G, "resolve", lambda mode: G.Gpu("/dev/dri/renderD999"))
    out = tmp_path / "o.mp4"
    assert cli.main([str(v), "--audio", str(_song(tmp_path)), "--no-intro", "--preview",
                     "--no-loudnorm", "-o", str(out)]) == 0
    err = capsys.readouterr().err
    assert err.count("gpu: encode failed") == 1 and "continuing on the CPU" in err
    i = probe(out)
    assert (i.width, i.height) == (1080, 1920) and i.fps == pytest.approx(30, abs=0.01)


def test_required_gpu_does_not_fall_back(tmp_path, monkeypatch):
    v = tmp_path / "v.mp4"
    make_video(v, "640x360", 30, 5)
    monkeypatch.setattr(G, "resolve", lambda mode: G.Gpu("/dev/dri/renderD999"))
    assert cli.main([str(v), "--audio", str(_song(tmp_path)), "--no-intro", "--preview",
                     "--gpu", "vaapi", "-o", str(tmp_path / "o.mp4")]) != 0


def test_ui_gpu_choice(tmp_path):
    v = tmp_path / "v.mp4"
    v.write_bytes(b"")
    song = tmp_path / "s.wav"
    song.write_bytes(b"")
    argv = U.render_argv({"input": str(v), "audio": str(song), "gpu": "off"}, tmp_path)
    assert argv[argv.index("--gpu") + 1] == "off"
    assert "--gpu" not in U.render_argv({"input": str(v), "audio": str(song)}, tmp_path)
