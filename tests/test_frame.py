import base64
import http.client
import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from clips import cli
from clips import frame as FR
from clips import reframe as RF
from clips.ffmpeg import ClipsError, ffmpeg_exe, probe
from test_reframe import ff, moving_box


@pytest.fixture(scope="module")
def pano_dir(tmp_path_factory):
    """A still textured equirect clip (testsrc2 frame held 3 s) plus an sbs .insv of it."""
    d = tmp_path_factory.mktemp("frame")
    png = d / "f.png"
    ff("-f", "lavfi", "-i", "testsrc2=size=1920x960:rate=1:duration=1", "-frames:v", "1", png)
    eq = d / "pano.mp4"
    ff("-loop", "1", "-i", png, "-t", "3", "-r", "30", "-c:v", "libx264", "-crf", "10",
       "-pix_fmt", "yuv420p", eq)
    insv = d / "VID_20261013_130000_00_001.insv"
    ff("-i", eq, "-vf", f"v360=input=e:output=dfisheye:h_fov={RF.FISHEYE_FOV}:"
       f"v_fov={RF.FISHEYE_FOV}:w=1920:h=960", "-c:v", "libx264", "-pix_fmt", "yuv420p",
       "-f", "mp4", insv)
    return d


@pytest.fixture()
def server(pano_dir, tmp_path):
    clip = tmp_path / "pano.mp4"
    shutil.copy(pano_dir / "pano.mp4", clip)
    srv = FR.serve(clip, "equirect", port=0, browser=False, cache=tmp_path / "cache", width=960)
    yield srv, clip
    srv.shutdown()
    srv.server_close()


def req(srv, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    out = r.status, dict(r.getheaders()), r.read()
    c.close()
    return out


# ------------------------------------------------------------------ proxy
def test_proxy_unwraps_insv_and_is_cached(pano_dir, tmp_path):
    clip = pano_dir / "VID_20261013_130000_00_001.insv"
    view = RF.detect_view(clip, probe(clip))
    assert view.kind == "dfisheye"
    p = FR.make_proxy(clip, view, tmp_path, width=960)
    info = probe(p)
    assert (info.width, info.height) == (960, 480) and abs(info.duration - 3) < 0.2
    mtime = p.stat().st_mtime_ns
    assert FR.make_proxy(clip, view, tmp_path, width=960) == p and p.stat().st_mtime_ns == mtime


def test_frame_rejects_flat_and_second_lens(tmp_path):
    flat = tmp_path / "flat.mp4"
    moving_box(flat, (640, 360), 320, 180, secs=1)
    with pytest.raises(ClipsError, match="360"):
        FR.serve(flat, browser=False, cache=tmp_path)
    (tmp_path / "VID_1_00_001.insv").write_bytes(b"")
    (tmp_path / "VID_1_10_001.insv").write_bytes(b"")
    with pytest.raises(ClipsError, match="second lens"):
        FR.serve(tmp_path / "VID_1_10_001.insv", browser=False, cache=tmp_path)


# ------------------------------------------------------------------ server
def test_server_ranges_meta_and_save(server):
    srv, clip = server
    size = srv.proxy.stat().st_size
    st, h, body = req(srv, "GET", "/video", headers={"Range": "bytes=10-109"})
    assert st == 206 and len(body) == 100 and h["Content-Range"] == f"bytes 10-109/{size}"
    assert body == srv.proxy.read_bytes()[10:110]
    st, h, body = req(srv, "GET", "/video", headers={"Range": "bytes=-50"})
    assert st == 206 and len(body) == 50
    st, _, _ = req(srv, "GET", "/video", headers={"Range": f"bytes={size + 5}-"})
    assert st == 416
    st, _, body = req(srv, "GET", "/meta")
    meta = json.loads(body)
    assert meta["name"] == "pano.mp4" and meta["kind"] == "equirect" and abs(meta["duration"] - 3) < 0.1
    assert json.loads(req(srv, "GET", "/views")[2]) == {"version": 1, "keyframes": []}
    st, _, body = req(srv, "GET", "/")
    assert st == 200 and b"clips frame" in body

    data = json.dumps({"version": 1, "keyframes": [{"t": 1, "yaw": 40, "pitch": 5, "fov": 50,
                                                    "roll": 0}, {"t": 0, "yaw": -20}]})
    hdr = {"Content-Type": "application/json"}
    assert req(srv, "POST", "/views", data, hdr)[0] == 403                 # no X-Clips header
    assert req(srv, "POST", "/views", data, {**hdr, "X-Clips": "1", "Host": "evil.example"})[0] == 403
    assert req(srv, "GET", "/views", headers={"Host": "evil.example:8360"})[0] == 403
    st, _, body = req(srv, "POST", "/views", '{"version": 1, "keyframes": [{"t": 0, "fov": 0}]}',
                      {**hdr, "X-Clips": "1"})
    assert st == 400 and not RF.sidecar_path(clip).exists()
    st, _, body = req(srv, "POST", "/views", data, {**hdr, "X-Clips": "1"})
    assert st == 200 and json.loads(body)["keyframes"] == 2
    keys = RF.load_keyframes(clip)
    assert [k.t for k in keys] == [0, 1] and keys[1].yaw == 40
    assert json.loads(req(srv, "GET", "/views")[2])["keyframes"][1]["fov"] == 50


def test_cli_frame_parser():
    a = cli.build_frame_parser().parse_args(["x.insv", "--no-browser", "--port", "0"])
    assert a.clip == "x.insv" and a.no_browser and a.port == 0 and a.mode360 == "auto"


# ------------------------------------------------------------------ browser
def gray(png_bytes, size):
    raw = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", "-", "-vf",
                          f"scale={size[0]}:{size[1]},format=gray", "-f", "rawvideo", "-"],
                         input=png_bytes, stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(size[1], size[0]).astype(float)


def v360_frame(proxy, yaw, pitch, fov, roll, size):
    v = RF.View("equirect", yaw=yaw, pitch=pitch, fov=fov, roll=roll)
    g = RF.view_graph(v, 30, "0:v", None, "f", size)
    out = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-ss", "1", "-i", str(proxy),
                          "-filter_complex", g, "-map", "[f]", "-frames:v", "1", "-f", "image2pipe",
                          "-c:v", "png", "-"], stdout=subprocess.PIPE, check=True).stdout
    return gray(out, size)


@pytest.fixture()
def page(server):
    sync = pytest.importorskip("playwright.sync_api")
    exe = os.environ.get("CLIPS_CHROMIUM", "/opt/pw-browsers/chromium")
    kw = {"executable_path": exe} if os.path.exists(exe) else {}   # else Playwright's own
    with sync.sync_playwright() as p:
        try:
            b = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader",
                                        "--enable-unsafe-swiftshader"], **kw)
        except Exception as exc:                     # no browser installed
            pytest.skip(f"no chromium: {exc}")
        pg = b.new_page(viewport={"width": 1100, "height": 900})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto(server[0].url)
        pg.wait_for_selector("body[data-ready='1']", timeout=20000)
        yield pg, server, errors
        b.close()


def test_browser_view_matches_render(page):
    """The page shows what a render with the same yaw/pitch/fov/roll shows."""
    pg, (srv, clip), errors = page
    size = (270, 480)
    for yaw, pitch, fov, roll in [(0, 0, 60, 0), (120, 25, 80, 0), (-60, -30, 45, 20)]:
        pg.evaluate("""([y, p, f, r]) => new Promise((ok) => {
            const c = window.clipsFrame; c.S.follow = false; c.video.pause();
            c.video.addEventListener('seeked', () => { c.setView(y, p, f, r);
              requestAnimationFrame(() => requestAnimationFrame(ok)); }, {once: true});
            c.seek(1.0); })""", [yaw, pitch, fov, roll])
        url = pg.evaluate("document.getElementById('view').toDataURL('image/png')")
        got = gray(base64.b64decode(url.split(",", 1)[1]), size)
        want = v360_frame(srv.proxy, yaw, pitch, fov, roll, size)
        assert np.abs(got - want)[10:-10, 10:-10].mean() < 8, (yaw, pitch, fov, roll)
    other = v360_frame(srv.proxy, 0, 0, 60, 0, size)
    assert np.abs(got - other).mean() > 20                     # the comparison can fail
    assert not errors


def test_browser_keyframes_and_save(page):
    pg, (srv, clip), errors = page
    pg.click("#view")
    pg.evaluate("window.clipsFrame.seek(0.5)")
    pg.wait_for_function("Math.abs(window.clipsFrame.video.currentTime - 0.5) < 0.05")
    pg.mouse.move(300, 400)
    pg.mouse.down()
    pg.mouse.move(200, 380, steps=5)                         # drag: turn right, tilt up
    pg.mouse.up()
    s = pg.evaluate("({...window.clipsFrame.S, keys: null})")
    assert s["yaw"] > 5 and s["pitch"] < 0 and not s["follow"]
    pg.keyboard.press("e")                                   # roll +1
    pg.keyboard.press("k")
    pg.evaluate("window.clipsFrame.seek(2.0)")
    pg.wait_for_function("Math.abs(window.clipsFrame.video.currentTime - 2.0) < 0.05")
    pg.evaluate("window.clipsFrame.S.follow = false; window.clipsFrame.setView(-90, 10, 40, 0)")
    pg.keyboard.press("k")
    assert pg.locator(".mark").count() == 2
    pg.keyboard.press("Control+s")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('saved')")
    keys = RF.load_keyframes(clip)
    assert len(keys) == 2 and keys[0].t == pytest.approx(0.5, abs=0.05)
    assert keys[0].yaw == pytest.approx(s["yaw"]) and keys[0].roll == 1
    assert (keys[1].yaw, keys[1].pitch, keys[1].fov) == (-90, 10, 40)
    # the page follows the same easing as renders
    t = 1.3
    js = pg.evaluate(f"window.clipsFrame.interp(window.clipsFrame.S.keys, {t})")
    assert js == pytest.approx(list(RF.interpolate(keys, t)), abs=0.01)
    pg.keyboard.press("Delete")                              # removes the keyframe at 2.0
    assert pg.locator(".mark").count() == 1
    assert not errors
