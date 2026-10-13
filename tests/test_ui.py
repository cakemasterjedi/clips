import http.client
import json
import sys
import time

import numpy as np
import pytest
import soundfile as sf

from clips import ui
from clips.ffmpeg import ClipsError, probe
from conftest import SR, clicks, make_video


@pytest.fixture
def media(tmp_path):
    d = tmp_path / "footage"
    d.mkdir()
    make_video(d / "a.mp4", "1280x720", 30, 3)
    song = tmp_path / "song.wav"
    times = np.arange(0.25, 4, 0.5)
    sf.write(song, clicks(times, np.full(times.size, 0.8), 4), SR)
    return d, song


def test_render_argv(tmp_path, media):
    d, song = media
    r = tmp_path / "renders"
    argv = ui.render_argv({"input": str(d), "audio": str(song), "name": "my edit!.mp4",
                           "duration": "12", "audio_start": "auto", "smart_pick": True,
                           "loop": False, "variations": "1", "subject": "person",
                           "subject_desc": "the red jacket", "extra": "--music-in drop"}, r)
    assert argv[:5] == [str(d), "--audio", str(song), "-o", str(r / "my-edit.mp4")]
    assert argv[argv.index("--duration") + 1] == "12"
    assert argv[argv.index("--subject-desc") + 1] == "the red jacket"
    assert "--smart-pick" in argv and "--loop" not in argv and "--variations" not in argv
    assert argv[-2:] == ["--music-in", "drop"]
    with pytest.raises(ClipsError, match="song"):
        ui.render_argv({"input": str(d)}, r)
    with pytest.raises(ClipsError, match="footage"):
        ui.render_argv({"input": str(tmp_path / "nope"), "audio": str(song)}, r)
    with pytest.raises(ClipsError, match="invalid choice"):
        ui.render_argv({"input": str(d), "audio": str(song), "fit": "stretch"}, r)
    with pytest.raises(ClipsError, match="name box"):
        ui.render_argv({"input": str(d), "audio": str(song), "extra": "-o /etc/x.mp4"}, r)


def test_render_argv_sound_captions_options(tmp_path, media):
    d, song = media
    r = tmp_path / "renders"
    argv = ui.render_argv({"input": str(d), "music": "clips", "audio": str(song), "zoom": "0.8",
                           "bpm": "100", "audio_start": "auto", "captions": "none",
                           "subs": "x.srt", "options": [["--music-in", "drop"], ["--360", "off"],
                                                        ["--no-intro", True], ["bad", 1]]}, r)
    assert argv[1] == "--no-music" and "--audio" not in argv and "--audio-start" not in argv
    assert "--subs" not in argv and argv[argv.index("--zoom") + 1] == "0.8"
    assert argv[argv.index("--bpm") + 1] == "100"
    assert argv[-5:] == ["--music-in", "drop", "--360", "off", "--no-intro"]
    assert ui.render_argv({"input": str(d), "music": "mute"}, r)[1] == "--mute"
    with pytest.raises(ClipsError, match="caption"):
        ui.render_argv({"input": str(d), "audio": str(song), "captions": "file"}, r)
    flags = {o["flag"]: o for o in ui.option_list()}
    assert "--music-in" in flags and flags["--music-in"]["choices"] == ["rise", "drop"]
    assert "--audio" not in flags and "--zoom" not in flags and flags["--intro"]["repeat"]


def test_upload_and_mkdir(tmp_path):
    srv = ui.serve(tmp_path / "work", 0, False)
    try:
        d = tmp_path / "up"
        d.mkdir()
        assert json.loads(req(srv, "POST", "/api/mkdir", {"path": str(d), "name": "new"})[1])["ok"]
        assert (d / "new").is_dir()
        assert req(srv, "POST", "/api/mkdir", {"path": str(d), "name": "../x"})[0] == 400

        def up(name, data, folder=d):
            c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
            c.request("POST", f"/api/upload?dir={folder}&name={name}", data, {"X-Clips": "1"})
            r = c.getresponse()
            return r.status, json.loads(r.read())
        st, body = up("clip.mp4", b"abc" * 1000)
        assert st == 200 and (d / "clip.mp4").read_bytes() == b"abc" * 1000
        st, body = up("..%2F..%2Fclip.mp4", b"zz")             # path parts are dropped
        assert st == 200 and body["name"] == "clip-1.mp4" and (d / "clip-1.mp4").exists()
        assert up("x.mp4", b"1", tmp_path / "missing")[0] == 400
        assert not any(p.name.endswith(".part") for p in d.iterdir())
    finally:
        srv.shutdown()


def test_stage_and_cpu():
    assert ui.stage_of("shot 3/9") == "rendering shot 3/9"
    assert ui.stage_of("final pass...").startswith("final pass")
    assert ui.stage_of("[12:00:00] ffmpeg start: a -> b") is None
    import os
    cpu = ui.group_cpu(os.getpgid(0))
    assert cpu is None or cpu > 0


def test_learn_and_frame_argv(tmp_path):
    p = tmp_path / "profiles"
    assert ui.learn_argv({"links": "https://a/1\n https://a/2 ", "name": "Cool Creator"}, p) == [
        "learn", "https://a/1", "https://a/2", "-o", str(p / "Cool-Creator.json"),
        "--name", "Cool-Creator"]
    with pytest.raises(ClipsError):
        ui.learn_argv({"links": "", "name": "x"}, p)
    with pytest.raises(ClipsError):
        ui.frame_argv({"clip": str(tmp_path / "missing.insv")})


def test_progress():
    k, f = ui.progress_of("rendering 4 shots with 2 workers...", 0, 1, 0.0)
    assert (k, f) == (0, 0.0)
    k, f = ui.progress_of("shot 2/4", k, 1, f)
    assert f == pytest.approx(0.1 + 0.75 / 2)
    assert ui.progress_of("final pass...", k, 1, f)[1] > f
    assert ui.progress_of("variation 2/3: x_v2.mp4", 0, 3, 0.97) == (1, 0.0)


def test_list_dir(tmp_path, media):
    d, song = media
    (d / "notes.txt").write_text("x")
    (d / "sub").mkdir()
    r = ui.list_dir(str(d), "video")
    assert r["dirs"] == ["sub"] and [f["name"] for f in r["files"]] == ["a.mp4"]
    assert [f["name"] for f in ui.list_dir(str(song), "audio")["files"]] == ["song.wav"]


FAKE = r"""
import sys, time
args = sys.argv[1:]
if "--slow" in args:
    print("rendering 2 shots", file=sys.stderr, flush=True); time.sleep(30)
out = args[args.index("-o") + 1]
for line in ("shot 1/2", "shot 2/2", "final pass..."):
    print(line, file=sys.stderr, flush=True)
open(out, "wb").write(b"x" * 1000)
print("wrote " + out, file=sys.stderr)
"""


def req(srv, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    h = {"X-Clips": "1", "Content-Type": "application/json", **(headers or {})}
    c.request(method, path, json.dumps(body) if body is not None else None, h)
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, data, r


def wait_job(srv, jid, want=("done", "failed", "cancelled"), timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = srv.jobs.get(jid)
        if j.status in want:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {jid} stuck in {j.status}: {list(j.log)}")


def test_server_jobs(tmp_path, media):
    d, song = media
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    srv = ui.serve(tmp_path / "work", 0, False, [sys.executable, str(fake)])
    try:
        assert req(srv, "GET", "/")[0] == 200
        assert req(srv, "GET", "/api/state", headers={"Host": "evil.example"})[0] == 403
        st, _, _ = req(srv, "POST", "/api/render", {"input": str(d), "audio": str(song)},
                       {"X-Clips": ""})
        assert st == 403
        st, body, _ = req(srv, "POST", "/api/render", {"input": str(d)})
        assert st == 400 and "song" in json.loads(body)["error"]

        st, body, _ = req(srv, "POST", "/api/render",
                          {"input": str(d), "audio": str(song), "name": "t1"})
        assert st == 200
        j = wait_job(srv, json.loads(body)["id"])
        assert j.status == "done" and j.progress == 1.0 and "shot 2/2" in [x[2] for x in j.log]
        state = json.loads(req(srv, "GET", "/api/state")[1])
        assert [r["name"] for r in state["renders"]] == ["t1.mp4"]
        assert state["jobs"][0]["outputs"] == ["t1.mp4"]
        st, data, r = req(srv, "GET", "/renders/t1.mp4?dl=1", headers={"Range": "bytes=10-19"})
        assert st == 206 and len(data) == 10 and "attachment" in r.getheader("Content-Disposition")
        assert req(srv, "GET", "/renders/..%2Ffake.py")[0] == 404

        st, body, _ = req(srv, "POST", "/api/render", {"input": str(d), "audio": str(song),
                                                       "name": "slow", "extra": "--slow"})
        # --slow is not a clips flag, so the real parser refuses it
        assert st == 400
    finally:
        srv.shutdown()


def test_cancel(tmp_path, media):
    d, song = media
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE.replace('"--slow" in args', '"--shuffle" in args'))
    srv = ui.serve(tmp_path / "work", 0, False, [sys.executable, str(fake)])
    try:
        a = json.loads(req(srv, "POST", "/api/render", {"input": str(d), "audio": str(song),
                                                        "name": "a", "shuffle": True})[1])["id"]
        b = json.loads(req(srv, "POST", "/api/render", {"input": str(d), "audio": str(song),
                                                        "name": "b"})[1])["id"]
        wait_job(srv, a, ("running",))
        assert srv.jobs.get(b).status == "queued"          # one render at a time
        assert req(srv, "POST", f"/api/jobs/{b}/cancel", {})[0] == 200
        assert srv.jobs.get(b).status == "cancelled"
        assert req(srv, "POST", f"/api/jobs/{a}/cancel", {})[0] == 200
        assert wait_job(srv, a, timeout=10).status == "cancelled"
        assert not (tmp_path / "work" / ".tmp" / str(a)).exists()
    finally:
        srv.shutdown()


def test_real_render(tmp_path, media):
    d, song = media
    srv = ui.serve(tmp_path / "work", 0, False)
    try:
        st, body, _ = req(srv, "POST", "/api/render", {"input": str(d), "audio": str(song),
                                                       "duration": "3", "preview": True,
                                                       "name": "real"})
        assert st == 200, body
        j = wait_job(srv, json.loads(body)["id"], timeout=180)
        assert j.status == "done", list(j.log)
        info = probe(srv.renders / "real.mp4")
        assert (info.width, info.height) == (1080, 1920)
        assert any(x[2].startswith("shot ") for x in j.log)
        assert any("ffmpeg done" in x[2] for x in j.log)          # verbose detail lines
        assert j.logfile.is_file() and "final pass" in j.logfile.read_text()
    finally:
        srv.shutdown()


def test_page_in_browser(tmp_path, media):
    """The page drives a render end to end: pick folder and song, render, see the result."""
    import os
    sync = pytest.importorskip("playwright.sync_api")
    d, song = media
    fake = tmp_path / "fake.py"
    fake.write_text(FAKE)
    srv = ui.serve(tmp_path / "work", 0, False, [sys.executable, str(fake)])
    exe = os.environ.get("CLIPS_CHROMIUM", "/opt/pw-browsers/chromium")
    kw = {"executable_path": exe} if os.path.exists(exe) else {}
    try:
        with sync.sync_playwright() as p:
            b = p.chromium.launch(**kw)
            pg = b.new_page(viewport={"width": 1280, "height": 900})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(srv.url)
            pg.click("[data-pick=input]")
            pg.fill("#input", "")
            pg.evaluate(f"pkOpen({json.dumps(str(d))})")
            pg.wait_for_selector("#pk-list .f")
            pg.click("#pk-ok")
            assert pg.input_value("#input") == str(d)
            pg.click("[data-pick=audio]")
            pg.evaluate(f"pkOpen({json.dumps(str(song.parent))})")
            pg.click("#pk-list .f >> text=song.wav")
            pg.click("#pk-ok")
            assert pg.input_value("#audio") == str(song)
            pg.fill("#name", "fromui")
            pg.click("#render")
            pg.wait_for_selector(".vid video", timeout=20000)
            assert "fromui.mp4" in pg.inner_text("#renders")
            assert "done" in pg.inner_text("#jobs")
            if os.environ.get("CLIPS_UI_SHOT"):
                pg.screenshot(path=os.environ["CLIPS_UI_SHOT"], full_page=True)
            b.close()
        assert not errors
    finally:
        srv.shutdown()


CHATTY = r"""
import sys, time
for i in range(200):
    print(f"[00:00:00] ffmpeg done in 0.1s: seg_{i:04d}.mp4" if i % 2 else f"shot {i}/200",
          file=sys.stderr, flush=True)
    time.sleep(0.04)
"""


def test_page_live_log(tmp_path, media):
    """Song/caption choices, the option picker, and a log that keeps its scroll position."""
    import os
    sync = pytest.importorskip("playwright.sync_api")
    d, song = media
    fake = tmp_path / "chatty.py"
    fake.write_text(CHATTY)
    srv = ui.serve(tmp_path / "work", 0, False, [sys.executable, str(fake)])
    exe = os.environ.get("CLIPS_CHROMIUM", "/opt/pw-browsers/chromium")
    kw = {"executable_path": exe} if os.path.exists(exe) else {}
    try:
        with sync.sync_playwright() as p:
            b = p.chromium.launch(**kw)
            pg = b.new_page(viewport={"width": 1280, "height": 1000})
            errors = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(srv.url)
            pg.fill("#input", str(d))
            pg.select_option("#music", "clips")
            assert not pg.is_visible("#audio") and pg.is_visible("#bpm")
            assert not pg.is_visible("#subs")
            pg.click("#more summary")
            pg.select_option("#opt-add", "--music-in")
            pg.select_option("#opts select", "drop")
            pg.click("#render")
            pg.wait_for_function("document.querySelectorAll('pre.log div').length > 40")
            first = pg.inner_text("pre.log div:first-child")
            assert "--no-music" in first and "--music-in drop" in first
            assert "running" in pg.inner_text(".job .meta")
            pg.evaluate("document.querySelector('pre.log').scrollTop = 0")
            n0 = pg.evaluate("document.querySelectorAll('pre.log div').length")
            pg.wait_for_timeout(2500)
            assert pg.evaluate("document.querySelectorAll('pre.log div').length") > n0
            assert pg.evaluate("document.querySelector('pre.log').scrollTop") < 5   # stayed put
            assert not pg.is_visible("pre.log div.v")             # verbose lines hidden by default
            pg.check("#adv")
            assert pg.is_visible("pre.log div.v")
            if os.environ.get("CLIPS_UI_SHOT"):
                pg.screenshot(path=os.environ["CLIPS_UI_SHOT"], full_page=True)
            b.close()
        assert not errors
    finally:
        srv.shutdown()


def test_lan_host(tmp_path, monkeypatch):
    srv = ui.serve(tmp_path / "work", 0, False, host="0.0.0.0")
    try:
        assert not srv.local_only and srv.key and srv.lan_urls()[0].endswith("?key=" + srv.key)
        assert req(srv, "GET", "/api/state")[0] == 200          # from this PC: no key needed
        monkeypatch.setattr(ui, "is_loopback", lambda h: False)  # requests "from the LAN"
        lan = {"Host": "192.168.1.20:8361"}
        assert req(srv, "GET", "/api/state", headers=lan)[0] == 403
        assert req(srv, "GET", "/?key=wrong", headers=lan)[0] == 403
        st, _, r = req(srv, "GET", "/?key=" + srv.key, headers=lan)
        assert st == 303 and r.getheader("Location") == "/"
        ck = {**lan, "Cookie": r.getheader("Set-Cookie").split(";")[0]}
        assert req(srv, "GET", "/api/state", headers=ck)[0] == 200
        assert req(srv, "POST", "/api/render", {}, ck)[0] == 400
        assert req(srv, "POST", "/api/render", {}, {**ck, "X-Clips": ""})[0] == 403
    finally:
        srv.shutdown()
    monkeypatch.undo()
    srv = ui.serve(tmp_path / "work", 0, False)
    try:
        assert srv.local_only and srv.key is None and srv.lan_urls() == []
        assert req(srv, "GET", "/api/state")[0] == 200
    finally:
        srv.shutdown()


def test_learn_cookies_and_eta(tmp_path):
    argv = ui.learn_argv({"links": "https://a/1 https://a/2", "name": "x", "cookies": "firefox"},
                         tmp_path)
    assert argv[:5] == ["learn", "https://a/1", "https://a/2", "--cookies-from-browser", "firefox"]
    with pytest.raises(ClipsError):
        ui.learn_argv({"links": "https://a/1", "name": "x", "cookies": "fire fox; rm"}, tmp_path)
    j = ui.Job("render", [], "t")
    j.status, j.started = "running", 100.0
    assert j.eta(110.0) is None
    j.p0, j.progress = (100.0, 0.1), 0.5                    # 0.4 done in 20 s -> 0.5 left = 25 s
    assert j.eta(120.0) == pytest.approx(25.0)


def test_update_job(tmp_path, monkeypatch):
    script = tmp_path / "update.sh"
    script.write_text("echo 'updated abc -> def:'\necho '  - something new'\n")
    monkeypatch.setattr(ui, "UPDATER", script)
    srv = ui.serve(tmp_path / "work", 0, False)
    called = []
    srv.jobs.on_updated = lambda: called.append(1)
    try:
        state = json.loads(req(srv, "GET", "/api/state")[1])
        assert state["can_update"] and state["ncpu"] >= 1 and state["boot"]
        st, body, _ = req(srv, "POST", "/api/update", {})
        assert st == 200
        j = wait_job(srv, json.loads(body)["id"])
        assert j.status == "done" and called == [1]
        assert json.loads(req(srv, "GET", "/api/state")[1])["jobs"][0]["command"].startswith("bash ")
        script.write_text("echo 'already up to date'\n")
        j = wait_job(srv, json.loads(req(srv, "POST", "/api/update", {})[1])["id"])
        assert j.status == "done" and called == [1]          # nothing new: no restart
        monkeypatch.setattr(ui, "UPDATER", tmp_path / "missing.sh")
        assert req(srv, "POST", "/api/update", {})[0] == 400
    finally:
        srv.shutdown()
