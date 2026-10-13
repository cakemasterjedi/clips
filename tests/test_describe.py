import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from clips import describe as D
from clips import subject as S


def det(x, y, s=10.0, score=0.9, cls=0):
    return S.Det(x, y, s, 2 * s, score, cls, [1.0 / 32] * 32)


def two_people_frames():
    # person A: big, left; person B: smaller, right
    return [[det(-60, 0, 14), det(60, 0, 8)] for _ in range(6)]


class FakeOllama(BaseHTTPRequestHandler):
    answers = []
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append(body)
        if body["model"] == "missing":
            self.send_response(404)
            self.end_headers()
            return
        out = {"message": {"role": "assistant", "content": FakeOllama.answers.pop(0)}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def ollama(monkeypatch):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeOllama)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{srv.server_address[1]}")
    FakeOllama.answers, FakeOllama.seen = [], []
    yield FakeOllama
    srv.shutdown()


def test_host_defaults_and_normalises(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert D.host() == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert D.host() == "http://127.0.0.1:11434"
    monkeypatch.setenv("OLLAMA_HOST", "http://box:1234/")
    assert D.host() == "http://box:1234"


def test_score_parses_json_and_loose_text(ollama):
    ollama.answers = ['{"match": 8}', "I'd say 3 out of 10", '{"match": 42}']
    assert D.score(b"jpg", "a dog", "m") == 8
    assert D.score(b"jpg", "a dog", "m") == 3
    assert D.score(b"jpg", "a dog", "m") == 10
    req = ollama.seen[0]
    assert req["model"] == "m" and req["stream"] is False and req["format"] == D.FORMAT
    assert "a dog" in req["messages"][0]["content"] and req["messages"][0]["images"]


def test_pick_best_and_fallbacks(ollama, monkeypatch, capsys):
    ollama.answers = ['{"match": 2}', '{"match": 9}', '{"match": 6}']
    assert D.pick([b"a", b"b", b"c"], "x", "m") == (1, [2, 9, 6])
    ollama.answers = ['{"match": 2}', '{"match": 4}']
    assert D.pick([b"a", b"b"], "x", "m") == (None, [2, 4])
    assert "no subject looks like" in capsys.readouterr().err
    assert D.pick([b"a"], "x", "missing")[0] is None
    assert "ollama pull missing" in capsys.readouterr().err
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:9")          # nothing listens there
    assert D.pick([b"a"], "x", "m") == (None, [])
    assert "not reachable" in capsys.readouterr().err


def test_chooser_overrides_prominence_and_caches(ollama, tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPS_CACHE", str(tmp_path / "cache"))
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(D, "close_ups", lambda path, view, tracks, size=(0, 0): [b"img"] * len(tracks))
    opts = S.Options("person", desc="the one on the right")
    view = S.RF.View("equirect")
    frames = two_people_frames()
    assert S.subject_path(frames, "person", True)[0][0] == pytest.approx(-60)   # prominence
    ollama.answers = ['{"match": 1}', '{"match": 9}']
    ch = D.chooser(clip, view, 3.0, opts)
    assert S.subject_path(frames, "person", True, choose=ch)[0][0] == pytest.approx(60)
    # second run: cached, no request
    n = len(ollama.seen)
    assert S.subject_path(frames, "person", True, choose=D.chooser(clip, view, 3.0, opts))[0][0] \
        == pytest.approx(60)
    assert len(ollama.seen) == n


def test_view_crop_is_centred_on_the_detection():
    eq = np.zeros((90, 180, 3), np.uint8)
    eq[38:42, 138:142] = 255                                # yaw 100, pitch 10
    crop = D.view_crop(eq, S.Det(100.0, 10.0, 20.0, 20.0, 0.9, 0), size=64)
    v, u = np.argwhere(crop[..., 0] > 0).mean(axis=0)
    assert abs(u - 31.5) < 3 and abs(v - 31.5) < 3


def test_flat_crop_adds_margin_and_clamps():
    img = np.zeros((100, 200, 3), np.uint8)
    c = D.flat_crop(img, S.Det(0.5, 0.5, 0.2, 0.4, 0.9, 0))
    assert c.shape[:2] == (60, 60)
    c = D.flat_crop(img, S.Det(0.02, 0.5, 0.2, 0.4, 0.9, 0))
    assert c.shape[1] < 60


def test_options_from_desc_implies_auto():
    class A:
        subject, subject_desc, detector, device, ollama_model = None, "red jacket", "s", "auto", None
    o = S.options_from(A())
    assert o.what == "auto" and o.desc == "red jacket" and o.vlm == "qwen2.5vl"


def test_jpeg_roundtrip():
    img = np.full((31, 41, 3), 128, np.uint8)
    assert D.jpeg(img)[:2] == b"\xff\xd8"
