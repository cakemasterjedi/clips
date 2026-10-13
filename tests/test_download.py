import sys
import types

import pytest

from clips import style
from clips.ffmpeg import ClipsError


def fake_ytdlp(monkeypatch, tmp_path, fail_plain=True, fail_all=False):
    calls = []

    class YDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download):
            calls.append(self.opts)
            if fail_all or (fail_plain and "impersonate" not in self.opts):
                raise Exception("ERROR: [TikTok] 123: Unexpected response from webpage request")
            (tmp_path / "TikTok-123.mp4").write_bytes(b"x")
            return {"id": "123"}

        def prepare_filename(self, info):
            return str(tmp_path / "TikTok-123.mp4")

    mod = types.ModuleType("yt_dlp")
    mod.YoutubeDL = YDL
    mod.version = types.SimpleNamespace(__version__="test")
    monkeypatch.setitem(sys.modules, "yt_dlp", mod)
    monkeypatch.setattr(style, "impersonate_target", lambda: "chrome")
    return calls


def test_retry_as_chrome(monkeypatch, tmp_path):
    calls = fake_ytdlp(monkeypatch, tmp_path)
    p = style.download("https://www.tiktok.com/t/x/", tmp_path, cookies="firefox")
    assert p.name == "TikTok-123.mp4" and len(calls) == 2
    assert calls[1]["impersonate"] == "chrome" and calls[0]["cookiesfrombrowser"] == ("firefox",)


def test_blocked_hint(monkeypatch, tmp_path):
    fake_ytdlp(monkeypatch, tmp_path, fail_all=True)
    with pytest.raises(ClipsError, match="cookies-from-browser"):
        style.download("https://www.tiktok.com/t/x/", tmp_path)
