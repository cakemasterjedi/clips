"""`clips frame CLIP`: a local browser tool to frame a 360 clip by hand.

The clip is unwrapped once to an equirectangular VP9 WebM proxy (any layout: equirect, .insv
side by side, two streams or a _00_/_10_ pair), cached under ~/.cache/clips/proxies. A small
HTTP server on 127.0.0.1 serves the page (WebGL, no libraries), the proxy (with Range
requests, so the browser can seek) and the clip's keyframe file, which the page saves next to
the clip as `<clip>.view.json`. Renders pick that file up on their own (see reframe.py).

The server only listens on localhost, only answers requests whose Host is localhost, and
only accepts a save that carries the X-Clips header (a cross-site page cannot send it without
a CORS preflight, which is never granted).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from .ffmpeg import ClipsError, probe, run_ffmpeg
from . import reframe as RF

UI = Path(__file__).with_name("frame_ui.html")
PROXY_W = 1920
PROXY_GOP = 15            # a keyframe every 0.5 s at 30 fps keeps scrubbing snappy
MAX_BODY = 1 << 20


def cache_dir(cache: Optional[Path] = None) -> Path:
    return Path(cache) if cache else Path.home() / ".cache" / "clips" / "proxies"


def proxy_args(src: Path, view: RF.View, out: Path, width: int = PROXY_W,
               audio: bool = True) -> list:
    """FFmpeg args: the whole clip as a width x width/2 equirect VP9 WebM (+ Opus audio)."""
    w, h = width, width // 2
    inp, inp2 = RF.input_labels(view, 0)
    pre, opts = RF.unwrap_graph(view, inp, inp2, "u").split("|")
    conv = f"scale={w}:{h}" if view.kind == "equirect" else \
        f"v360={opts}:output=e:w={w}:h={h}:interp=linear"
    args = ["-y", "-i", str(src)]
    if view.kind == "dfisheye" and view.lenses == "pair" and view.pair is not None:
        args += ["-i", str(view.pair)]
    args += ["-filter_complex", f"{pre};[u]{conv},format=yuv420p[v]", "-map", "[v]"]
    if audio:
        args += ["-map", "0:a:0?", "-c:a", "libopus", "-b:a", "96k"]
    args += ["-c:v", "libvpx-vp9", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
             "-b:v", "0", "-crf", "34", "-g", str(PROXY_GOP), "-f", "webm", str(out)]
    return args


def make_proxy(src: Path, view: RF.View, cache: Optional[Path] = None,
               width: int = PROXY_W) -> Path:
    """Cached proxy path, building it if needed (keyed by path, size, mtime and layout)."""
    src = Path(src).resolve()
    st = src.stat()
    key = f"{src}|{st.st_size}|{st.st_mtime_ns}|{view.kind}|{view.lenses}|{width}"
    if view.pair is not None:
        ps = Path(view.pair).stat()
        key += f"|{ps.st_size}|{ps.st_mtime_ns}"
    d = cache_dir(cache)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', src.stem)}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.webm"
    if out.exists() and out.stat().st_size > 0:
        return out
    tmp = out.with_suffix(".part.webm")
    print(f"building framing proxy for {src.name}...", file=sys.stderr)
    run_ffmpeg(proxy_args(src, view, tmp, width))
    os.replace(tmp, out)
    return out


def save_views(clip: Path, data) -> int:
    """Validate and write the clip's keyframe file (atomically). Returns the keyframe count."""
    keys = RF.parse_keyframes(data)
    side = RF.sidecar_path(clip)
    tmp = side.with_name(side.name + ".tmp")
    tmp.write_text(json.dumps(RF.keyframes_json(keys), indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, side)
    return len(keys)


def current_views(clip: Path) -> dict:
    keys = RF.load_keyframes(clip) or []
    return RF.keyframes_json(keys)


def send_file(h, f: Path, ctype: str, extra: Optional[dict] = None):
    """Serve a file with Range support (the browser seeks videos with byte ranges)."""
    size = f.stat().st_size
    rng = h.headers.get("Range")
    start, end, code = 0, size - 1, 200
    m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
    if m and (m.group(1) or m.group(2)):
        if m.group(1):
            start = int(m.group(1))
            end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
        else:                                         # suffix range: last N bytes
            start = max(0, size - int(m.group(2)))
        if start > end or start >= size:
            return h._send(416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
        code = 206
    h.send_response(code)
    h.send_header("Content-Type", ctype)
    h.send_header("Accept-Ranges", "bytes")
    h.send_header("Content-Length", str(end - start + 1))
    if code == 206:
        h.send_header("Content-Range", f"bytes {start}-{end}/{size}")
    for k, v in (extra or {}).items():
        h.send_header(k, v)
    h.end_headers()
    if h.command == "HEAD":
        return
    with open(f, "rb") as fh:
        fh.seek(start)
        left = end - start + 1
        try:
            while left > 0:
                chunk = fh.read(min(1 << 16, left))
                if not chunk:
                    break
                h.wfile.write(chunk)
                left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass                                      # the browser seeks by hanging up


class FrameServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, clip: Path, proxy: Path, meta: dict, port: int = 0):
        self.clip, self.proxy, self.meta = Path(clip), Path(proxy), meta
        super().__init__(("127.0.0.1", port), Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"


class Handler(BaseHTTPRequestHandler):
    server: FrameServer

    def log_message(self, fmt, *args):      # quiet
        pass

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _send(self, code: int, body: bytes, ctype: str, extra: Optional[dict] = None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code: int, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._send(403, b"forbidden", "text/plain")
        path = self.path.split("?", 1)[0]
        if path == "/":
            return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
        if path == "/meta":
            return self._json(200, self.server.meta)
        if path == "/views":
            try:
                return self._json(200, current_views(self.server.clip))
            except ClipsError as exc:
                return self._json(200, {"version": 1, "keyframes": [], "error": str(exc)})
        if path == "/video":
            return self._video()
        self._send(404, b"not found", "text/plain")

    def _video(self):
        send_file(self, self.server.proxy, "video/webm")

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Clips") != "1":
            return self._send(403, b"forbidden", "text/plain")
        if self.path.split("?", 1)[0] != "/views":
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._json(400, {"error": "bad body"})
        try:
            data = json.loads(self.rfile.read(n))
            count = save_views(self.server.clip, data)
        except (ValueError, ClipsError) as exc:
            return self._json(400, {"error": str(exc)})
        self._json(200, {"ok": True, "keyframes": count,
                         "file": RF.sidecar_path(self.server.clip).name})


def clip_meta(clip: Path, info, view: RF.View) -> dict:
    return {"name": clip.name, "duration": info.duration, "fps": getattr(info, "fps", None) or 30,
            "kind": view.kind, "lenses": view.lenses, "sidecar": RF.sidecar_path(clip).name,
            "fov": RF.H_FOV}


def serve(clip: Path, mode360: str = "auto", port: int = 8360, browser: bool = True,
          cache: Optional[Path] = None, width: int = PROXY_W,
          ready: Optional[threading.Event] = None) -> FrameServer:
    """Build the proxy and start serving (in a background thread). Returns the server."""
    clip = Path(clip)
    if not clip.is_file():
        raise ClipsError(f"clip not found: {clip}")
    if RF.is_pair_secondary(clip):
        raise ClipsError(f"{clip.name} is the second lens of a pair; open the _00_ file")
    info = probe(clip)
    view = RF.detect_view(clip, info, mode360)
    if not view.is360:
        raise ClipsError(f"{clip.name} does not look like 360 footage "
                         "(use --360 on, or --360 equirect / dfisheye to force a layout)")
    proxy = make_proxy(clip, view, cache, width)
    try:
        srv = FrameServer(clip, proxy, clip_meta(clip, info, view), port)
    except OSError:
        srv = FrameServer(clip, proxy, clip_meta(clip, info, view), 0)   # port taken
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if browser:
        webbrowser.open(srv.url)
    if ready:
        ready.set()
    return srv


def run_frame(args) -> int:
    srv = serve(Path(args.clip), args.mode360, args.port, not args.no_browser, args.cache,
                args.proxy_width)
    print(f"framing {Path(args.clip).name} at {srv.url}  (keyframes save to "
          f"{RF.sidecar_path(Path(args.clip)).name}; Ctrl+C to stop)", file=sys.stderr)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
    return 0
