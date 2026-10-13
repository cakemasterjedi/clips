"""`clips ui`: a local web page for the whole tool.

Pick a footage folder and a song with a server-side file browser (a browser page cannot see
real paths), set the main options, learn a creator style from links, and queue renders. Each
job runs the normal `clips` command in a subprocess; its output is the job log, and the
`shot k/N`, `final pass...` and `wrote ...` lines drive the progress bar. Renders and learned
profiles live in a work folder (default ~/Videos/clips: renders/, profiles/), and finished
renders can be played or downloaded from the page. `clips frame` runs as its own job and the
page links to it.

Same rules as `clips frame`: by default the server listens on 127.0.0.1 only, answers only
requests whose Host is localhost, and every POST must carry the X-Clips header (a cross-site
page cannot send it without a CORS preflight, which is never granted). `--host 0.0.0.0` opens
it to other devices on the network: then any Host is accepted, and anyone who can reach the
port can browse folders and start renders. Renders run one at a time.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import queue
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Optional
from urllib.parse import parse_qs, urlparse

from .ffmpeg import ClipsError
from .frame import send_file

PAGE = Path(__file__).with_name("ui.html")
DEFAULT_WORKDIR = Path.home() / "Videos" / "clips"
MAX_BODY = 1 << 20
LOG_LINES = 400

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".insv"}
AUDIO_EXT = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus"}
SUBS_EXT = {".srt", ".ass"}
KINDS = {"video": VIDEO_EXT, "audio": AUDIO_EXT, "subs": SUBS_EXT, "json": {".json"}}

# checkbox name -> flag
FLAGS = {"smart_pick": "--smart-pick", "loop": "--loop", "ramp": "--ramp", "preview": "--preview",
         "punch": "--punch", "flash": "--flash", "shuffle": "--shuffle",
         "word_reveal": "--word-reveal", "auto_frame": "--auto-frame", "no_intro": "--no-intro",
         "no_loudnorm": "--no-loudnorm"}
# text/select name -> flag (empty = left out)
VALUES = {"duration": "--duration", "fps": "--fps", "fit": "--fit", "variations": "--variations",
          "subject": "--subject", "subject_desc": "--subject-desc", "sub_style": "--sub-style",
          "seed": "--seed", "subs": "--subs", "style": "--style"}


def safe_name(name: str, default: str) -> str:
    """Output file stem: letters, digits, - _ . only, no path."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(str(name or "")).stem).strip("-.")
    return stem or default


def parse_ok(parser: argparse.ArgumentParser, argv: List[str]) -> None:
    """Raise ClipsError with argparse's message instead of exiting."""
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            parser.parse_args(argv)
    except SystemExit:
        msg = err.getvalue().strip().splitlines()
        raise ClipsError(msg[-1].split("error: ", 1)[-1] if msg else "invalid options") from None


def render_argv(form: dict, renders: Path) -> List[str]:
    """`clips` arguments for a render form; validated with the real parser."""
    from .cli import build_parser

    inp, song = str(form.get("input") or "").strip(), str(form.get("audio") or "").strip()
    if not inp or not Path(inp).exists():
        raise ClipsError("pick a footage folder or clip")
    if not song or not Path(song).is_file():
        raise ClipsError("pick a song")
    out = renders / f"{safe_name(form.get('name'), time.strftime('edit-%Y%m%d-%H%M%S'))}.mp4"
    argv = [inp, "--audio", song, "-o", str(out)]
    start = str(form.get("audio_start") or "").strip()
    if start:
        argv += ["--audio-start", start]
    for key, flag in VALUES.items():
        v = form.get(key)
        if v is None or str(v).strip() == "" or (key == "variations" and str(v) == "1"):
            continue
        argv += [flag, str(v).strip()]
    argv += [flag for key, flag in FLAGS.items() if form.get(key)]
    extra = str(form.get("extra") or "").strip()
    if extra:
        try:
            argv += shlex.split(extra)
        except ValueError as exc:
            raise ClipsError(f"extra flags: {exc}") from None
    if any(a in ("-o", "--output") or a.startswith("--output=") for a in argv[5:]):
        raise ClipsError("set the output name in the name box, not with -o")
    parse_ok(build_parser(), argv)
    return argv


def learn_argv(form: dict, profiles: Path) -> List[str]:
    from .cli import build_learn_parser

    refs = [r for r in re.split(r"\s+", str(form.get("links") or "")) if r]
    if not refs:
        raise ClipsError("paste at least one link or video path")
    name = safe_name(form.get("name"), "")
    if not name:
        raise ClipsError("give the style a name")
    argv = ["learn", *refs, "-o", str(profiles / f"{name}.json"), "--name", name]
    parse_ok(build_learn_parser(), argv[1:])
    return argv


def frame_argv(form: dict) -> List[str]:
    clip = str(form.get("clip") or "").strip()
    if not clip or not Path(clip).is_file():
        raise ClipsError("pick a 360 clip")
    return ["frame", clip, "--no-browser", "--port", "0"]


def outputs_of(out: Path) -> List[Path]:
    """The file a render wrote, or its _v1.._vN variations."""
    found = [out] if out.exists() else []
    found += sorted(out.with_name(f"{out.stem}_v{k}{out.suffix}") for k in range(1, 11)
                    if out.with_name(f"{out.stem}_v{k}{out.suffix}").exists())
    return found


def progress_of(line: str, k: int, n: int, f: float):
    """(variation index, fraction of that variation) after one log line of a render."""
    m = re.match(r"variation (\d+)/(\d+)", line)
    if m:
        return int(m.group(1)) - 1, 0.0
    m = re.match(r"shot (\d+)/(\d+)", line)
    if m:
        return k, 0.1 + 0.75 * int(m.group(1)) / int(m.group(2))
    if line.startswith("final pass"):
        return k, 0.88
    if line.startswith("loudness"):
        return k, 0.97
    return k, f


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind: str, argv: List[str], title: str, out: Optional[Path] = None,
                 variations: int = 1):
        self.id = next(Job._ids)
        self.kind, self.argv, self.title, self.out, self.variations = kind, argv, title, out, variations
        self.status = "queued"
        self.log: deque = deque(maxlen=LOG_LINES)
        self.progress, self.var_k, self.var_f = 0.0, 0, 0.0
        self.url: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None
        self.created, self.ended = time.time(), None
        self.cancelled = False

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "progress": round(self.progress, 3), "url": self.url,
                "command": "clips " + shlex.join(self.argv), "log": list(self.log)[-60:],
                "outputs": [p.name for p in outputs_of(self.out)] if self.out else [],
                "created": self.created, "ended": self.ended}


class Jobs:
    """Renders and learns run one at a time in a worker thread; framing jobs start at once."""

    def __init__(self, workdir: Path, cmd: Optional[List[str]] = None):
        self.workdir = workdir
        self.cmd = cmd or [sys.executable, "-m", "clips"]
        self.items: List[Job] = []
        self.lock = threading.Lock()
        self.q: "queue.Queue[Job]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def add(self, job: Job) -> Job:
        with self.lock:
            self.items.append(job)
        if job.kind == "frame":
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
        else:
            self.q.put(job)
        return job

    def get(self, jid: int) -> Optional[Job]:
        with self.lock:
            return next((j for j in self.items if j.id == jid), None)

    def snapshot(self) -> List[dict]:
        with self.lock:
            return [j.to_dict() for j in reversed(self.items)]

    def cancel(self, job: Job) -> None:
        job.cancelled = True
        p = job.proc
        if p and p.poll() is None:
            try:
                if hasattr(os, "killpg"):
                    os.killpg(p.pid, signal.SIGTERM)    # clips and its ffmpeg children
                else:
                    p.terminate()
            except (ProcessLookupError, PermissionError):
                pass
        elif job.status == "queued":
            job.status, job.ended = "cancelled", time.time()

    def _worker(self):
        while True:
            job = self.q.get()
            if not job.cancelled:
                self._run(job)

    def _run(self, job: Job):
        tmp = self.workdir / ".tmp" / str(job.id)       # render scratch, removed even on cancel
        tmp.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUNBUFFERED="1", TMPDIR=str(tmp))
        job.status = "running"
        try:
            job.proc = subprocess.Popen(self.cmd + job.argv, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, start_new_session=True)
        except OSError as exc:
            job.log.append(f"error: {exc}")
            job.status, job.ended = "failed", time.time()
            shutil.rmtree(tmp, ignore_errors=True)
            return
        buf = b""
        while True:
            chunk = job.proc.stdout.read1(4096) if hasattr(job.proc.stdout, "read1") else \
                job.proc.stdout.read(4096)
            if not chunk:
                break
            buf += chunk
            *lines, buf = re.split(rb"[\r\n]", buf)
            for raw in lines:
                self._line(job, raw.decode("utf-8", "replace").rstrip())
        if buf:
            self._line(job, buf.decode("utf-8", "replace").rstrip())
        code = job.proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)
        job.ended = time.time()
        if job.cancelled:
            job.status = "cancelled"
        elif code == 0:
            job.status, job.progress = "done", 1.0
        else:
            job.status = "failed"

    def _line(self, job: Job, line: str):
        if not line:
            return
        job.log.append(line)
        if job.kind == "render":
            job.var_k, job.var_f = progress_of(line, job.var_k, job.variations, job.var_f)
            job.progress = min(0.99, (job.var_k + job.var_f) / max(1, job.variations))
        m = re.search(r" at (http://127\.0\.0\.1:\d+/)", line)
        if job.kind == "frame" and m:
            job.url = m.group(1)


def list_dir(path: str, kind: str = "") -> dict:
    """Folders and matching files of one directory, for the page's file picker."""
    p = Path(path).expanduser() if path else Path.home()
    if not p.is_dir():
        p = p.parent if p.parent.is_dir() else Path.home()
    p = p.resolve()
    exts = KINDS.get(kind)
    dirs, files = [], []
    try:
        for e in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    dirs.append(e.name)
                elif exts is None or e.suffix.lower() in exts:
                    files.append({"name": e.name, "size": e.stat().st_size})
            except OSError:
                continue
    except OSError as exc:
        return {"path": str(p), "parent": str(p.parent), "dirs": [], "files": [],
                "error": str(exc)}
    return {"path": str(p), "parent": str(p.parent), "dirs": dirs, "files": files,
            "home": str(Path.home())}


def renders_list(renders: Path) -> List[dict]:
    items = sorted((f for f in renders.glob("*.mp4") if f.is_file()),
                   key=lambda f: f.stat().st_mtime, reverse=True)
    return [{"name": f.name, "size": f.stat().st_size, "mtime": f.stat().st_mtime}
            for f in items]


def profiles_list(profiles: Path) -> List[dict]:
    return [{"name": f.stem, "path": str(f)} for f in sorted(profiles.glob("*.json"))]


class UIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, workdir: Path, port: int = 0, cmd: Optional[List[str]] = None,
                 host: str = "127.0.0.1"):
        self.workdir = Path(workdir).expanduser().resolve()
        self.renders, self.profiles = self.workdir / "renders", self.workdir / "profiles"
        self.renders.mkdir(parents=True, exist_ok=True)
        self.profiles.mkdir(parents=True, exist_ok=True)
        self.jobs = Jobs(self.workdir, cmd)
        self.local_only = is_loopback(host)
        super().__init__((host, port), Handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"

    def lan_urls(self) -> List[str]:
        """Addresses other devices can use (only when not bound to loopback)."""
        if self.local_only:
            return []
        host = self.server_address[0]
        ips = [host] if host not in ("0.0.0.0", "") else []
        if not ips:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    s.connect(("10.255.255.255", 1))        # no packet is sent
                    ips = [s.getsockname()[0]]
            except OSError:
                ips = [socket.gethostname()]
        return [f"http://{ip}:{self.server_address[1]}/" for ip in ips]


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")


class Handler(BaseHTTPRequestHandler):
    server: UIServer

    def log_message(self, fmt, *args):      # quiet
        pass

    def _host_ok(self) -> bool:
        if not self.server.local_only:
            return True
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
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        srv = self.server
        if u.path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self._json(200, {"workdir": str(srv.workdir), "renders": renders_list(srv.renders),
                                    "profiles": profiles_list(srv.profiles),
                                    "jobs": srv.jobs.snapshot()})
        if u.path == "/api/ls":
            return self._json(200, list_dir(q.get("path", ""), q.get("kind", "")))
        if u.path.startswith("/renders/"):
            name = Path(u.path[len("/renders/"):]).name
            f = srv.renders / name
            if not name.endswith(".mp4") or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            extra = {"Content-Disposition": f'attachment; filename="{name}"'} if "dl" in q else None
            return send_file(self, f, "video/mp4", extra)
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if not self._host_ok() or self.headers.get("X-Clips") != "1":
            return self._send(403, b"forbidden", "text/plain")
        n = int(self.headers.get("Content-Length") or 0)
        if n < 0 or n > MAX_BODY:
            return self._json(400, {"error": "bad body"})
        try:
            form = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(form, dict):
                raise ValueError("expected an object")
        except ValueError as exc:
            return self._json(400, {"error": str(exc)})
        srv, path = self.server, urlparse(self.path).path
        try:
            if path == "/api/render":
                argv = render_argv(form, srv.renders)
                out = Path(argv[argv.index("-o") + 1])
                try:
                    var = int(argv[argv.index("--variations") + 1]) if "--variations" in argv else 1
                except ValueError:
                    var = 1
                job = srv.jobs.add(Job("render", argv, out.name, out, var))
            elif path == "/api/learn":
                argv = learn_argv(form, srv.profiles)
                job = srv.jobs.add(Job("learn", argv, f"learn {Path(argv[-3]).stem}"))
            elif path == "/api/frame":
                argv = frame_argv(form)
                job = srv.jobs.add(Job("frame", argv, f"frame {Path(argv[1]).name}"))
            elif m := re.fullmatch(r"/api/jobs/(\d+)/cancel", path):
                job = srv.jobs.get(int(m.group(1)))
                if job is None:
                    return self._json(404, {"error": "no such job"})
                srv.jobs.cancel(job)
            else:
                return self._send(404, b"not found", "text/plain")
        except ClipsError as exc:
            return self._json(400, {"error": str(exc)})
        self._json(200, job.to_dict())


def serve(workdir: Path = DEFAULT_WORKDIR, port: int = 8361, browser: bool = True,
          cmd: Optional[List[str]] = None, host: str = "127.0.0.1") -> UIServer:
    try:
        srv = UIServer(workdir, port, cmd, host)
    except OSError:
        srv = UIServer(workdir, 0, cmd, host)           # port taken
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    if browser:
        webbrowser.open(srv.url)
    return srv


def run_ui(args) -> int:
    srv = serve(Path(args.workdir), args.port, not args.no_browser, host=args.host)
    print(f"clips ui running at {srv.url}  (renders in {srv.renders})", file=sys.stderr)
    for u in srv.lan_urls():
        print(f"  from other devices: {u}  (anyone on your network can use it)", file=sys.stderr)
    print("Keep this terminal open while you use the page; Ctrl+C stops it.", file=sys.stderr)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for j in list(srv.jobs.items):
            if j.status in ("running", "queued"):
                srv.jobs.cancel(j)
        srv.shutdown()
    return 0
