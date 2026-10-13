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
it to other devices on the network: then any Host is accepted, and every request needs the
random key from the printed link (stored in a cookie on first visit); anyone with that link
can browse folders and start renders. Renders run one at a time.
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
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.cookies import SimpleCookie
from hmac import compare_digest
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
LOG_LINES = 5000

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
          "seed": "--seed", "style": "--style", "zoom": "--zoom", "bpm": "--bpm"}
# flags with their own control on the page (left out of the "add an option" list)
FORM_DESTS = {"input", "audio", "output", "help", "no_music", "mute", "audio_start", "subs",
              "dry_run", *FLAGS, *VALUES}
MAX_UPLOAD = 64 << 30
LOG_SEND = 300              # newest log lines sent per job on each refresh


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


def option_list() -> List[dict]:
    """Every `clips` option without its own control on the page, for the "add an option" menu."""
    from .cli import build_parser

    out = []
    for act in build_parser()._actions:
        if not act.option_strings or act.dest in FORM_DESTS:
            continue
        flag = max(act.option_strings, key=len)
        kind = "bool" if act.nargs == 0 else ("choice" if act.choices else "value")
        out.append({"flag": flag, "kind": kind, "help": (act.help or "").replace("%%", "%"),
                    "choices": [str(c) for c in act.choices] if act.choices else None,
                    "metavar": act.metavar or "", "repeat": isinstance(act.default, list)})
    return sorted(out, key=lambda o: o["flag"])


def render_argv(form: dict, renders: Path) -> List[str]:
    """`clips` arguments for a render form; validated with the real parser."""
    from .cli import build_parser

    inp = str(form.get("input") or "").strip()
    if not inp or not Path(inp).exists():
        raise ClipsError("pick a footage folder or clip")
    music = form.get("music") or "file"
    if music == "file":
        song = str(form.get("audio") or "").strip()
        if not song or not Path(song).is_file():
            raise ClipsError("pick a song (or choose no song)")
        sound = ["--audio", song]
    else:
        sound = ["--mute"] if music == "mute" else ["--no-music"]
    out = renders / f"{safe_name(form.get('name'), time.strftime('edit-%Y%m%d-%H%M%S'))}.mp4"
    argv = [inp, *sound, "-o", str(out)]
    start = str(form.get("audio_start") or "").strip()
    if start and music == "file":
        argv += ["--audio-start", start]
    if form.get("captions") == "file":
        subs = str(form.get("subs") or "").strip()
        if not subs:
            raise ClipsError("pick a caption file (or choose no captions)")
        argv += ["--subs", subs]
    for key, flag in VALUES.items():
        v = form.get(key)
        if v is None or str(v).strip() == "" or (key in ("variations", "zoom") and float(v or 1) == 1):
            continue
        if key == "bpm" and music == "file":
            continue
        argv += [flag, str(v).strip()]
    argv += [flag for key, flag in FLAGS.items() if form.get(key)]
    extra = []
    for item in form.get("options") or []:              # [flag, value] rows from the picker
        if not isinstance(item, (list, tuple)) or not item or not str(item[0]).startswith("--"):
            continue
        extra.append(str(item[0]))
        if len(item) > 1 and item[1] not in (None, True, ""):
            extra.append(str(item[1]))
    text = str(form.get("extra") or "").strip()
    if text:
        try:
            extra += shlex.split(text)
        except ValueError as exc:
            raise ClipsError(f"extra flags: {exc}") from None
    if any(a in ("-o", "--output") or a.startswith("--output=") for a in extra):
        raise ClipsError("set the output name in the name box, not with -o")
    argv += extra
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


STAGES = [(r"^variation (\d+)/(\d+)", "variation {0}/{1}"),
          (r"^analysing (.+)\.\.\.", "analysing {0}"),
          (r"^downloading|\[download\]", "downloading"),
          (r"^note: --audio-start auto", "picking the song section"),
          (r"^auto-framed", "auto framing"),
          (r"^mode=", "planning cuts"),
          (r"^rendering (\d+) shots", "rendering {0} shots"),
          (r"^shot (\d+)/(\d+)", "rendering shot {0}/{1}"),
          (r"^final pass", "final pass (joining, captions, sound)"),
          (r"^loudness", "loudness"),
          (r"^building framing proxy", "building the 360 preview"),
          (r"^framing .* at http", "framing tool ready"),
          (r"^wrote ", "finished writing")]


def stage_of(line: str) -> Optional[str]:
    for pat, label in STAGES:
        m = re.search(pat, line)
        if m:
            return label.format(*m.groups())
    return None


def group_cpu(pgid: int) -> Optional[float]:
    """CPU seconds used so far by every process in a process group (Linux /proc), so the
    page can show work is happening even when nothing is printed."""
    try:
        tick = os.sysconf("SC_CLK_TCK")
        total = 0
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat", "rb") as fh:
                    st = fh.read().rsplit(b")", 1)[1].split()
            except OSError:
                continue
            if int(st[2]) == pgid:                       # fields after "(comm)": state ppid pgrp
                total += int(st[11]) + int(st[12]) + int(st[13]) + int(st[14])
        return total / tick
    except (OSError, ValueError, IndexError, AttributeError):
        return None


class Job:
    _ids = itertools.count(1)

    def __init__(self, kind: str, argv: List[str], title: str, out: Optional[Path] = None,
                 variations: int = 1):
        self.id = next(Job._ids)
        self.kind, self.argv, self.title, self.out, self.variations = kind, argv, title, out, variations
        self.status = "queued"
        self.log: deque = deque(maxlen=LOG_LINES)        # (seq, seconds since start, line)
        self.seq = 0
        self.progress, self.var_k, self.var_f = 0.0, 0, 0.0
        self.url: Optional[str] = None
        self.proc: Optional[subprocess.Popen] = None
        self.created, self.started, self.ended = time.time(), None, None
        self.last_line = None
        self.stage = "waiting for the job before it" if kind != "frame" else "starting"
        self.cancelled = False
        self.logfile: Optional[Path] = None

    def to_dict(self) -> dict:
        now = time.time()
        running = self.status == "running" and self.proc is not None
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "progress": round(self.progress, 3), "url": self.url, "stage": self.stage,
                "command": "clips " + shlex.join(self.argv),
                "log": [list(x) for x in list(self.log)[-LOG_SEND:]], "seq": self.seq,
                "elapsed": round(((self.ended or now) - self.started), 1) if self.started else 0,
                "idle": round(now - (self.last_line or self.started or now), 1) if running else None,
                "cpu": group_cpu(self.proc.pid) if running and hasattr(os, "killpg") else None,
                "pid": self.proc.pid if self.proc else None,
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
        logs = self.workdir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        job.logfile = logs / f"{time.strftime('%Y%m%d-%H%M%S')}-{job.id}-{safe_name(job.title, job.kind)}.log"
        env = dict(os.environ, PYTHONUNBUFFERED="1", TMPDIR=str(tmp), CLIPS_VERBOSE="1")
        job.status, job.started, job.stage = "running", time.time(), "starting"
        self._line(job, "$ clips " + shlex.join(job.argv))
        try:
            job.proc = subprocess.Popen(self.cmd + job.argv, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                        env=env, start_new_session=True)
        except OSError as exc:
            self._line(job, f"error: {exc}")
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
        ended = time.time()
        status = "cancelled" if job.cancelled else "done" if code == 0 else "failed"
        self._line(job, f"-- {status} (exit {code}) after {ended - job.started:.1f}s")
        if status == "done":
            job.progress = 1.0
        job.stage, job.ended, job.status = status, ended, status      # status last: readers poll it

    def _line(self, job: Job, line: str):
        if not line:
            return
        now = time.time()
        job.last_line = now
        job.seq += 1
        t = round(now - (job.started or now), 1)
        job.log.append((job.seq, t, line))
        if job.logfile:
            try:
                with open(job.logfile, "a", encoding="utf-8") as fh:
                    fh.write(f"[{time.strftime('%H:%M:%S')} +{t:.1f}s] {line}\n")
            except OSError:
                pass
        st = stage_of(line)
        if st:
            job.stage = st
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
        # on the network the page needs a key from the printed link (kept in a cookie after)
        self.key = None if self.local_only else secrets.token_urlsafe(12)
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
        return [f"http://{ip}:{self.server_address[1]}/?key={self.key}" for ip in ips]


def is_loopback(host: str) -> bool:
    return host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")


class Handler(BaseHTTPRequestHandler):
    server: UIServer

    def log_message(self, fmt, *args):      # quiet
        pass

    def _host_ok(self) -> bool:
        if not self.server.local_only:                     # this PC itself needs no key
            return is_loopback(self.client_address[0]) or self._key_ok()
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost", "[::1]")

    def _query_key(self) -> str:
        return parse_qs(urlparse(self.path).query).get("key", [""])[0]

    def _key_ok(self) -> bool:
        want = self.server.key or ""
        c = SimpleCookie()
        try:
            c.load(self.headers.get("Cookie") or "")
        except Exception:
            pass
        have = c["clips_key"].value if "clips_key" in c else ""
        return bool(want) and (compare_digest(have, want) or compare_digest(self._query_key(), want))

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
            msg = (b"forbidden" if self.server.local_only else
                   b"open the full link that `clips ui` printed (it ends in ?key=...)")
            return self._send(403, msg, "text/plain")
        u = urlparse(self.path)
        if self.server.key and self._query_key() and u.path == "/":          # keep the key, drop it from the URL
            return self._send(303, b"", "text/plain", {
                "Location": "/", "Set-Cookie": f"clips_key={self.server.key}; Path=/; HttpOnly; "
                                               "SameSite=Strict; Max-Age=2592000"})
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
        if u.path == "/api/options":
            return self._json(200, option_list())
        if m := re.fullmatch(r"/api/jobs/(\d+)/log", u.path):
            job = srv.jobs.get(int(m.group(1)))
            if job is None or not job.logfile or not job.logfile.is_file():
                return self._send(404, b"no log yet", "text/plain")
            return self._send(200, job.logfile.read_bytes(), "text/plain; charset=utf-8")
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
        if urlparse(self.path).path == "/api/upload":
            return self._upload(n)
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
            elif path == "/api/mkdir":
                return self._json(200, make_folder(str(form.get("path") or ""),
                                                   str(form.get("name") or "")))
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

    def _upload(self, n: int):
        """Raw file body (the page sends one file per request) into ?dir= as ?name=."""
        q = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
        try:
            dest = upload_target(q.get("dir", ""), q.get("name", ""))
        except ClipsError as exc:
            return self._json(400, {"error": str(exc)})
        if n <= 0 or n > MAX_UPLOAD:
            return self._json(400, {"error": "empty or too large"})
        part = dest.with_name("." + dest.name + ".part")
        left = n
        try:
            with open(part, "wb") as fh:
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        raise OSError("upload interrupted")
                    fh.write(chunk)
                    left -= len(chunk)
            os.replace(part, dest)
        except OSError as exc:
            part.unlink(missing_ok=True)
            return self._json(400, {"error": str(exc)})
        self._json(200, {"ok": True, "path": str(dest), "name": dest.name, "size": n})


def upload_target(folder: str, name: str) -> Path:
    """Where an uploaded file goes: inside an existing folder, plain file name, never
    replacing a file (name-1.ext, name-2.ext ...)."""
    d = Path(folder).expanduser()
    if not folder or not d.is_dir():
        raise ClipsError("pick a folder to upload into")
    base = Path(name.replace("\\", "/")).name.strip().lstrip(".")
    if not base:
        raise ClipsError("bad file name")
    dest = d / base
    k = 1
    while dest.exists():
        dest = d / f"{Path(base).stem}-{k}{Path(base).suffix}"
        k += 1
    return dest


def make_folder(parent: str, name: str) -> dict:
    p = Path(parent).expanduser()
    if not parent or not p.is_dir():
        raise ClipsError("open a folder first")
    name = name.strip()
    if not name or name in (".", "..") or "/" in name or "\\" in name or name.startswith("."):
        raise ClipsError("folder names cannot contain / or start with a dot")
    new = p / name
    try:
        new.mkdir()
    except FileExistsError:
        raise ClipsError(f"{name} already exists") from None
    except OSError as exc:
        raise ClipsError(str(exc)) from None
    return {"ok": True, "path": str(new.resolve())}


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
        print(f"  from other devices: {u}\n  (anyone with this link can use clips on this PC; if it does "
              f"not load, allow the port in your firewall, e.g. sudo ufw allow "
              f"{srv.server_address[1]}/tcp)", file=sys.stderr)
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
