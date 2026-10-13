"""SRT -> ASS conversion and ffmpeg filter-path escaping."""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, NamedTuple

from .ffmpeg import ClipsError

HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial Black,90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,60,60,520,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

POP = r"{\fscx120\fscy120\t(0,120,\fscx100\fscy100)\fad(60,80)}"


class Cue(NamedTuple):
    start: float
    end: float
    text: str


_TIME = r"(\d+):(\d{1,2}):(\d{1,2})(?:[,.](\d{1,3}))?"
_ARROW = re.compile(_TIME + r"\s*-->\s*" + _TIME)


def _secs(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int((ms or "0").ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> List[Cue]:
    text = text.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    cues: List[Cue] = []
    for block in re.split(r"\n[ \t]*\n", text):
        lines = [ln for ln in block.split("\n")]
        for i, ln in enumerate(lines):
            m = _ARROW.search(ln)
            if m:
                g = m.groups()
                start, end = _secs(*g[:4]), _secs(*g[4:])
                body = "\n".join(lines[i + 1:]).strip()
                if body and end > start:
                    cues.append(Cue(start, end, body))
                break
    return cues


def _fmt(t: float) -> str:
    cs = int(round(t * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


_TAGS = {"i": "i", "b": "b", "u": "u", "s": "s"}


def srt_text_to_ass(text: str) -> str:
    text = re.sub(r"\{\\an?\d\}", "", text)          # SRT-embedded ASS alignment tags
    text = text.replace("{", "(").replace("}", ")")  # braces would open ASS override blocks

    def tag(m: re.Match) -> str:
        close, name = m.group(1) == "/", m.group(2).lower()
        if name in _TAGS:
            # closing tag reverts to the style default (Default style is bold)
            return "{\\%s}" % _TAGS[name] if close else "{\\%s1}" % _TAGS[name]
        return ""

    text = re.sub(r"<\s*(/?)\s*([a-zA-Z]+)[^>]*>", tag, text)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return "\\N".join(lines)


STYLES = ("pop", "glow", "top")

_HDR_FORMAT = HEADER.split("[V4+ Styles]")[1].split("Style:")[0].strip()
_FONT_HEAD = HEADER.split("[V4+ Styles]")[0]
_EVENTS = "\n[Events]" + HEADER.split("[Events]")[1]

# name -> (style line, glow layer?, cue-level pop override, uppercase)
_STYLE_LINES = {
    "pop": HEADER.split("Style: ")[1].split("\n")[0],
    "glow": "Default,DejaVu Serif,120,&H00D8F0FF,&H00D8F0FF,&H00101010,&H80000000,0,0,0,0,100,100,2,0,1,2,0,5,70,70,0,1",
    # bottom-anchored (alignment 2) so the text bottom sits at y = 1920 - 1300 = 620
    "top": "Default,Arial,58,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,2,80,80,1300,1",
}
_POPS = {
    "pop": POP,
    "glow": r"{\fad(80,100)\fscx110\fscy110\t(0,150,\fscx100\fscy100)}",
    "top": r"{\fad(80,80)}",
}
_GLOW_L0 = r"{\bord14\blur12\shad0\1a&HFF&\3c&H0030A0FF&}"
WORD_POP = r"{\fscx125\fscy125\t(0,100,\fscx100\fscy100)}"
WORD_POP_RESET = r"{\fscx100\fscy100}"
MIN_WORD = 0.08
REVEAL_FRAC = 0.75
# (hidden, visible) alpha overrides; visible restores the style's alphas (BackColour alpha 80)
_ALPHA = ("{\\alpha&HFF&}", "{\\alpha&H00&\\4a&H80&}")
_ALPHA_GLOW0 = ("{\\alpha&HFF&}", "{\\1a&HFF&\\3a&H00&\\4a&HFF&}")


def build_header(style: str = "pop") -> str:
    return (_FONT_HEAD + "[V4+ Styles]\n" + _HDR_FORMAT + "\nStyle: " + _STYLE_LINES[style]
            + "\n" + _EVENTS)


def _visible_len(tok: str) -> int:
    return len(re.sub(r"\{[^}]*\}", "", tok))


def word_slices(start: float, end: float, lengths: List[int]) -> List[tuple]:
    """Contiguous (s, e) slices, one per word, covering [start, end].

    Words get time proportional to character count (min 80 ms) within the first
    REVEAL_FRAC of the cue; the last word's slice runs to the cue end.
    """
    n = len(lengths)
    dur = end - start
    if n == 1:
        return [(start, end)]
    total = min(max(dur * REVEAL_FRAC, n * MIN_WORD), dur * 0.9)
    wsum = float(sum(max(1, x) for x in lengths))
    d = [max(MIN_WORD, total * max(1, x) / wsum) for x in lengths]
    scale = min(1.0, total / sum(d))
    d = [x * scale for x in d]
    starts = [start]
    for x in d[:-1]:
        starts.append(starts[-1] + x)
    ends = starts[1:] + [end]
    return list(zip(starts, ends))


def cue_events(cue: Cue, style: str = "pop", word_reveal: bool = False) -> List[str]:
    raw = cue.text.upper() if style == "glow" else cue.text
    ass_text = srt_text_to_ass(raw)
    layers = [(_GLOW_L0, _ALPHA_GLOW0), ("", _ALPHA)] if style == "glow" else [("", _ALPHA)]

    def ev(layer, s, e, text):
        return f"Dialogue: {layer},{_fmt(s)},{_fmt(e)},Default,,0,0,0,,{text}\n"

    out = []
    if not word_reveal:
        for li, (pre, _a) in enumerate(layers):
            out.append(ev(li, cue.start, cue.end, _merge(_POPS[style], pre) + ass_text))
        return out
    toks = [t for t in re.split(r"(\\N|\s+)", ass_text) if t and not t.isspace()]
    widx = [i for i, t in enumerate(toks) if t != "\\N"]
    slices = word_slices(cue.start, cue.end, [_visible_len(toks[i]) for i in widx])
    n = len(widx)
    for k, (s, e) in enumerate(slices):
        fad = "{\\fad(%d,%d)}" % (60 if k == 0 else 0, 80 if k == n - 1 else 0)
        for li, (pre, (hid, vis)) in enumerate(layers):
            parts, seen = [], 0
            for i, t in enumerate(toks):
                if t == "\\N":
                    parts.append(t)
                    continue
                if seen < k:
                    parts.append(t + " ")
                elif seen == k:
                    parts.append(WORD_POP + t + WORD_POP_RESET + " ")
                else:
                    parts.append(hid + t + vis + " ")
                seen += 1
            body = "".join(parts).replace(" \\N", "\\N").rstrip()
            out.append(ev(li, s, e, _merge(fad, pre) + body))
    return out


def _merge(a: str, b: str) -> str:
    """Join two override blocks into one."""
    if not b:
        return a
    return a[:-1] + b[1:]


def srt_to_ass(text: str, style: str = "pop", word_reveal: bool = False) -> str:
    if style not in STYLES:
        raise ClipsError(f"unknown sub style {style!r} (choose from {', '.join(STYLES)})")
    out = [build_header(style) if style != "pop" else HEADER]
    for c in parse_srt(text):
        out.extend(cue_events(c, style, word_reveal))
    return "".join(out)


def _read(path: Path) -> str:
    raw = path.read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def check_subs_path(path) -> Path:
    p = Path(path)
    if p.suffix.lower() not in (".srt", ".ass"):
        raise ClipsError(f"unsupported subtitle format '{p.suffix}' (use .srt or .ass): {p}")
    if not p.is_file():
        raise ClipsError(f"subtitle file not found: {p}")
    return p


def prepare_subs(path, workdir, style: str = "pop", word_reveal: bool = False) -> Path:
    """Return an absolute .ass path to burn in (.srt converted into workdir)."""
    p = check_subs_path(path)
    if p.suffix.lower() == ".ass":
        return p.resolve()
    ass = Path(workdir) / "subs.ass"
    ass.write_text(srt_to_ass(_read(p), style, word_reveal), encoding="utf-8")
    return ass.resolve()


def _escape(s: str, chars: str) -> str:
    return "".join("\\" + c if c in chars else c for c in s)


def escape_filter_path(path) -> str:
    """Escape a path for use as a filter option value inside a filtergraph string.

    Level 1 (option value): backslash, colon, single quote. Level 2 (filtergraph):
    backslash, single quote, brackets, comma, semicolon.
    """
    return _escape(_escape(str(path), "\\':"), "\\'[],;")
