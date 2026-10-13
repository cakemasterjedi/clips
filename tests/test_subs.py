import pytest

from clips import subs as U
from clips.ffmpeg import ClipsError


def test_parse_edge_cases():
    txt = ("\ufeff1\r\n00:00:01,500 --> 00:00:03.250\r\nHello\r\nworld\r\n\r\n\r\n   \r\n"
           "2\r\n0:00:04,5 --> 0:00:05,000\r\n<i>Hi</i> <b>there</b> <font color=\"red\">x</font>\r\n")
    cues = U.parse_srt(txt)
    assert len(cues) == 2
    assert cues[0].start == 1.5 and cues[0].end == 3.25
    assert cues[0].text == "Hello\nworld"
    assert cues[1].start == 4.5
    ass = U.srt_to_ass(txt)
    assert "Hello\\Nworld" in ass
    assert "{\\i1}Hi{\\i} {\\b1}there{\\b} x" in ass
    assert "0:00:01.50,0:00:03.25" in ass
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass and "WrapStyle: 0" in ass
    assert "ScaledBorderAndShadow: yes" in ass and ",2,60,60,520,1" in ass
    assert U.POP in ass


def test_no_trailing_newline_and_braces():
    cues = U.parse_srt("1\n00:00:00,000 --> 00:00:01,000\na {b}")
    assert cues[0].text == "a {b}"
    assert "a (b)" in U.srt_to_ass("1\n00:00:00,000 --> 00:00:01,000\na {b}")


def test_ass_passthrough_and_unknown(tmp_path):
    f = tmp_path / "x.ass"
    f.write_text("[Script Info]\n")
    assert U.prepare_subs(f, tmp_path) == f.resolve()
    assert f.read_text() == "[Script Info]\n"
    with pytest.raises(ClipsError):
        U.check_subs_path(tmp_path / "x.vtt")


def test_srt_file_converted(tmp_path):
    f = tmp_path / "x.srt"
    f.write_bytes("\ufeff1\r\n00:00:00,000 --> 00:00:01,000\r\nHi\r\n".encode())
    out = U.prepare_subs(f, tmp_path)
    assert out.suffix == ".ass" and "Dialogue" in out.read_text()


def test_escape_nasty():
    assert U.escape_filter_path("/tmp/a b/it's: [x],y.ass") == \
        "/tmp/a b/it\\\\\\'s\\\\: \\[x\\]\\,y.ass"
    assert U.escape_filter_path("C:\\a") == "C\\\\:\\\\\\\\a"
