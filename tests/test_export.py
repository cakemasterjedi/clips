import numpy as np
import soundfile as sf

from clips import cli
from clips import export as E
from conftest import SR, clicks, make_video


def test_timecode():
    assert E.timecode(0, 30) == "00:00:00:00"
    assert E.timecode(61.5, 30) == "00:01:01:15"
    assert E.timecode(3600 + 1 / 60, 60) == "01:00:00:01"


def test_edl_from_dry_run(tmp_path, capsys):
    d = tmp_path / "in"
    d.mkdir()
    make_video(d / "a.mp4", "640x360", 30, 5)
    make_video(d / "b.mp4", "640x360", 30, 5, "smptebars")
    t = np.arange(0.25, 12, 0.5)
    song = tmp_path / "song.wav"
    sf.write(song, clicks(t, np.full(t.size, 0.8), 12), SR)
    edl = tmp_path / "cut.edl"
    assert cli.main([str(d), "--audio", str(song), "--no-intro", "--duration", "4",
                     "--audio-start", "2", "--dry-run", "--export-edl", str(edl)]) == 0
    text = edl.read_text()
    lines = text.splitlines()
    assert lines[0] == "TITLE: clips" and lines[1] == "FCM: NON-DROP FRAME"
    v = [l for l in lines if l[5:13] == "AX      " and l.split()[2] == "V"]
    a = [l for l in lines if l[5:13] == "AX      " and l.split()[2] == "A"]
    assert len(v) >= 2 and len(a) == 1
    assert v[0].split()[6] == "00:00:00:00"                         # record starts at 0
    assert v[-1].split()[7] == "00:00:04:00"                        # ...and ends at 4 s
    assert a[0].split()[4:8] == ["00:00:02:00", "00:00:06:00", "00:00:00:00", "00:00:04:00"]
    assert "* FROM CLIP NAME: song.wav" in text and "* FROM CLIP NAME: a.mp4" in text
    for prev, nxt in zip(v, v[1:]):                                 # contiguous record times
        assert prev.split()[7] == nxt.split()[6]
