import json
import shutil
import subprocess

import numpy as np
import pytest

from clips import autoframe as AF
from clips import cli
from clips import reframe as RF
from clips import style as S
from clips.ffmpeg import ffmpeg_exe
from test_reframe import ff, song  # noqa: F401  (fixture)

W, H = 288, 512


def texture(h, w, seed=0):
    r = np.random.default_rng(seed)
    img = np.zeros((h, w))
    for k in (16, 32, 64):
        img += np.kron(r.uniform(0, 255, (h // k + 1, w // k + 1)), np.ones((k, k)))[:h, :w]
    return img / 3


def write(path, frames, fps=30):
    n, h, w = frames.shape
    subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
                    "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt",
                    "yuv420p", str(path)],
                   input=np.clip(frames, 0, 255).astype(np.uint8).tobytes(), check=True)


def ref_frames(kind, n=120):
    big = texture(H * 2, W * 4)
    if kind == "pan":                                    # 0.2 frame widths / s
        return np.stack([big[100:100 + H, int(20 + i / 30 * 0.2 * W):][:, :W] for i in range(n)])
    if kind == "push":                                   # zoom 0.1 / s
        return np.stack([S.zoom(big[:H, :W], float(np.exp(0.1 * i / 30))) for i in range(n)])
    still = np.repeat(big[None, :H, :W] * 0.5, n, axis=0)    # locked-off, action low in frame
    for i in range(n):
        x = int(W / 2 + 60 * np.sin(i / 5))
        still[i, int(0.72 * H) - 20:int(0.72 * H) + 20, x - 20:x + 20] = 230
    return still


# ------------------------------------------------------------------ learning
@pytest.mark.parametrize("kind", ["pan", "push", "still"])
def test_shot_framing(kind):
    f = ref_frames(kind).astype(np.uint8)
    s = S.shot_framing(f[:, ::2, ::2], 0, len(f), slice(None))     # analysis size 144x256
    if kind == "pan":
        assert s["pan"] == pytest.approx(0.2, abs=0.02) and abs(s["push"]) < S.MOVE_PUSH
    elif kind == "push":
        assert s["push"] == pytest.approx(0.1, abs=0.02) and s["pan"] < S.MOVE_PAN
    else:
        assert s["pan"] < S.MOVE_PAN and abs(s["push"]) < S.MOVE_PUSH
        assert s["subject_y"] == pytest.approx(0.72, abs=0.05)


def test_learn_framing_end_to_end(tmp_path):
    for kind in ("pan", "push", "still"):
        write(tmp_path / f"{kind}.mp4", ref_frames(kind))
    st = {k: S.analyze_reference(tmp_path / f"{k}.mp4").extra["framing"] for k in ("pan", "push", "still")}
    assert st["pan"]["moving"] == 1 and st["pan"]["pan_speed"] == pytest.approx(0.2, abs=0.03)
    assert st["push"]["push_share"] == 1 and st["push"]["push"] == pytest.approx(0.1, abs=0.03)
    assert st["still"]["moving"] == 0 and st["still"]["subject_y"] > 0.6
    prof = S.build_profile([S.analyze_reference(tmp_path / "pan.mp4"),
                            S.analyze_reference(tmp_path / "still.mp4")], "c")
    fr = prof["framing"]
    assert fr["moving"] == 0.5 and fr["pan_speed"] == pytest.approx(0.2, abs=0.03)


# ------------------------------------------------------------------ camera
def jump(at=2.0, to=90.0, dur=12.0):
    return [(t, 0.0 if t < at else to, 0.0) for t in np.arange(0, dur, 0.5)]


def first_reach(keys, yaw, tol=5):
    return next(k.t for k in keys if abs(RF._wrap(k.yaw - yaw)) < tol)


def test_camera_speed_follows_style():
    fast = AF.camera_path(jump(), 12, dict(AF.DEFAULT, moving=1.0, pan_speed=1.0))
    slow = AF.camera_path(jump(), 12, dict(AF.DEFAULT, moving=1.0, pan_speed=0.05))
    assert first_reach(fast, 90) - 2.0 <= 2.5
    assert first_reach(slow, 90) - 2.0 >= 8.0
    steps = np.abs(np.diff([k.yaw for k in slow]))
    assert steps.max() <= AF.MIN_SPEED * AF.STEP + 1e-6           # speed cap (deg per sample)


def test_dead_zone_for_still_creators():
    small = jump(to=15.0)
    still = AF.camera_path(small, 12, dict(AF.DEFAULT, moving=0.0))
    mover = AF.camera_path(small, 12, dict(AF.DEFAULT, moving=1.0, pan_speed=0.5))
    assert all(k.yaw == 0 for k in still)                         # 15 deg stays inside 21 deg
    assert abs(mover[-1].yaw - 15) < 3


def test_push_in_and_subject_height():
    hold = [(t, 30.0, 0.0) for t in np.arange(0, 10, 0.5)]
    keys = AF.camera_path(hold, 10, dict(AF.DEFAULT, push_share=1.0, push=0.05, subject_y=0.7))
    assert keys[0].fov == 60 and keys[-1].fov < 60 * 0.7 ** 0.5 and keys[-1].fov >= 45
    assert all(b.fov <= a.fov for a, b in zip(keys, keys[1:]))
    vf = RF.v_fov(60, (1080, 1920))
    assert keys[0].pitch == pytest.approx(0.2 * vf, abs=0.1)      # action low -> look up


def test_simplify_keeps_the_path():
    keys = AF.camera_path(jump(), 12, dict(AF.DEFAULT, moving=1.0, pan_speed=0.3))
    few = AF.simplify(keys)
    assert 2 <= len(few) < len(keys) / 2
    for k in keys:
        y, p, f, _ = RF.interpolate(few, k.t)
        assert abs(RF._wrap(y - k.yaw)) <= AF.TOL_ANGLE + 1e-6 and abs(f - k.fov) <= AF.TOL_FOV + 1e-6
    flat = [RF.Keyframe(t, 10, 0, 60, 0) for t in range(10)]
    assert len(AF.simplify(flat)) == 2


def test_flat_track_style():
    tr, place = AF.flat_track((0.2, 0.3, 0.8, 0.5), dict(AF.DEFAULT, moving=0.1, subject_y=0.7))
    assert tr == (0.5, 0.4, 0.5, 0.4) and place == 0.7
    tr, _ = AF.flat_track((0.2, 0.3, 0.8, 0.5), dict(AF.DEFAULT, moving=0.9))
    assert tr == (0.2, 0.3, 0.8, 0.5)
    crop = RF.track_crop(RF.View(track=tr, place=0.7), 2.0, (1080, 1920))
    assert "*ih-1920*0.7000" in crop and "*iw-1080*0.5000" in crop


# ------------------------------------------------------------------ end to end
@pytest.fixture(scope="module")
def walker(tmp_path_factory):
    """Equirect pano where a box walks from yaw -90 to +90 over 8 s."""
    d = tmp_path_factory.mktemp("walk")
    p = d / "walk.mp4"
    ff("-f", "lavfi", "-i", "color=black:s=1280x640:r=30:d=8",
       "-f", "lavfi", "-i", "color=white:s=80x80:r=30:d=8",
       "-filter_complex", "[0][1]overlay=x='280+80*t+20*sin(6*t)':y=280",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", p)
    return p


def test_autoframe_command_follows_action(walker, tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    clip = d / "walk.mp4"
    shutil.copy(walker, clip)
    prof = tmp_path / "p.json"
    prof.write_text(json.dumps({"version": 1, "framing": {"moving": 1.0, "pan_speed": 0.6}}))
    assert cli.main(["autoframe", str(d), "--360", "equirect", "--style", str(prof)]) == 0
    keys = RF.load_keyframes(clip)
    assert keys and len(keys) >= 3
    for t in (2.0, 4.0, 6.0):
        box = (280 + 80 * t + 40) / 1280 * 360 - 180
        assert abs(RF._wrap(RF.interpolate(keys, t)[0] - box)) < 30, t
    RF.sidecar_path(clip).write_text(json.dumps(RF.keyframes_json([RF.Keyframe(0, 5)])))
    assert cli.main(["autoframe", str(d), "--360", "equirect"]) == 0      # kept: hand-made
    assert RF.load_keyframes(clip)[0].yaw == 5
    assert cli.main(["autoframe", str(d), "--360", "equirect", "--force"]) == 0
    assert abs(RF.load_keyframes(clip)[0].yaw - (-90)) < 30           # replaced by auto


def test_render_auto_frame(walker, song, tmp_path, capsys):  # noqa: F811
    d = tmp_path / "in"
    d.mkdir()
    clip = d / "walk.mp4"
    shutil.copy(walker, clip)
    common = [str(d), "--audio", str(song), "--360", "equirect", "--auto-frame", "--no-loudnorm"]
    assert cli.main(common + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert all(s["view"].get("keyframes", 0) >= 2 for s in plan["shots"])
    assert not RF.sidecar_path(clip).exists()                    # dry run writes nothing
    out = tmp_path / "o.mp4"
    assert cli.main(common + ["-o", str(out)]) == 0
    assert RF.sidecar_path(clip).exists() and out.exists()
