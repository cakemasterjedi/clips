import math

import numpy as np
import pytest

from clips import subject as S
from clips.ffmpeg import ClipsError


def det(x, y, s=10.0, score=0.9, cls=0, hist=None):
    return S.Det(x, y, s, 2 * s, score, cls, hist)


# ---------------------------------------------------------------- options
def test_classes_of():
    assert S.classes_of("auto") is None
    assert S.classes_of("biggest") is None
    assert S.classes_of("person") == [0]
    assert S.classes_of("Dog, bike") == [16, 1]
    with pytest.raises(ClipsError):
        S.classes_of("unicorn")


def test_providers_prefers_gpu_and_keeps_cpu():
    avail = ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert S.providers("auto", avail) == ["DmlExecutionProvider", "CPUExecutionProvider"]
    assert S.providers("cpu", avail) == ["CPUExecutionProvider"]
    assert S.providers("auto", ["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
    assert S.providers("gpu", ["MIGraphXExecutionProvider", "CPUExecutionProvider"])[0] == \
        "MIGraphXExecutionProvider"
    with pytest.raises(ClipsError, match="onnxruntime-directml"):
        S.providers("gpu", ["CPUExecutionProvider"])


# ---------------------------------------------------------------- model file
def test_model_download_verifies_checksum(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPS_CACHE", str(tmp_path / "cache"))
    src = tmp_path / "srv"
    src.mkdir()
    (src / "yolox_nano.onnx").write_bytes(b"not the model")
    with pytest.raises(ClipsError, match="checksum"):
        S.model_path("nano", src.as_uri() + "/")
    assert not (tmp_path / "cache" / "models" / "yolox_nano.onnx").exists()
    with pytest.raises(ClipsError):
        S.model_path("huge")


# ---------------------------------------------------------------- decoding
def test_postprocess_decodes_grid_and_suppresses_duplicates():
    size = 64
    grid, stride = S._grids(size)
    out = np.full((len(grid), 85), -20.0, np.float32)
    out[:, 4] = 0.0
    out[:, 5:] = 0.0
    # cell (x=3, y=2) at stride 8: centre (3+0.5)*8, box 16x24, class 16 (dog)
    i = int(np.where((grid[:, 0] == 3) & (grid[:, 1] == 2) & (stride[:, 0] == 8))[0][0])
    out[i, :4] = [0.5, 0.5, math.log(2), math.log(3)]
    out[i, 4], out[i, 5 + 16] = 0.9, 1.0
    j = int(np.where((grid[:, 0] == 4) & (grid[:, 1] == 2) & (stride[:, 0] == 8))[0][0])
    out[j, :4] = [-0.4, 0.5, math.log(2), math.log(3)]          # same box, lower score
    out[j, 4], out[j, 5 + 16] = 0.8, 1.0
    r = S.postprocess(out, size)
    assert len(r) == 1
    x0, y0, x1, y1, sc, cls = r[0]
    assert (x0, y0, x1, y1) == pytest.approx((20, 8, 36, 32), abs=1e-3)
    assert sc == pytest.approx(0.9) and cls == 16


def test_preprocess_pads_and_swaps_to_bgr():
    img = np.zeros((2, 3, 3), np.uint8)
    img[..., 0] = 200                                   # red
    x = S.preprocess(img, 4)
    assert x.shape == (1, 3, 4, 4)
    assert x[0, 2, 0, 0] == 200 and x[0, 0, 0, 0] == 0  # R is the last BGR channel
    assert x[0, 0, 3, 3] == 114


# ---------------------------------------------------------------- 360 geometry
def test_tile_centre_maps_to_tile_yaw():
    for yaw in S.tile_yaws():
        y, p = S.tile_to_angles(32, 32, 64, yaw)
        assert S.RF._wrap(y - yaw) == pytest.approx(0, abs=1e-6) and p == pytest.approx(0, abs=1e-6)
    # right of centre = larger yaw, above centre = positive pitch
    y, p = S.tile_to_angles(48, 16, 64, 0.0)
    assert y > 0 and p > 0


def test_tile_maps_sample_the_right_place():
    eq = np.zeros((90, 180, 3), np.uint8)
    # bright patch at yaw 100, pitch 10 (col = (100+180)/360*180, row = (90-10)/180*90)
    eq[38:42, 138:142] = 255
    size = 64
    maps = S.tile_maps(size, (180, 90))
    hits = []
    for (row, col), yaw in zip(maps, S.tile_yaws()):
        tile = eq[row, col]
        if tile.max() > 0:
            v, u = np.argwhere(tile[..., 0] > 0).mean(axis=0)
            hits.append(S.tile_to_angles(u + 0.5, v + 0.5, size, yaw))
    assert hits
    for y, p in hits:
        assert y == pytest.approx(100, abs=3) and p == pytest.approx(10, abs=3)


def test_merge_drops_duplicate_from_overlapping_tiles_across_seam():
    a = det(179.0, 0.0, 10, 0.9)
    b = det(-179.5, 0.5, 10, 0.7)
    c = det(-179.5, 0.5, 10, 0.7, cls=16)               # other class stays
    out = S.merge([b, a, c], wrap=True)
    assert a in out and c in out and b not in out


# ---------------------------------------------------------------- tracking
def test_track_follows_subject_across_the_seam():
    frames = [[det(170 + 4 * k if 170 + 4 * k <= 180 else 170 + 4 * k - 360, 0)] for k in range(8)]
    tr = S.track(frames, wrap=True)
    assert len(tr) == 1 and len(tr[0].samples) == 8


def test_track_keeps_two_people_apart_and_uses_low_scores():
    frames = []
    for k in range(6):
        frames.append([det(0.0 + k, 0), det(60.0 - k, 0, score=0.9 if k == 0 else 0.3)])
    tr = S.track(frames, wrap=True)
    assert len(tr) == 2
    assert sorted(len(t.samples) for t in tr) == [6, 6]


def test_pick_chain_reacquires_lookalike_after_occlusion():
    red = [0.0] * 32
    red[3] = 1.0
    blue = [0.0] * 32
    blue[20] = 1.0
    frames = []
    for k in range(20):
        f = []
        if k < 6:
            f.append(det(10 + k, 0, 12, 0.9, hist=red))          # subject
        if 10 <= k:
            f.append(det(-60, 0, 12, 0.9, hist=red))             # subject again, elsewhere
        f.append(det(90, 0, 8, 0.9, hist=blue))                  # someone else, smaller
        frames.append(f)
    path = S.subject_path(frames, "person", wrap=True)
    assert path[0][0] == pytest.approx(10) and path[15][0] == pytest.approx(-60)
    assert all(p is None or abs(p[0] - 90) > 1 for p in path)
    # the 4 missing samples between the two tracks are within FILL_GAP (2 s), so bridged
    assert path[7] is not None


def test_auto_prefers_people_and_biggest_takes_size():
    frames = [[det(0, 0, 5, 0.9, cls=0), det(90, 0, 30, 0.9, cls=2)] for _ in range(4)]
    assert S.subject_path(frames, "auto", True)[0][0] == pytest.approx(0)
    assert S.subject_path(frames, "biggest", True)[0][0] == pytest.approx(90)
    assert S.subject_path(frames, "dog", True) == [None] * 4


def test_person_aim_is_above_box_centre():
    d = det(0, 0.5, 0.1, cls=0)
    assert S.aim(d, wrap=False)[1] < 0.5
    assert S.aim(det(0, 0, 10, cls=0), wrap=True)[1] > 0


def test_fill_gaps_interpolates_short_gaps_the_short_way_round():
    p = [(170.0, 0.0), None, (-170.0, 2.0), None, None, None, None, None, (0.0, 0.0)]
    out = S.fill_gaps(p, wrap=True, max_gap=2)
    assert out[1] == pytest.approx((180.0, 1.0)) or out[1] == pytest.approx((-180.0, 1.0))
    assert out[4] is None                                  # too long to bridge


# ---------------------------------------------------------------- outputs
def test_targets_use_motion_only_where_subject_missing():
    path = [(10.0, 0.0), (12.0, 0.0), None, None, None, None]
    motion = [(0.5, 90.0, 0.0), (2.5, -90.0, 5.0)]
    tg = S.targets(path, motion)
    assert (0.0, 10.0, 0.0) in tg and (0.5, 12.0, 0.0) in tg
    assert (2.5, -90.0, 5.0) in tg and (0.5, 90.0, 0.0) not in tg


def test_shot_view_and_track():
    path = [(170.0, 4.0), (-170.0, 6.0), None, None]
    y, p = S.shot_view(path, None, 0.0, 1.0)
    assert abs(S.RF._wrap(y - 180)) < 1e-6 and p == pytest.approx(5.0)
    assert S.shot_view(path, None, 1.0, 1.0) is None
    flat = [(0.2, 0.5), (0.3, 0.5), (0.7, 0.4)]
    assert S.shot_track(flat, 0.0, 1.5) == pytest.approx((0.2, 0.5, 0.7, 0.4))


def test_missing_onnxruntime_is_a_clear_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "onnxruntime":
            raise ImportError
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ClipsError, match="requirements-vision"):
        S._ort()


# ---------------------------------------------------------------- real model (optional)
def _have_model():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    p = S.cache_dir() / "models" / S.MODELS["nano"][0]
    return p.exists()


@pytest.mark.skipif(not _have_model(), reason="onnxruntime or cached yolox_nano.onnx missing")
def test_real_model_runs_on_a_tiled_360_frame():
    d = S.Detector("nano", "cpu")
    tiles = [np.full((d.size, d.size, 3), 114, np.uint8) for _ in range(S.TILES)]
    out = S.dets_360(tiles, [d(t) for t in tiles], d.size)
    assert out == []                                     # nothing to find in flat grey
