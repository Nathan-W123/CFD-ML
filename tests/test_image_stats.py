"""Real-vs-synthetic image statistics (scripts/image_stats.py).

On constructed arrays: the noise estimate recovers a known sigma and ignores
scene structure, JPEG raises blockiness, contrast / brightness / flicker /
global shift are exact, sharpness does not depend on contrast; the coverage of
real clips by the synthetic [P1, P99] range and the notes point at the right
render parameters; and a tiny end-to-end run (a video, an image folder and a
missing clip in a manifest, stand-in synthetic sequences).
"""

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import image_stats as S  # noqa: E402


def _scene(h=192, w=256, seed=0):
    """Float scene in [30, 220] with a ramp, axis-aligned rectangles, a disc and a smooth texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 60.0 + 0.4 * x + 0.25 * y
    img[40:100, 30:120] += 40.0
    img[120:170, 150:230] -= 30.0
    cv2.circle(img, (w - 50, 50), 30, 180.0, -1)
    tex = cv2.GaussianBlur(rng.normal(0.0, 1.0, (h, w)), (0, 0), 3.0)
    return img + 15.0 * tex / tex.std()


def _u8(img):
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _noisy(sigma, seed=1, base=None):
    base = _scene() if base is None else base
    return _u8(base + np.random.default_rng(seed).normal(0.0, sigma, base.shape))


@pytest.mark.parametrize("sigma", [1.5, 3.0, 6.0, 10.0])
def test_noise_sigma_recovers_known_sigma(sigma):
    est = S.noise_sigma(_noisy(sigma))
    assert abs(est / sigma - 1.0) <= 0.15, (sigma, est)


def test_noise_sigma_ignores_scene_structure_and_clipping():
    flat = S.noise_sigma(_noisy(4.0, base=np.full((192, 256), 128.0)))
    scene = S.noise_sigma(_noisy(4.0))
    assert abs(scene / flat - 1.0) < 0.05
    # Saturated pixels hold no noise: a half-clipped frame still measures the unclipped half.
    half = _noisy(4.0, base=np.full((192, 256), 128.0))
    half[:, :128] = 255
    assert abs(S.noise_sigma(half) / 4.0 - 1.0) <= 0.15
    assert math.isnan(S.noise_sigma(np.full((32, 32), 255, np.uint8)))
    with pytest.raises(ValueError):
        S.noise_sigma(np.zeros((32, 32), np.float32))


def test_blockiness_higher_for_jpeg_than_raw():
    raw = _noisy(3.0)
    b_raw = S.blockiness(raw)
    assert 0.9 < b_raw < 1.1

    def jpeg(q):
        return cv2.imdecode(cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)

    b60, b30, b10 = (S.blockiness(jpeg(q)) for q in (60, 30, 10))
    assert b60 > b_raw + 0.05 and b30 > b_raw + 0.15 and b10 > b30 > b60


def test_contrast_and_brightness_are_exact():
    img = np.full((50, 40), 200, np.uint8)
    img[:20] = 50                                           # 40 % at 50, 60 % at 200
    st = S.frame_stats(img)
    assert st["contrast"] == pytest.approx(150 / 255)
    assert st["brightness"] == pytest.approx((0.4 * 50 + 0.6 * 200) / 255)
    ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))  # every gray level once per row
    assert S.frame_stats(ramp)["contrast"] == pytest.approx((243 - 12) / 255)  # lower P5 / P95
    assert S.frame_stats(ramp)["brightness"] == pytest.approx(127.5 / 255)


def test_sharpness_is_normalised_by_contrast_and_drops_with_blur():
    img = _noisy(2.0)
    low = _u8(0.5 * img.astype(np.float64) + 64.0)          # half the contrast, same structure
    s_full, s_low = S.frame_stats(img)["sharpness"], S.frame_stats(low)["sharpness"]
    assert abs(s_low / s_full - 1.0) < 0.15
    blurred = cv2.GaussianBlur(img, (0, 0), 1.5)
    assert S.frame_stats(blurred)["sharpness"] < 0.5 * s_full
    assert math.isnan(S.frame_stats(np.full((20, 20), 90, np.uint8))["sharpness"])


def test_clip_flicker_motion_and_interval():
    base = _scene(seed=3) * 0.5 + 40.0                      # mean about 100, far from clipping
    gains = np.array([1.0, 1.1, 0.9])
    frames = np.stack([_u8(base * g) for g in gains])
    st = S.clip_stats(frames, [0.0, 5.0, 10.0])
    assert st["flicker"] == pytest.approx(gains.std(ddof=1) / gains.mean(), abs=2e-3)
    assert st["interval_s"] == 5.0 and st["n_frames"] == 3
    # A rigid circular shift of (3, -2) px per frame: phase correlation recovers its length.
    img = _noisy(2.0)
    rolled = np.stack([np.roll(np.roll(img, 3 * k, axis=1), -2 * k, axis=0) for k in range(4)])
    st = S.clip_stats(rolled, [0.0, 5.0, 10.0, 15.0])
    assert st["motion_px"] == pytest.approx(math.hypot(3, 2), abs=0.1)
    assert st["flicker"] < 1e-9
    one = S.clip_stats(img[None])
    assert math.isnan(one["flicker"]) and math.isnan(one["motion_px"]) and math.isfinite(one["noise_sigma"])


def test_resize_keeps_aspect_and_matches_pixel_count():
    frames = np.zeros((2, 288, 352), np.uint8)
    small, scale = S.resize_to_pixels(frames, (192, 192))
    h, w = small.shape[1:]
    assert abs(h * w / 192 ** 2 - 1.0) < 0.01 and abs((w / h) / (352 / 288) - 1.0) < 0.01 and scale < 1.0
    same, scale = S.resize_to_pixels(frames[:, :192, :192], (192, 192))
    assert same.shape == (2, 192, 192) and scale == 1.0


def test_sample_times_uses_interval_then_falls_back_to_even():
    t, mode = S.sample_times(0.0, 12.0, 5.0, 12, 3)
    assert mode == "interval" and np.allclose(t, [0.0, 5.0, 10.0])
    t, mode = S.sample_times(0.0, 120.0, 5.0, 4, 3)
    assert mode == "interval" and np.allclose(t, [0.0, 5.0, 10.0, 15.0])       # capped at K
    t, mode = S.sample_times(0.0, 30.0, 60.0, 4, 3)
    assert mode == "even" and np.allclose(t, [0.0, 10.0, 20.0, 30.0])


def _records(real_values, temporal=None):
    temporal = temporal or [True] * len(real_values)
    real = [{"native": {s: v for s in S.STATS}, "resized": {s: v for s in S.STATS}, "temporal_comparable": ok}
            for v, ok in zip(real_values, temporal)]
    synthetic = [{"stats": {s: float(v) for s in S.STATS}, "render_params": {}} for v in range(101)]
    return real, synthetic                                    # synthetic P1 = 1, P99 = 99


def test_compare_coverage_verdicts_and_notes():
    render_cfg = {"noise_sigma": [0.004, 0.035], "contrast": [0.6, 1.3], "exposure_flicker": [0.0, 0.04],
                  "camera_shake_px": [0.0, 2.5], "jpeg_quality": [35, 95], "jpeg_prob": 0.6}
    real, syn = _records([0.5] + [50.0] * 7 + [120.0, 130.0], temporal=[True] * 9 + [False])
    out = S.compare(real, syn, render_cfg, interval_s=5.0)
    noise = out["noise_sigma"]
    assert noise["synthetic"]["p1"] == pytest.approx(1.0) and noise["synthetic"]["p99"] == pytest.approx(99.0)
    r = noise["real_resized"]
    assert (r["n"], r["below"], r["above"]) == (10, pytest.approx(0.1), pytest.approx(0.2))
    assert noise["verdict"] == "widen_both"
    assert out["blockiness"]["comparable_basis"] == "native"
    # The clip without frames at the comparison spacing is left out of flicker and motion only.
    fl = out["flicker"]["real_resized"]
    assert (fl["n"], fl["below"], fl["above"]) == (9, pytest.approx(1 / 9), pytest.approx(1 / 9))
    expect = {"noise_sigma": "render.noise_sigma", "contrast": "render.contrast",
              "brightness": "render.brightness_offset", "sharpness": "render.jpeg_quality",
              "blockiness": "render.jpeg_quality", "flicker": "render.exposure_flicker",
              "motion_px": "render.camera_shake_px"}
    for stat, param in expect.items():
        assert param in out[stat]["note"], (stat, out[stat]["note"])
    assert "[0.004, 0.035]" in noise["note"] and "starts at 0" in out["flicker"]["note"]
    assert "starts at 0" in out["motion_px"]["note"] and "Lower the lower end" not in out["motion_px"]["note"]
    shifted = dict(render_cfg, camera_shake_px=[0.5, 2.5], exposure_flicker=[0.01, 0.04])
    out2 = S.compare(real, syn, shifted, interval_s=5.0)
    assert "Lower the lower end of render.camera_shake_px (now [0.5, 2.5])" in out2["motion_px"]["note"]
    assert "Lower the lower end of render.exposure_flicker (now [0.01, 0.04])" in out2["flicker"]["note"]
    # Inside the range: covered; above only: the high side alone.
    out = S.compare(*_records([20.0, 50.0, 80.0]), render_cfg, interval_s=5.0)
    assert all(e["verdict"] == "covered" and e["note"].startswith("Covered") for e in out.values())
    out = S.compare(*_records([50.0] * 8 + [150.0] * 2), render_cfg, interval_s=5.0)
    assert out["contrast"]["verdict"] == "widen_high" and "Raise the upper end of render.contrast" in \
        out["contrast"]["note"] and "Lower the lower end" not in out["contrast"]["note"]
    out = S.compare(*_records([]), render_cfg, interval_s=5.0)
    assert all(e["verdict"] == "no_data" for e in out.values())


def test_spearman():
    x = np.arange(10.0)
    assert S.spearman(x, x ** 3) == pytest.approx(1.0)
    assert S.spearman(x, -x) == pytest.approx(-1.0)
    assert S.spearman(x, np.ones(10)) is None and S.spearman([1.0, 2.0], [1.0, 2.0]) is None


def test_end_to_end_video_folder_and_synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    texture = _u8(_scene(80, 96, seed=5))
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (96, 80))
    assert writer.isOpened()
    for k in range(120):                                       # 12 s at 10 frame/s
        frame = _u8(np.roll(texture, k // 10, axis=1) + rng.normal(0.0, 2.0, texture.shape))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()
    folder = tmp_path / "folder"
    folder.mkdir()
    for t in (0, 5, 10, 15):                                   # one image every 5 s, time in the filename
        cv2.imwrite(str(folder / f"cam_t{t}s.png"), _u8(_scene(64, 80, seed=6) + rng.normal(0, 3, (64, 80))))
    manifest = tmp_path / "real.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["video_path", "camera_id", "video_id", "smoke", "fog"])
        w.writerow([str(video), "cam_a", "clip", 1, 0])
        w.writerow(["folder", "cam_b", "folder", 0, 0])          # relative to the manifest
        w.writerow([str(tmp_path / "missing.avi"), "cam_c", "missing", 0, 0])

    data = tmp_path / "synthetic"
    (data / "seqs" / "train").mkdir(parents=True)
    (data / "config_used.yaml").write_text("render:\n  native_size: [48, 48]\n  noise_sigma: [0.004, 0.035]\n"
                                           "time:\n  frame_interval_s: 5.0\n")
    for i in range(4):
        (data / "seqs" / "train" / f"seq{i}.npz").write_bytes(b"")

    def fake_load_sequence(path):                              # stand-in for the stored-sequence format
        k = int(Path(path).stem[-1])
        frames = np.stack([_noisy(1.0 + k, seed=10 * k + t, base=_scene(48, 48, seed=k)) for t in range(12)])
        return {"frames": frames, "timestamps": np.arange(12) * 5.0, "seq_id": Path(path).stem,
                "category": "neither", "render_params": {"noise_sigma": 0.004 * (1 + k), "jpeg": False}}

    import src.data.dataset

    monkeypatch.setattr(src.data.dataset, "load_sequence", fake_load_sequence)
    out = tmp_path / "stats.json"
    rc = S.main(["--real-manifest", str(manifest), "--synthetic", str(data), "--out", str(out),
                 "--n-synthetic", "3", "--max-frames", "12"])
    assert rc == 0
    doc = json.loads(out.read_text())
    clips = {c["video_id"]: c for c in doc["real"]["clips"]}
    assert set(clips) == {"clip", "folder"} and len(doc["real"]["skipped"]) == 1
    assert clips["clip"]["sampling"] == "interval" and clips["clip"]["native"]["n_frames"] == 3
    assert clips["clip"]["native"]["interval_s"] == pytest.approx(5.0, abs=0.06)
    assert clips["folder"]["native"]["n_frames"] == 4 and clips["folder"]["temporal_comparable"]
    for c in clips.values():
        h, w = c["resized_size"]
        assert abs(h * w / 48 ** 2 - 1.0) < 0.1
    assert len(doc["synthetic"]["sequences"]) == 3 and doc["synthetic"]["available"] == 4
    assert doc["settings"]["interval_s"] == 5.0 and doc["settings"]["synthetic_size"] == [48, 48]
    assert set(doc["stats"]) == set(S.STATS)
    assert doc["synthetic_tracking"]["noise_sigma~noise_sigma"] == pytest.approx(1.0)
    md = out.with_suffix(".md").read_text()
    assert "## What to change in `render:`" in md and "missing.avi" in md
