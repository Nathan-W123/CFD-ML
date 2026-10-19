"""Tests for src/data/features.py: the shared synthetic/real motion-feature extractor.

Every scene is built here with numpy/cv2: textured blobs (smoothed noise
sampled in the blob's own coordinates, so the texture moves, scales and
rotates with it) over a textured static background, with known motion
between each lag frame and its observation. Expected flow values follow from
the contract: displacement / L / lag, L = sqrt(H * W) of the resized ROI,
so for a square ROI of native side S the expected value is d_native / S / lag.
"""

from __future__ import annotations

import csv
import math
import time

import cv2
import numpy as np
import pytest

from src.data.feature_spec import FEATURE_INDEX, FEATURE_NAMES, N_FEATURES, gate_masks
from src.data.features import (
    FeatureConfig,
    IncompatibleFrameTiming,
    extract_features,
    extract_features_with_meta,
    load_frames,
    select_window,
    stabilize_sequence,
)
from src.utils.io import dumps_json

H = W = 192
T = 12
LAG_S = 1.0
TS = np.arange(T) * 5.0
FI = FEATURE_INDEX
PAIR_FEATURES = [FI[k] for k in ("area_growth", "centroid_dx", "centroid_dz", "anchor_stability")]
FLOW_FEATURES = [FI[k] for k in ("flow_u_mean", "flow_v_mean", "flow_u_std", "flow_v_std",
                                 "flow_divergence", "flow_vorticity_abs")]


# ---------------------------------------------------------------------------
# scene builders
# ---------------------------------------------------------------------------


def _texture(rng, size=512, sigma=3.0):
    n = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), sigma)
    return n / n.std()


def _background(rng, h=H, w=W, level=0.35, contrast=0.08, sigma=2.0):
    t = _texture(rng, max(h, w) + 8, sigma)[:h, :w]
    return np.clip(level + contrast * t, 0, 1).astype(np.float32)


def _blob(center, radius, tex, background, angle=0.0, tex_scale=1.0, brightness=0.65, contrast=0.12):
    """Textured soft disc; the texture is attached to the disc (moves, rotates, scales with it)."""
    h, w = background.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dx, dy = xx - center[0], yy - center[1]
    ca, sa = math.cos(-angle), math.sin(-angle)
    half = tex.shape[0] / 2.0
    lx = ((ca * dx - sa * dy) / tex_scale + half).astype(np.float32)
    ly = ((sa * dx + ca * dy) / tex_scale + half).astype(np.float32)
    tx = cv2.remap(tex, lx, ly, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    r = np.hypot(dx, dy)
    alpha = np.clip((radius - r) / 1.5 + 0.5, 0.0, 1.0).astype(np.float32)
    img = (1 - alpha) * background + alpha * (brightness + contrast * tx)
    return np.clip(img, 0, 1).astype(np.float32), r <= radius


def _u8(x):
    return np.rint(np.clip(np.asarray(x), 0, 1) * 255).astype(np.uint8)


def _blob_sequence(rng, obs_center, lag_center, radius=24.0, obs_angle=None, lag_angle=None,
                   obs_scale=None, lag_scale=None, obs_radius=None, lag_radius=None, background=None):
    """frames, lag_frames (uint8), masks for per-observation blob states given as callables of t."""
    tex = _texture(rng)
    bg = _background(rng) if background is None else background
    frames, lags, masks = [], [], []
    for t in range(T):
        kw_o = dict(angle=obs_angle(t) if obs_angle else 0.0, tex_scale=obs_scale(t) if obs_scale else 1.0)
        kw_l = dict(angle=lag_angle(t) if lag_angle else 0.0, tex_scale=lag_scale(t) if lag_scale else 1.0)
        f, m = _blob(obs_center(t), obs_radius(t) if obs_radius else radius, tex, bg, **kw_o)
        lf, _ = _blob(lag_center(t), lag_radius(t) if lag_radius else radius, tex, bg, **kw_l)
        frames.append(f)
        lags.append(lf)
        masks.append(m)
    return _u8(np.stack(frames)), _u8(np.stack(lags)), np.stack(masks)


def _extract(frames, masks, lags=None, cfg=None):
    return extract_features_with_meta(frames, TS, masks, lags, None if lags is None else TS - LAG_S, cfg)


def _roi_side(meta):
    x0, y0, x1, y1 = meta["roi"]["box"]
    assert x1 - x0 == y1 - y0, "test scenes are built so the ROI stays square"
    return float(x1 - x0)


def _native_px(value, meta, dt):
    """Feature value (L-units / s) back to native pixel displacement over dt."""
    return value * meta["roi"]["L_px"] * dt / meta["roi"]["work_px_per_native_px"]


def _shake(img, M, nearest=False):
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    border = cv2.BORDER_CONSTANT if nearest else cv2.BORDER_REFLECT
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=flags, borderMode=border)


def _random_shake(rng, std_px=2.0, rot_deg=0.3):
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), float(rng.uniform(-rot_deg, rot_deg)), 1.0)
    M[:, 2] += np.clip(rng.normal(0.0, std_px, 2), -3 * std_px, 3 * std_px)
    return M


# ---------------------------------------------------------------------------
# shape, gates, validity
# ---------------------------------------------------------------------------


def test_random_inputs_give_finite_t_by_14():
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, (T, H, W), dtype=np.uint8)
    lags = rng.integers(0, 256, (T, H, W), dtype=np.uint8)
    noise = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 8)
    masks = np.stack([np.roll(noise, 3 * t, axis=1) > 0.5 * noise.std() for t in range(T)])
    for lag in (lags, None):
        X = extract_features(frames, TS, masks, lag, None if lag is None else TS - LAG_S)
        assert X.shape == (T, N_FEATURES) == (12, 14)
        assert X.dtype == np.float32
        assert np.all(np.isfinite(X))
        assert np.all(X[~gate_masks(X)] == 0.0)
    # random float frames and random (noisy) masks, too
    Xf = extract_features(rng.random((T, H, W)), TS, rng.random((T, H, W)) > 0.7, rng.random((T, H, W)))
    assert Xf.shape == (12, 14) and np.all(np.isfinite(Xf))


def test_empty_masks_close_every_gate():
    rng = np.random.default_rng(1)
    frames, lags, _ = _blob_sequence(rng, lambda t: (96, 96), lambda t: (92, 96))
    X, meta = _extract(frames, np.zeros((T, H, W), bool), lags)
    assert np.all(X[:, FI["mask_valid"]] == 0.0)
    assert np.all(X[:, FI["flow_valid"]] == 0.0)
    assert np.all(X == 0.0)                      # every feature is gated by mask or flow validity
    assert meta["roi"]["reason"] == "full_no_mask"
    X_none = extract_features(frames, TS, None, lags, TS - LAG_S)
    assert np.all(X_none == 0.0)


def test_first_observation_has_no_mask_pair_features():
    rng = np.random.default_rng(2)
    frames, lags, masks = _blob_sequence(rng, lambda t: (70 + 4 * t, 96), lambda t: (70 + 4 * t - 0.8, 96))
    X, meta = _extract(frames, masks, lags)
    assert X[0, FI["mask_valid"]] == 1.0 and X[0, FI["flow_valid"]] == 1.0   # lag frame gives t=0 a flow
    assert np.all(X[0, PAIR_FEATURES] == 0.0)
    assert np.all(X[1:, FI["mask_valid"]] == 1.0)
    assert np.all(X[1:, FI["centroid_dx"]] > 0)
    assert meta["counts"] == {"mask_valid": 12, "flow_valid": 12, "mask_pair_valid": 11}
    dumps_json(meta)                              # meta is JSON-able (finite numbers only)


# ---------------------------------------------------------------------------
# flow magnitude, sign and units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dx", [4.0, -3.5])
def test_rigid_translation_recovered_within_10_percent(dx):
    rng = np.random.default_rng(3)
    frames, lags, masks = _blob_sequence(rng, lambda t: (96, 96), lambda t: (96 - dx, 96))
    X, meta = _extract(frames, masks, lags)
    expected = dx / _roi_side(meta) / LAG_S      # px / L / s with L = work_size = 128 px of a side-S ROI
    assert np.all(X[:, FI["flow_valid"]] == 1.0)
    u = X[:, FI["flow_u_mean"]]
    assert np.all(np.sign(u) == np.sign(dx))
    np.testing.assert_allclose(u, expected, rtol=0.10)
    assert np.all(np.abs(X[:, FI["flow_v_mean"]]) < 0.1 * abs(expected))
    # rates are per second: the same displacement over half the lag is twice as fast
    X_half = extract_features(frames, TS, masks, lags, TS - 0.5 * LAG_S)
    np.testing.assert_allclose(X_half[:, FLOW_FEATURES], 2 * X[:, FLOW_FEATURES], rtol=1e-5, atol=1e-7)


def test_upward_motion_is_positive_flow_v_and_centroid_dz():
    rng = np.random.default_rng(4)
    speed = 1.2                                   # native px / s, upward (rows decrease)
    obs = lambda t: (96.0, 160.0 - speed * TS[t])
    lag = lambda t: (96.0, 160.0 - speed * (TS[t] - LAG_S))
    frames, lags, masks = _blob_sequence(rng, obs, lag, radius=20.0)
    X, meta = _extract(frames, masks, lags)
    expected = speed / _roi_side(meta)
    v = X[:, FI["flow_v_mean"]]
    dz = X[1:, FI["centroid_dz"]]
    assert np.all(v > 0) and np.all(dz > 0)
    # small object in a full-frame ROI: DIS smooths across the static edge, hence the looser bound
    np.testing.assert_allclose(np.median(v), expected, rtol=0.25)
    np.testing.assert_allclose(np.median(dz), expected, rtol=0.10)
    assert np.all(np.abs(X[1:, FI["centroid_dx"]]) < 0.1 * expected)


def test_uint8_and_float_inputs_agree():
    rng = np.random.default_rng(5)
    frames, lags, masks = _blob_sequence(rng, lambda t: (90 + 2 * t, 100), lambda t: (88.5 + 2 * t, 100))
    X8 = extract_features(frames, TS, masks, lags, TS - LAG_S)
    Xf = extract_features(frames / 255.0, TS, masks.astype(np.float32), lags.astype(np.float64) / 255.0, TS - LAG_S)
    np.testing.assert_allclose(Xf, X8, atol=1e-3)
    # 3-channel BGR input converts to the same grayscale
    bgr = np.repeat(frames[..., None], 3, axis=-1)
    Xc = extract_features(bgr, TS, masks, np.repeat(lags[..., None], 3, axis=-1), TS - LAG_S)
    np.testing.assert_allclose(Xc, X8, atol=1e-3)


def test_consecutive_mode_without_lag_frames():
    rng = np.random.default_rng(6)
    frames, _, masks = _blob_sequence(rng, lambda t: (60 + 6 * t, 96), lambda t: (0, 0))
    X, meta = _extract(frames, masks, None)
    assert meta["flow_mode"] == "consecutive"
    assert "consecutive_flow" in meta["timing"]["reasons"]
    assert X[0, FI["flow_valid"]] == 0.0
    assert X[1:, FI["flow_valid"]].sum() >= 9
    ok = X[:, FI["flow_valid"]] == 1.0
    expected = 6.0 / _roi_side(meta) / 5.0
    np.testing.assert_allclose(np.median(X[ok, FI["flow_u_mean"]]), expected, rtol=0.15)


# ---------------------------------------------------------------------------
# stabilisation
# ---------------------------------------------------------------------------


def test_camera_shake_is_removed_by_stabilisation():
    rng = np.random.default_rng(7)
    frames, lags, masks = _blob_sequence(rng, lambda t: (96, 96), lambda t: (96, 96), radius=26.0)
    sf, sl, sm = [], [], []
    for t in range(T):
        Mo, Ml = _random_shake(rng), _random_shake(rng)
        sf.append(_shake(frames[t], Mo))
        sm.append(_shake(masks[t].astype(np.uint8), Mo, nearest=True) > 0)   # masks live in shaken coordinates
        sl.append(_shake(lags[t], Ml))
    sf, sl, sm = np.stack(sf), np.stack(sl), np.stack(sm)

    X, meta = _extract(sf, sm, sl)
    assert meta["stabilization"]["applied"]
    assert np.all(X[:, FI["flow_valid"]] == 1.0)
    for name in ("flow_u_mean", "flow_v_mean"):
        resid = np.abs(_native_px(X[:, FI[name]], meta, LAG_S))
        assert resid.max() < 0.2, (name, resid)

    # Without stabilisation the same clip shows the shake as motion (the test has teeth).
    X0, meta0 = _extract(sf, sm, sl, FeatureConfig(stabilize=False))
    ok = X0[:, FI["flow_valid"]] == 1.0
    raw = np.hypot(_native_px(X0[ok, FI["flow_u_mean"]], meta0, LAG_S),
                   _native_px(X0[ok, FI["flow_v_mean"]], meta0, LAG_S))
    assert (~ok).any() or np.median(raw) > 1.0

    # The public helper aligns frames onto observation 0.
    out_f, out_m, out_l, info = stabilize_sequence(sf, sm, sl)
    assert out_f.shape == sf.shape and out_m.shape == sm.shape and out_l.shape == sl.shape
    assert info["applied"] and all(info["obs_ok"]) and all(info["lag_ok"])
    err_before = np.abs(sf[1:].astype(float) - sf[:1]).mean()
    err_after = np.abs(out_f[1:, 30:-30, 30:-30].astype(float) - out_f[:1, 30:-30, 30:-30]).mean()
    assert err_after < 0.35 * err_before


def _fog_bank(rng, speed, ground_rows=0):
    """A textured fog field translating right at `speed` px/s over the whole frame (above `ground_rows`)."""
    tex = _texture(rng, 1024, 4.0)
    ground = _background(rng, contrast=0.15, sigma=1.5)
    frames, lags = [], []
    for t in range(T):
        for store, time_s in ((frames, TS[t]), (lags, TS[t] - LAG_S)):
            shift = speed * time_s
            M = np.float32([[1, 0, 300 + shift], [0, 1, 300]])          # moves the texture right
            fog = 0.6 + 0.1 * cv2.warpAffine(tex, M, (W, H), flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_REFLECT)
            img = fog.copy()
            if ground_rows:
                img[H - ground_rows:] = ground[H - ground_rows:]
            store.append(img)
    masks = np.zeros((T, H, W), bool)
    masks[:, : H - ground_rows] = True
    return _u8(np.stack(frames)), _u8(np.stack(lags)), masks


def test_frame_filling_fog_bank_keeps_its_motion():
    rng = np.random.default_rng(8)
    speed = 1.5
    frames, lags, masks = _fog_bank(rng, speed)
    X, meta = _extract(frames, masks, lags)
    assert not meta["stabilization"]["applied"]
    assert meta["stabilization"]["reason"] == "insufficient_textured_background"
    # Camera motion cannot be separated from the fog's: by default the flow gate closes ...
    assert np.all(X[:, FI["flow_valid"]] == 0.0)
    assert set(meta["flow"]["status"]) == {"stabilization_unavailable"}
    assert np.all(X[:, :6] == 0.0)
    # ... and a footage source declared static (rigid mount) keeps the motion.
    X, meta = _extract(frames, masks, lags, FeatureConfig(require_stabilization=False))
    assert np.all(X[:, FI["flow_valid"]] == 1.0)
    expected = speed / _roi_side(meta) / LAG_S
    np.testing.assert_allclose(X[:, FI["flow_u_mean"]], expected, rtol=0.10)

    # Fog over a static textured ground band: stabilisation locks onto the ground
    # and the fog keeps its motion (it is not "stabilised away").
    frames, lags, masks = _fog_bank(rng, speed, ground_rows=72)
    X, meta = _extract(frames, masks, lags)
    assert meta["stabilization"]["applied"]
    ok = X[:, FI["flow_valid"]] == 1.0
    assert ok.sum() >= 10
    expected = speed * meta["roi"]["work_px_per_native_px"] / meta["roi"]["L_px"]
    np.testing.assert_allclose(X[ok, FI["flow_u_mean"]], expected, rtol=0.10)

    # A detector that covers only part of the fog bank leaves moving fog as the
    # "background": its steady drift (or untrackable motion) is recognised and the
    # step is skipped instead of aligning to the fog. The fog's motion is never
    # stabilised away: the flow is closed by default, kept for a static mount.
    for v in (0.5, speed):
        frames, lags, _ = _fog_bank(rng, v)
        partial = np.zeros((T, H, W), bool)
        partial[:, 60:130, 50:140] = True
        X, meta = _extract(frames, partial, lags)
        assert not meta["stabilization"]["applied"]
        assert meta["stabilization"]["reason"] in ("background_moving", "unreliable_background")
        assert np.all(X[:, FI["flow_valid"]] == 0.0)
        X, meta = _extract(frames, partial, lags, FeatureConfig(require_stabilization=False))
        assert np.all(X[:, FI["flow_valid"]] == 1.0)
        np.testing.assert_allclose(X[:, FI["flow_u_mean"]], v / _roi_side(meta) / LAG_S, rtol=0.10)


# ---------------------------------------------------------------------------
# divergence, vorticity, masks and geometry
# ---------------------------------------------------------------------------


def test_expanding_blob_has_positive_divergence_and_area_growth():
    rng = np.random.default_rng(9)
    rate = 0.04                                   # relative expansion per second
    r_obs = lambda t: 14.0 * (1 + rate) ** TS[t]
    r_lag = lambda t: 14.0 * (1 + rate) ** (TS[t] - LAG_S)
    frames, lags, masks = _blob_sequence(
        rng, lambda t: (96, 96), lambda t: (96, 96), obs_radius=r_obs, lag_radius=r_lag,
        obs_scale=lambda t: r_obs(t) / 14.0, lag_scale=lambda t: r_lag(t) / 14.0)
    # A static camera over an untextured background: nothing to stabilise against.
    X, meta = _extract(frames, masks, lags, FeatureConfig(require_stabilization=False))
    ok = X[:, FI["flow_valid"]] == 1.0
    assert ok.sum() >= 10
    div = X[ok, FI["flow_divergence"]]
    expected_div = 2 * (1 - 1 / (1 + rate)) / LAG_S          # displacement field (s-1)/s (x - c), both axes
    assert np.all(div > 0)
    np.testing.assert_allclose(np.median(div), expected_div, rtol=0.3)
    growth = X[1:, FI["area_growth"]]
    assert np.all(growth > 0)
    np.testing.assert_allclose(np.median(growth), 2 * math.log(1 + rate), rtol=0.2)


def test_rotating_disc_has_more_vorticity_than_translating_disc():
    rng = np.random.default_rng(10)
    omega = 0.08                                  # rad / s
    rot = _blob_sequence(rng, lambda t: (96, 96), lambda t: (96, 96), radius=28.0,
                         obs_angle=lambda t: omega * TS[t], lag_angle=lambda t: omega * (TS[t] - LAG_S))
    trans = _blob_sequence(rng, lambda t: (96, 96), lambda t: (96 - 2.5, 96), radius=28.0)
    Xr, _ = _extract(rot[0], rot[2], rot[1])
    Xt, _ = _extract(trans[0], trans[2], trans[1])
    assert np.all(Xr[:, FI["flow_valid"]] == 1.0) and np.all(Xt[:, FI["flow_valid"]] == 1.0)
    vr = np.median(Xr[:, FI["flow_vorticity_abs"]])
    vt = np.median(Xt[:, FI["flow_vorticity_abs"]])
    assert vr > 3 * vt
    # |curl| of a rigid rotation is 2 omega; DIS smooths the static rim of the disc, which
    # pulls the region mean down, so only the order of magnitude is checked here.
    assert 0.3 * 2 * omega < vr < 1.3 * 2 * omega
    assert abs(np.median(Xr[:, FI["flow_divergence"]])) < 0.25 * vr


def test_persistent_filter_drops_single_frame_false_positive():
    rng = np.random.default_rng(11)
    frames, lags, masks = _blob_sequence(rng, lambda t: (70 + 3 * t, 110), lambda t: (69 + 3 * t, 110))
    X_ref, meta_ref = _extract(frames, masks, lags)
    noisy = masks.copy()
    yy, xx = np.mgrid[0:H, 0:W]
    noisy[5] |= np.hypot(xx - 160, yy - 25) < 9              # one-frame detector false positive
    X, meta = _extract(frames, noisy, lags)
    assert meta["persistent_filter"]["removed_components"][5] == 1
    assert sum(meta["persistent_filter"]["removed_components"]) == 1
    assert meta["roi"]["box"] == meta_ref["roi"]["box"]
    np.testing.assert_allclose(X, X_ref, atol=1e-6)

    # Two persistent objects (e.g. smoke and fog) are both kept.
    second = masks.copy()
    second[:, 150:175, 20:60] = True
    _, meta2 = _extract(frames, second, lags)
    assert sum(meta2["persistent_filter"]["removed_components"]) == 0


def test_flow_features_invariant_to_apparent_scene_scale():
    feats = []
    for scale in (1.0, 2.0):
        rng = np.random.default_rng(12)                       # same texture for both scales
        frames, lags, masks = _blob_sequence(
            rng, lambda t: (96, 96), lambda t: (96 - 1.6 * scale, 96 + 0.8 * scale), radius=12.0 * scale,
            obs_scale=lambda t: scale, lag_scale=lambda t: scale)
        X, meta = _extract(frames, masks, lags)
        assert np.all(X[:, FI["flow_valid"]] == 1.0)
        feats.append(np.median(X, axis=0))
    small, large = feats
    for name in ("flow_u_mean", "flow_v_mean", "region_area"):
        np.testing.assert_allclose(small[FI[name]], large[FI[name]], rtol=0.2, err_msg=name)


def test_anchor_stability_plume_vs_drifting_patch():
    rng = np.random.default_rng(13)
    bg = _background(rng)
    tex = _texture(rng)
    yy, xx = np.mgrid[0:H, 0:W]

    def render(mask_fn):
        frames, lags, masks = [], [], []
        for t in range(T):
            m = mask_fn(t)
            alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)
            img = (1 - alpha) * bg + alpha * (0.65 + 0.12 * tex[:H, :W])
            frames.append(img)
            lags.append(img)
            masks.append(m)
        return _u8(np.stack(frames)), _u8(np.stack(lags)), np.stack(masks)

    # plume: fixed base at row 165, a column growing upward and widening with height
    def plume(t):
        top = 150 - 8 * t
        half_width = 7 + 0.09 * np.clip(165 - yy, 0, None)
        return (yy <= 165) & (yy >= top) & (np.abs(xx - 96) <= half_width)

    # patch: an ellipse drifting sideways at constant height
    def patch(t):
        return ((xx - (40 + 10 * t)) / 18.0) ** 2 + ((yy - 120) / 10.0) ** 2 <= 1.0

    frames, lags, masks = render(plume)
    Xp, _ = _extract(frames, masks, lags)
    frames, lags, masks = render(patch)
    Xd, _ = _extract(frames, masks, lags)
    a_plume = Xp[1:, FI["anchor_stability"]]
    a_patch = Xd[1:, FI["anchor_stability"]]
    assert np.all(Xp[1:, FI["mask_valid"]] == 1.0) and np.all(Xd[1:, FI["mask_valid"]] == 1.0)
    assert np.all((a_plume >= 0) & (a_plume <= 1)) and np.all((a_patch >= 0) & (a_patch <= 1))
    assert a_plume.mean() > 0.8
    assert a_plume.mean() > a_patch.mean() + 0.3


def test_roughness_zero_for_disc_and_positive_for_ragged_region():
    rng = np.random.default_rng(14)
    frames, lags, masks = _blob_sequence(rng, lambda t: (96, 96), lambda t: (95, 96), radius=22.0)
    X, _ = _extract(frames, masks, lags)
    assert np.all(X[:, FI["boundary_roughness"]] < 0.05)
    yy, xx = np.mgrid[0:H, 0:W]
    ang = np.arctan2(yy - 96, xx - 96)
    ragged = np.hypot(xx - 96, yy - 96) < 22 * (1 + 0.3 * np.sin(9 * ang))
    Xr, _ = _extract(frames, np.repeat(ragged[None], T, axis=0), lags)
    assert np.all(Xr[:, FI["boundary_roughness"]] > 0.15)


# ---------------------------------------------------------------------------
# timing, loading, performance
# ---------------------------------------------------------------------------


def test_select_window_refuses_one_minute_imagery():
    with pytest.raises(IncompatibleFrameTiming):
        select_window(np.arange(30) * 60.0)                   # HPWREN / FIgLib style
    with pytest.raises(IncompatibleFrameTiming):
        select_window(np.arange(0, 30, 0.1))                  # too short for 12 x 5 s
    with pytest.raises(IncompatibleFrameTiming):
        select_window(np.r_[np.arange(0, 40, 0.1), np.arange(39.9, 80, 0.1)])  # a repeated frame time
    with pytest.raises(IncompatibleFrameTiming):
        select_window(np.arange(0, 80, 0.1), interval_s=0.04)  # observations would repeat indices
    gap = np.r_[np.arange(0, 20, 0.1), np.arange(30, 80, 0.1)]
    with pytest.raises(IncompatibleFrameTiming):
        select_window(gap)                                    # no frame near the 21 s target


def test_select_window_accepts_10fps_video():
    ts = np.arange(0, 70, 0.1)
    sel = select_window(ts)
    assert sel["obs_idx"].shape == (12,) and sel["lag_idx"].shape == (12,)
    assert np.all(np.diff(sel["obs_idx"]) == 50)
    assert np.all(sel["obs_idx"] - sel["lag_idx"] == 10)
    assert np.allclose(np.diff(ts[sel["obs_idx"]]), 5.0)
    assert np.allclose(ts[sel["obs_idx"]] - ts[sel["lag_idx"]], 1.0)
    assert np.all(np.abs(sel["offsets"]) < 0.06) and abs(sel["native_dt"] - 0.1) < 1e-9
    # jittered (variable frame rate) timestamps are fine too
    jit = ts + np.random.default_rng(0).uniform(-0.02, 0.02, ts.size)
    sel = select_window(jit, start=2.0)
    assert np.all(np.abs(sel["offsets"]) <= 0.5)


def test_load_frames_video_and_folder(tmp_path):
    rng = np.random.default_rng(15)
    imgs = [_u8(_background(rng, 48, 64, level=0.2 + 0.02 * i, contrast=0.05)) for i in range(30)]
    video = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48), True)
    assert writer.isOpened()
    for im in imgs:
        writer.write(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR))
    writer.release()
    frames, ts = load_frames(video)
    assert frames.shape == (30, 48, 64) and frames.dtype == np.uint8
    np.testing.assert_allclose(ts, np.arange(30) * 0.1, atol=1e-6)
    sub, sub_ts = load_frames(video, times=[0.52, 1.0, 2.94])
    np.testing.assert_allclose(sub_ts, [0.5, 1.0, 2.9], atol=1e-6)
    np.testing.assert_array_equal(sub, frames[[5, 10, 29]])

    folder = tmp_path / "frames"
    folder.mkdir()
    for i, im in enumerate(imgs[:5]):
        cv2.imwrite(str(folder / f"img_{i:03d}.png"), im)
    with pytest.raises(IncompatibleFrameTiming):
        load_frames(folder)                                   # images without timestamps are refused
    with open(folder / "timestamps.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["filename", "timestamp_s"])
        for i in range(5):
            wr.writerow([f"img_{i:03d}.png", 100.0 + 0.25 * i])
    frames, ts = load_frames(folder)
    np.testing.assert_allclose(ts, 100.0 + 0.25 * np.arange(5))
    np.testing.assert_array_equal(frames[2], imgs[2])
    one, one_ts = load_frames(folder, times=[100.6])
    assert one.shape == (1, 48, 64) and one_ts[0] == pytest.approx(100.5)


def test_extraction_time_per_sequence():
    """12 observations + 12 lag frames at 192 x 192 (target < 60 ms on one core; lenient bound here)."""
    rng = np.random.default_rng(16)
    frames, lags, masks = _blob_sequence(rng, lambda t: (80 + 2 * t, 110 - 2 * t),
                                         lambda t: (79 + 2 * t, 110.5 - 2 * t), radius=26.0)
    extract_features(frames, TS, masks, lags, TS - LAG_S)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        extract_features(frames, TS, masks, lags, TS - LAG_S)
        times.append(time.perf_counter() - t0)
    median_ms = 1e3 * float(np.median(times))
    print(f"feature extraction: {median_ms:.1f} ms per sequence (median of 5)")
    assert median_ms < 250.0        # generous: shared CI machines; the measured value is reported


def test_feature_layout_comes_from_feature_spec():
    import src.data.feature_spec as spec
    import src.data.features as features

    assert features.FEATURE_INDEX is spec.FEATURE_INDEX and features.FEATURE_GATES is spec.FEATURE_GATES
    assert len(FEATURE_NAMES) == N_FEATURES == 14
    assert FeatureConfig().work_size == 128 and FeatureConfig().flow_method == "dis"


def test_timing_compatibility_is_recorded():
    rng = np.random.default_rng(17)
    frames, lags, masks = _blob_sequence(rng, lambda t: (96, 96), lambda t: (95, 96))
    _, meta = _extract(frames, masks, lags)
    assert meta["timing"]["compatible"] and meta["timing"]["reasons"] == []
    X, meta = extract_features_with_meta(frames, np.arange(T) * 60.0, masks)    # one-minute stills
    assert not meta["timing"]["compatible"]
    assert "observation_interval" in meta["timing"]["reasons"] and "consecutive_flow" in meta["timing"]["reasons"]
    assert X.shape == (12, 14) and np.all(np.isfinite(X))


@pytest.mark.parametrize("case", ["farneback", "roi_full", "single_frame", "wide_frames", "thin_mask"])
def test_unusual_inputs_stay_finite(case):
    rng = np.random.default_rng(18)
    frames, lags, masks = _blob_sequence(rng, lambda t: (90 + 2 * t, 96), lambda t: (89 + 2 * t, 96))
    ts, lag_ts, cfg = TS, TS - LAG_S, None
    if case == "farneback":
        cfg = FeatureConfig(flow_method="farneback")
    elif case == "roi_full":
        cfg = FeatureConfig(roi="full")
    elif case == "single_frame":
        frames, lags, masks, ts, lag_ts = frames[:1], lags[:1], masks[:1], ts[:1], lag_ts[:1]
    elif case == "wide_frames":
        frames = np.stack([cv2.resize(f, (320, 120), interpolation=cv2.INTER_AREA) for f in frames])
        lags = np.stack([cv2.resize(f, (320, 120), interpolation=cv2.INTER_AREA) for f in lags])
        masks = np.stack([cv2.resize(m.astype(np.uint8), (320, 120), interpolation=cv2.INTER_NEAREST) > 0
                          for m in masks])
    elif case == "thin_mask":
        masks = np.zeros_like(masks)
        masks[:, 40:150, 94:99] = True                     # 5 px wide: nothing left after erosion
    X, meta = extract_features_with_meta(frames, ts, masks, lags, lag_ts, cfg)
    assert X.shape == (len(ts), 14) and X.dtype == np.float32 and np.all(np.isfinite(X))
    assert np.all(X[~gate_masks(X)] == 0.0)
    dumps_json(meta)
    if case == "thin_mask":
        assert np.all(X[:, FI["mask_valid"]] == 1.0) and np.all(X[:, FI["flow_valid"]] == 0.0)
        assert set(meta["flow"]["status"]) == {"region_too_small"}
    else:
        assert np.all(X[:, FI["flow_valid"]] == 1.0)
        assert np.all(X[:, FI["flow_u_mean"]] > 0)
