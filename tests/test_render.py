"""Rendering (src/data/render.py) and candidate masks (src/data/masks.py).

SimulationResult fixtures are built analytically (the CFD run is not needed):
a smoke blob translating at a known speed with texture coordinates advected
by the same velocity, a static fog layer, empty fields; `without_lag` turns
any of them into a simulation stored without lag snapshots.
"""

import dataclasses
import math

import cv2
import numpy as np
import pytest

from src.cfd.results import SimulationResult
from src.constants import CATEGORIES, CATEGORY_LABELS
from src.data.masks import boxes_from_masks, degrade_masks, detect_candidate_masks
from src.data.render import (
    MASK_SOURCES, SGS_GAIN_DEFAULT, SHAKE_TAU_S_DEFAULT, RenderedSequence, check_label_visibility,
    render_sequence, sample_render_params,
)
from src.utils.config import load_simulation_config
from src.utils.io import dumps_json
from src.utils.runtime import child_rng

NX = NZ = 64
DX = 5.0
T = 12
TIMES = np.arange(T) * 5.0
LAG_TIMES = TIMES - 1.0
CFG = load_simulation_config()

# Every random visual nuisance off: isolates the optics in a test.
CLEAN = {
    "zoom": [1.0, 1.0], "foreground_band": [0.1, 0.1], "noise_sigma": [0.0, 0.0], "jpeg_prob": 0.0,
    "exposure_flicker": [0.0, 0.0], "camera_shake_px": [0.0, 0.0], "camera_rotation_deg": [0.0, 0.0],
    "occlusion_prob": 0.0, "global_haze": [0.0, 0.0], "contrast": [1.0, 1.0], "brightness_offset": [0.0, 0.0],
    "masks": {"source_probs": {"degraded": 1.0}, "boundary_noise": [0.0, 0.0], "morph_px": [0, 0],
              "drop_component_prob": 0.0, "false_positive_prob": 0.0, "frame_dropout_prob": 0.0,
              "threshold_contrast": [0.03, 0.03]},
}


# ---------------------------------------------------------------------- fixtures
def _grid():
    x = (np.arange(NX) + 0.5) * DX
    z = (np.arange(NZ) + 0.5) * DX
    Z, X = np.meshgrid(z, x, indexing="ij")
    return X, Z


def _texcoords(times, u, w, period=20.0):
    """Two coordinate pairs advected by a uniform (u, w) m/s, reset to the identity on a
    staggered cycle; weights sin^2 / cos^2 of the pair's age so a pair is invisible at its reset."""
    X, Z = _grid()
    tc = np.empty((len(times), 4, NZ, NX))
    wts = np.empty((len(times), 2))
    for i, t in enumerate(times):
        for pair, phase in ((0, 0.25), (1, 0.75)):
            age = (t + phase * period) % period
            tc[i, 2 * pair] = X - u * age
            tc[i, 2 * pair + 1] = Z - w * age
            wts[i, pair] = math.sin(math.pi * age / period) ** 2
    return tc, wts / wts.sum(axis=1, keepdims=True)


def _blob(times, x0, z0, u, w, sx=30.0, sz=25.0, peak=1.0):
    X, Z = _grid()
    return np.stack([peak * np.exp(-0.5 * (((X - x0 - u * t) / sx) ** 2 + ((Z - z0 - w * t) / sz) ** 2))
                     for t in times])


def _fog_layer(times, top=70.0, lwc=0.12):
    X, Z = _grid()
    layer = lwc * 0.5 * (1.0 - np.tanh((Z - top) / 10.0)) * (1.0 + 0.2 * np.sin(2 * np.pi * X / 160.0))
    return np.repeat(layer[None], len(times), axis=0)


def _plume(times, xs=150.0, u=1.5, w_rise=2.5):
    """A steady bent-over plume: only its texture moves (the hard case for optical flow)."""
    X, Z = _grid()
    sig = 6.0 + 0.3 * Z
    xc = xs + u * Z / w_rise
    c = 2.0 * np.sqrt(6.0 / sig) * np.exp(-0.5 * ((X - xc) / sig) ** 2) * 0.5 * (1.0 - np.tanh((Z - 230.0) / 20.0))
    return np.repeat(c[None], len(times), axis=0)


def make_sim(category="smoke_only", smoke=None, lwc=None, tracer=None, smoke_lag=None, lwc_lag=None,
             tracer_lag=None, tex_uv=(0.0, 0.0), texcoords=True, solid=None, params=None):
    """A SimulationResult with analytic fields (zeros where not given)."""
    zeros = np.zeros((T, NZ, NX))
    smoke = zeros if smoke is None else smoke
    lwc = zeros if lwc is None else lwc
    tracer = zeros if tracer is None else tracer
    tc, wts = _texcoords(TIMES, *tex_uv)
    tcl, wtsl = _texcoords(LAG_TIMES, *tex_uv)
    p = {"render_hints": {}, "distractor_regime": None}
    p.update(params or {})
    return SimulationResult(
        sim_id=f"test-{category}", category=category, regime="test", labels=CATEGORY_LABELS[category],
        seed=0, params=p, dx=DX, times=TIMES.copy(),
        smoke=smoke.astype(np.float32), lwc=lwc.astype(np.float32), tracer=tracer.astype(np.float32),
        solid=np.zeros((NZ, NX), bool) if solid is None else solid, valid=True, validity={},
        texcoords=tc if texcoords else None, tex_weights=wts if texcoords else None,
        lag_times=LAG_TIMES.copy(),
        smoke_lag=(zeros if smoke_lag is None else smoke_lag).astype(np.float32),
        lwc_lag=(zeros if lwc_lag is None else lwc_lag).astype(np.float32),
        tracer_lag=(zeros if tracer_lag is None else tracer_lag).astype(np.float32),
        texcoords_lag=tcl if texcoords else None, tex_weights_lag=wtsl if texcoords else None,
    )


def smoke_blob_sim(u=4.0, w=0.0, peak=1.0, category="smoke_only", **kw):
    return make_sim(category, smoke=_blob(TIMES, 120.0, 90.0, u, w, peak=peak),
                    smoke_lag=_blob(LAG_TIMES, 120.0, 90.0, u, w, peak=peak), tex_uv=(u, w), **kw)


def fog_layer_sim(u=1.0, category="fog_only", **kw):
    return make_sim(category, lwc=_fog_layer(TIMES), lwc_lag=_fog_layer(LAG_TIMES), tex_uv=(u, 0.0), **kw)


def plume_sim(u=1.5, w=2.5):
    return make_sim("smoke_only", smoke=_plume(TIMES, u=u, w_rise=w), smoke_lag=_plume(LAG_TIMES, u=u, w_rise=w),
                    tex_uv=(u, w))


def without_lag(sim):
    """The same simulation stored without lag snapshots (flow_lag_s null / 0)."""
    return dataclasses.replace(sim, lag_times=None, smoke_lag=None, lwc_lag=None, tracer_lag=None,
                               texcoords_lag=None, tex_weights_lag=None, sgs_u_lag=None)


def with_sgs(sim, u_obs, u_lag=None):
    """Attach a subgrid velocity field u_sgs (m/s; a scalar means uniform) at the observation and lag times."""
    full = lambda u: np.broadcast_to(np.asarray(u, np.float32), (T, NZ, NX)).copy()
    sim.sgs_u = full(u_obs)
    sim.sgs_u_lag = None if u_lag is None else full(u_lag)
    return sim


def _params(seed=0, overrides=None, hints=None):
    return sample_render_params(np.random.default_rng(seed), CFG, hints, overrides)


def _clean(**extra):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CLEAN.items()}
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------- render
def test_shapes_dtypes_and_json_params():
    sim = smoke_blob_sim()
    rp = _params(1)
    out = render_sequence(sim, rp, np.random.default_rng(2))
    H, W = CFG["render"]["native_size"]
    assert isinstance(out, RenderedSequence)
    assert out.frames.shape == (T, H, W) and out.frames.dtype == np.uint8
    assert out.lag_frames.shape == (T, H, W) and out.lag_frames.dtype == np.uint8
    assert out.masks.shape == (T, H, W) and out.masks.dtype == bool
    assert out.opacity.shape == (T, H, W) and out.opacity.dtype == np.float32
    assert set(out.component_opacity) == {"smoke", "fog", "tracer", "cloud"}
    assert all(v.shape == (T, H, W) and v.dtype == np.float16 for v in out.component_opacity.values())
    np.testing.assert_array_equal(out.timestamps, TIMES)
    np.testing.assert_array_equal(out.lag_timestamps, LAG_TIMES)
    assert out.mask_source in MASK_SOURCES
    dumps_json(rp)                      # JSON-able params ...
    dumps_json(out.visibility)          # ... and statistics
    assert len(out.visibility["peak_opacity"]["smoke"]) == T


def test_render_is_deterministic_given_params_and_seed():
    sim = smoke_blob_sim()
    hints = {"shadow": True, "strong_artifact": True, "fp_boost": True, "distant_cloud": True}
    rp = _params(3, hints=hints)
    a = render_sequence(sim, rp, np.random.default_rng(7))
    b = render_sequence(sim, rp, np.random.default_rng(7))
    c = render_sequence(sim, rp, np.random.default_rng(8))
    np.testing.assert_array_equal(a.frames, b.frames)
    np.testing.assert_array_equal(a.lag_frames, b.lag_frames)
    np.testing.assert_array_equal(a.masks, b.masks)
    assert a.visibility == b.visibility
    assert not np.array_equal(a.frames, c.frames)


def _dis_flow(prev, nxt):
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return dis.calc(prev, nxt, None)


def test_texture_moves_with_coordinate_field_on_a_static_layer():
    """A quasi-steady concentration field only shows motion through the advected texture."""
    u = 5.0
    rp = _params(4, _clean(texture={"gain": [0.6, 0.6]}, peak_opacity=[0.95, 0.95]))
    s = None
    flows = {}
    for name, uv in (("moving", (u, 0.0)), ("still", (0.0, 0.0))):
        sim = make_sim("fog_only", lwc=_fog_layer(TIMES, top=150.0), lwc_lag=_fog_layer(LAG_TIMES, top=150.0),
                       tex_uv=uv)
        out = render_sequence(sim, rp, np.random.default_rng(0))
        s = out.visibility["metres_per_pixel"]
        # Dense fog only: where it is thin, the static terrain behind it shows through and
        # (correctly) pulls the apparent motion towards zero.
        fog = cv2.blur(out.component_opacity["fog"][5].astype(np.float32), (9, 9)) > 0.85
        region = cv2.erode(fog.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        assert region.sum() > 1000
        flow = _dis_flow(out.lag_frames[5], out.frames[5])
        flows[name] = (np.median(flow[..., 0][region]), np.median(flow[..., 1][region]))
        if name == "moving":
            # Exact texture shift between observations 5 and 6 (5 s, pair a dominant in both):
            # the texture factor is additive in log optical depth; no background is involved.
            op = out.component_opacity["fog"].astype(np.float32)
            log_tau = np.log(-np.log(np.clip(1.0 - op[5:7], 1e-3, 1.0)) + 1e-3)
            win = (slice(60, 150), slice(20, 172))
            (dx, dy), _ = cv2.phaseCorrelate(log_tau[0][win], log_tau[1][win])
    expected = u * 1.0 / s                       # px over the 1 s lag
    assert abs(dx - 5.0 * expected) < 0.05 * 5.0 * expected and abs(dy) < 0.3
    assert abs(flows["moving"][0] - expected) < 0.2 * expected
    assert abs(flows["moving"][1]) < 0.25
    assert abs(flows["still"][0]) < 0.25 and abs(flows["still"][1]) < 0.25


def test_texture_and_blob_shift_recovered_between_lag_and_observation():
    u, w = 4.0, 2.0
    rp = _params(5, _clean(texture={"gain": [0.4, 0.4]}, peak_opacity=[0.8, 0.8]))
    out = render_sequence(smoke_blob_sim(u=u, w=w), rp, np.random.default_rng(0))
    s = out.visibility["metres_per_pixel"]
    k = 6
    region = out.component_opacity["smoke"][k].astype(np.float32) > 0.3
    region = cv2.erode(region.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    flow = _dis_flow(out.lag_frames[k], out.frames[k])
    assert abs(np.median(flow[..., 0][region]) - u / s) < 0.25 * u / s
    assert abs(np.median(-flow[..., 1][region]) - w / s) < 0.35 * w / s + 0.2   # image rows grow downwards


def test_steady_plume_shows_rising_texture():
    """The plume's concentration field is frozen: all apparent motion is the advected texture."""
    u, w = 1.5, 2.5
    rp = _params(22, _clean(texture={"gain": [0.6, 0.6]}, peak_opacity=[0.9, 0.9]))
    rp["aerosol_brightness"]["smoke"] = 0.9            # contrast with the terrain behind it
    out = render_sequence(plume_sim(u, w), rp, np.random.default_rng(0))
    s = out.visibility["metres_per_pixel"]
    region = out.component_opacity["smoke"][5].astype(np.float32) > 0.4
    region = cv2.erode(region.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    flow = np.median(_dis_flow(out.lag_frames[5], out.frames[5])[region], axis=0)
    assert abs(flow[0] - u / s) < 0.35 * u / s
    assert abs(-flow[1] - w / s) < 0.35 * w / s     # upwards (image rows grow downwards)


def test_texture_variance_does_not_dip_during_cross_fade():
    sim = make_sim("fog_only", lwc=np.full((T, NZ, NX), 0.15), lwc_lag=np.full((T, NZ, NX), 0.15),
                   tex_uv=(3.0, 0.0))
    # Force an even cross-fade on every frame, with pair coordinates far apart.
    sim.tex_weights[:] = 0.5
    rp = _params(6, _clean(texture={"gain": [0.5, 0.5]}, peak_opacity=[0.5, 0.5]))
    half = render_sequence(sim, rp, np.random.default_rng(0)).component_opacity["fog"].astype(np.float32)
    sim.tex_weights[:] = [1.0, 0.0]
    one = render_sequence(sim, rp, np.random.default_rng(0)).component_opacity["fog"].astype(np.float32)
    rows = slice(20, 140)
    ratio = half[:, rows].std() / one[:, rows].std()
    assert 0.8 < ratio < 1.25


def _peak(out, comp):
    return max(out.visibility["peak_opacity"][comp])


def _reference(out, comp):
    """Opacity of the aerosol body: 90th percentile over occupied pixels (the normalisation target)."""
    op = np.asarray(out.component_opacity[comp], dtype=np.float64)
    tau = -np.log(np.clip(1.0 - op, 1e-6, 1.0))
    occupied = tau[tau > 0.02 * tau.max()]
    return 1.0 - math.exp(-float(np.percentile(occupied, 90.0)))


def test_exposure_normalisation_targets_per_group():
    over = _clean(texture={"gain": [0.0, 0.0]})
    rp = _params(7, over)
    rp["peak_opacity"] = {"smoke": 0.6, "fog": 0.3, "tracer": 0.45}
    rp["background_haze_opacity"] = 0.1
    smoke = _blob(TIMES, 120.0, 90.0, 0.0, 0.0, peak=0.05)
    lwc = _fog_layer(TIMES)
    haze = np.full((T, NZ, NX), 3.0)
    # smoke_only: smoke primary, tracer = background haze
    out = render_sequence(make_sim("smoke_only", smoke=smoke, smoke_lag=smoke, tracer=haze, tracer_lag=haze),
                          rp, np.random.default_rng(0))
    g = out.visibility["groups"]
    assert g["smoke"]["role"] == "primary" and g["tracer"]["role"] == "background"
    assert _reference(out, "smoke") == pytest.approx(0.6, abs=0.04)
    assert _peak(out, "smoke") >= 0.6
    assert _reference(out, "tracer") == pytest.approx(0.1, abs=0.01)
    # both: smoke and fog each to their own target
    out = render_sequence(make_sim("both", smoke=smoke, smoke_lag=smoke, lwc=lwc, lwc_lag=lwc), rp,
                          np.random.default_rng(0))
    assert _reference(out, "smoke") == pytest.approx(0.6, abs=0.04)
    assert _reference(out, "fog") == pytest.approx(0.3, abs=0.04)
    # neither: haze_drift tracer is primary; an illumination_change tracer is background haze
    for regime, target in (("haze_drift", 0.45), ("dust_burst", 0.45), ("illumination_change", 0.1)):
        sim = make_sim("neither", tracer=haze * _blob(TIMES, 160, 60, 0, 0), tracer_lag=haze * _blob(LAG_TIMES, 160, 60, 0, 0),
                       params={"distractor_regime": regime})
        out = render_sequence(sim, rp, np.random.default_rng(0))
        assert _reference(out, "tracer") == pytest.approx(target, abs=0.04), regime
        assert out.visibility["groups"]["tracer"]["role"] == ("background" if target == 0.1 else "primary")


def test_unlabelled_component_keeps_physical_scale():
    rp = _params(8, _clean(texture={"gain": [0.0, 0.0]}))
    tiny = _fog_layer(TIMES, lwc=1e-4)
    out = render_sequence(make_sim("smoke_only", smoke=_blob(TIMES, 120, 90, 0, 0), smoke_lag=_blob(TIMES, 120, 90, 0, 0),
                                   lwc=tiny, lwc_lag=tiny), rp, np.random.default_rng(0))
    g = out.visibility["groups"]["fog"]
    assert g["role"] == "unlabelled" and not g["normalised"] and g["scale"] == 1.0
    expected = 1 - math.exp(-0.1447 * (1e-4 * 1.2) ** 0.88 * rp["line_of_sight_m"]["fog"])
    assert _peak(out, "fog") < max(3 * expected, 1e-3)


def test_background_haze_not_scaled_by_primary_group():
    haze = np.full((T, NZ, NX), 2.0)
    peaks = []
    for target in (0.2, 0.9):
        rp = _params(9, _clean(texture={"gain": [0.0, 0.0]}))
        rp["peak_opacity"]["smoke"] = target
        smoke = _blob(TIMES, 120.0, 90.0, 0.0, 0.0, peak=0.3)
        sim = make_sim("smoke_only", smoke=smoke, smoke_lag=smoke, tracer=haze, tracer_lag=haze)
        out = render_sequence(sim, rp, np.random.default_rng(0))
        assert _reference(out, "smoke") == pytest.approx(target, abs=0.04)
        peaks.append(_peak(out, "tracer"))
    assert peaks[0] == pytest.approx(peaks[1], abs=1e-3)
    assert peaks[0] == pytest.approx(rp["background_haze_opacity"], abs=0.01)


def test_empty_simulation_renders_without_aerosol_or_errors():
    for category in CATEGORIES:
        sim = make_sim(category, texcoords=False)
        rp = _params(10, hints={"shadow": False})
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            out = render_sequence(sim, rp, np.random.default_rng(0))
        assert np.isfinite(out.opacity).all() and out.opacity.max() == 0.0
        for c in ("smoke", "fog", "tracer"):
            assert out.component_opacity[c].max() == 0.0
            assert not out.visibility["groups"][c]["normalised"]
        assert out.visibility["texture_advected"] is False
        assert out.frames.std() > 1.0       # the background is still there
        ok, reasons = check_label_visibility(out, CATEGORY_LABELS[category], CFG)
        assert ok == (category == "neither"), reasons


def test_terrain_rendered_as_ground_and_has_no_aerosol():
    solid = np.zeros((NZ, NX), bool)
    X, Z = _grid()
    solid[Z < 60.0 * np.exp(-((X - 160.0) / 40.0) ** 2)] = True
    lwc = np.where(solid[None], 0.0, np.full((T, NZ, NX), 0.2))
    rp = _params(11, _clean(texture={"gain": [0.0, 0.0]}))
    out = render_sequence(make_sim("fog_only", lwc=lwc, lwc_lag=lwc, solid=solid), rp, np.random.default_rng(0))
    fog = out.component_opacity["fog"][0].astype(np.float32)
    s = out.visibility["metres_per_pixel"]
    floor = out.visibility["floor_row"]
    col = int(round((160.0 - rp["view_x_frac"] * 0.0 - 30.0) / s))   # x = 160 m (x0 = 30 m at zoom 1)
    assert fog[floor - int(30 / s), col] < 0.05                      # inside the hill (z = 30 m)
    assert fog[floor - int(100 / s), col] > 0.2                      # above it
    assert fog[floor + 2:].max() == 0.0                              # foreground band: no aerosol


def _steady_centroids(out, u):
    """Mask centroids minus the blob's own steady motion (u m/s, 5 s per frame)."""
    s = out.visibility["metres_per_pixel"]
    return np.array([(np.nonzero(m)[1].mean() - u * 5.0 / s * t, np.nonzero(m)[0].mean())
                     for t, m in enumerate(out.masks)])


def test_masks_follow_the_shaken_frames():
    u = 1.5
    rp = _params(12, _clean(camera_shake_px=[4.0, 4.0], camera_rotation_deg=[0.5, 0.5]))
    blob = render_sequence(smoke_blob_sim(u=u, peak=1.0), rp, np.random.default_rng(3))
    # Same params and seed: same camera motion and noise, no aerosol.
    empty = render_sequence(make_sim("smoke_only"), rp, np.random.default_rng(3))
    diff = np.abs(blob.frames.astype(np.int16) - empty.frames.astype(np.int16)) > 6
    ious = [(diff[t] & blob.masks[t]).sum() / max((diff[t] | blob.masks[t]).sum(), 1) for t in range(T)]
    assert min(ious) > 0.6
    # The blob moves steadily; the residual centroid jitter is the camera shake.
    c = _steady_centroids(blob, u)
    assert (c - c.mean(axis=0)).std() > 1.5
    # Without shake (and without texture, which moves the mask edges) the masks are steady.
    still = render_sequence(smoke_blob_sim(u=u, peak=1.0),
                            _params(12, _clean(texture={"gain": [0.0, 0.0], "shading": [0.0, 0.0]})),
                            np.random.default_rng(3))
    c2 = _steady_centroids(still, u)
    assert (c2 - c2.mean(axis=0)).std() < 0.5


def test_check_label_visibility_accept_and_reject():
    rp = _params(13, _clean(texture={"gain": [0.2, 0.2]}))
    rp["peak_opacity"]["smoke"] = 0.5
    out = render_sequence(smoke_blob_sim(), rp, np.random.default_rng(0))
    ok, reasons = check_label_visibility(out, (1, 0), CFG)
    assert ok and reasons == []
    ok, reasons = check_label_visibility(out, (0, 1), CFG)
    assert not ok and any("smoke not labelled" in r for r in reasons) and any("fog labelled" in r for r in reasons)
    ok, reasons = check_label_visibility(out, (0, 0), CFG)
    assert not ok
    # a primary target well below label_min_opacity is rejected
    rp["peak_opacity"]["smoke"] = 0.02
    faint = render_sequence(smoke_blob_sim(), rp, np.random.default_rng(0))
    assert not check_label_visibility(faint, (1, 0), CFG)[0]
    # visible in too few frames: the blob leaves the view after a few frames
    rp["peak_opacity"]["smoke"] = 0.5
    fast = make_sim("smoke_only", smoke=_blob(TIMES, 60, 90, 25.0, 0.0), smoke_lag=_blob(LAG_TIMES, 60, 90, 25.0, 0.0),
                    tex_uv=(25.0, 0.0))
    out = render_sequence(fast, rp, np.random.default_rng(0))
    ok, reasons = check_label_visibility(out, (1, 0), CFG)
    assert not ok and "only" in reasons[0]


def test_nuisances_follow_the_simulation_hints():
    from src.data.render import _scene_canvas, _view

    hints = {"shadow": True, "strong_artifact": False, "fp_boost": False, "distant_cloud": True}
    sim = make_sim("neither", params={"render_hints": hints, "distractor_regime": "illumination_change"})
    with pytest.raises(ValueError, match="hints"):
        render_sequence(sim, _params(18, _clean()), np.random.default_rng(0))
    on = _params(18, _clean(), hints=hints)
    off = _params(18, _clean())
    out_on = render_sequence(sim, on, np.random.default_rng(0))
    out_off = render_sequence(make_sim("neither"), off, np.random.default_rng(0))
    diff = out_on.frames.astype(np.int16) - out_off.frames.astype(np.int16)
    view = _view(sim, on)
    scene, _, _ = _scene_canvas(sim, on, view)
    m = view.margin
    sky = scene[3, m:m + view.H, m:m + view.W] > 0.999
    cloud = out_on.component_opacity["cloud"].astype(np.float32)
    assert cloud.max() > 0.3 and out_off.component_opacity["cloud"].max() == 0.0
    assert cloud[:, ~sky].max() < 1e-3                      # a distant cloud is only seen against the sky
    # Away from the cloud, the shadow only darkens, and never the sky.
    no_cloud = cloud < 1e-4
    assert diff[no_cloud].max() <= 0
    assert diff[no_cloud & ~sky[None]].min() < -5
    assert np.abs(diff[no_cloud & sky[None]]).max() == 0
    # The moving shadow is a candidate region (brightness change), like the cloud.
    assert max(out_on.visibility["mask_area_frac"]) > 0.01


def test_optional_contrast_criterion():
    rp = _params(19, _clean(texture={"gain": [0.2, 0.2]}))
    rp["peak_opacity"]["smoke"] = 0.6
    out = render_sequence(smoke_blob_sim(), rp, np.random.default_rng(0))
    assert check_label_visibility(out, (1, 0), CFG)[0]
    strict = {"render": {**CFG["render"], "label_min_contrast": 1.0}}
    ok, reasons = check_label_visibility(out, (1, 0), strict)
    assert not ok and "contrast" in reasons[0]


def test_real_background_from_manifest_and_fallback(tmp_path):
    from src.utils.io import write_json

    img = (np.random.default_rng(0).random((300, 400)) * 255).astype(np.uint8)
    cv2.imwrite(str(tmp_path / "cam_a.png"), img)
    write_json(tmp_path / "manifest.json", {"images": [{"file": "cam_a.png", "camera_id": "a"},
                                                       {"file": "missing.png", "camera_id": "b"}]})
    rp = _params(20, {"background": {"procedural_prob": 0.0, "dir": str(tmp_path)}})
    assert rp["background"]["kind"] == "real" and rp["background"]["camera_id"] == "a"
    out = render_sequence(smoke_blob_sim(), rp, np.random.default_rng(0))
    assert out.frames.std() > 5
    excluded = _params(20, {"background": {"procedural_prob": 0.0, "dir": str(tmp_path), "exclude_cameras": ["a"]}})
    assert excluded["background"]["kind"] == "procedural" and excluded["background"]["requested_real"]
    dumps_json(rp)


# ---------------------------------------------------------------------- no lag frames (far-field profile)
def test_render_without_lag_frames_shapes_and_determinism():
    sim = without_lag(smoke_blob_sim())
    rp = _params(1)
    rp["peak_opacity"]["smoke"] = 0.5
    out = render_sequence(sim, rp, np.random.default_rng(2))
    H, W = CFG["render"]["native_size"]
    assert out.lag_frames is None and out.lag_timestamps is None
    assert out.frames.shape == (T, H, W) and out.frames.dtype == np.uint8
    assert out.masks.shape == (T, H, W) and out.masks.dtype == bool
    assert out.opacity.shape == (T, H, W)
    assert all(v.shape == (T, H, W) and v.dtype == np.float16 for v in out.component_opacity.values())
    np.testing.assert_array_equal(out.timestamps, TIMES)
    assert len(out.visibility["peak_opacity"]["smoke"]) == T and len(out.visibility["mask_area_frac"]) == T
    dumps_json(out.visibility)
    assert check_label_visibility(out, (1, 0), CFG)[0]
    again = render_sequence(sim, rp, np.random.default_rng(2))
    np.testing.assert_array_equal(out.frames, again.frames)
    np.testing.assert_array_equal(out.masks, again.masks)
    assert out.visibility == again.visibility
    assert not np.array_equal(out.frames, render_sequence(sim, rp, np.random.default_rng(3)).frames)


def test_dropping_lag_frames_leaves_the_observations_unchanged():
    """With every random nuisance off, observation frames, opacities and exposure do not depend on
    whether lag frames are rendered (only the length of the random stream does)."""
    rp = _params(5, _clean(texture={"gain": [0.4, 0.4]}))
    rp["peak_opacity"]["smoke"] = 0.6
    sim = smoke_blob_sim(u=3.0, w=1.0)
    lag = render_sequence(sim, rp, np.random.default_rng(0))
    nolag = render_sequence(without_lag(sim), rp, np.random.default_rng(0))
    np.testing.assert_array_equal(lag.frames, nolag.frames)
    np.testing.assert_array_equal(lag.opacity, nolag.opacity)
    assert lag.visibility["groups"] == nolag.visibility["groups"]
    assert lag.visibility["peak_opacity"] == nolag.visibility["peak_opacity"]
    assert check_label_visibility(nolag, (1, 0), CFG)[0]


# ---------------------------------------------------------------------- correlated camera shake
def test_shake_process_is_stationary_and_correlated_in_time():
    """OU shake at far-field frame times (observations 60 s apart, lag frames 1 s before them)."""
    from src.data.render import _shake_process

    obs = np.arange(T) * 60.0
    times = np.concatenate([obs, obs - 1.0])
    for tau, lag_corr, far_corr in ((20.0, math.exp(-1.0 / 20.0), math.exp(-3.0)), (2.0, math.exp(-0.5), 0.0)):
        z = np.stack([_shake_process(np.random.default_rng(s), times, tau) for s in range(2000)])   # (n, 2T, 4)
        assert z.std() == pytest.approx(1.0, abs=0.03), tau
        c_lag = np.mean([np.corrcoef(z[:, k, 0], z[:, T + k, 0])[0, 1] for k in range(T)])
        c_far = np.mean([np.corrcoef(z[:, k, 1], z[:, k + 1, 1])[0, 1] for k in range(T - 1)])
        assert c_lag == pytest.approx(lag_corr, abs=0.03), tau
        assert c_far == pytest.approx(far_corr, abs=0.03), tau
    # tau 0 (version-1 params): independent per frame, drawn in frame order.
    np.testing.assert_array_equal(_shake_process(np.random.default_rng(4), times, 0.0),
                                  np.clip(np.random.default_rng(4).standard_normal((2 * T, 4)), -3.0, 3.0))


def test_camera_affines_have_the_sampled_stationary_std():
    from src.data.render import _camera_affines, _view

    sim = make_sim("smoke_only")
    rp = _params(12, _clean(camera_shake_px=[3.0, 3.0], camera_rotation_deg=[0.5, 0.5]))
    rp["camera_shake_tau_s"] = 4.0
    view = _view(sim, rp)
    times = np.concatenate([TIMES, LAG_TIMES])
    A = np.stack([_camera_affines(np.random.default_rng(s), rp, times, view) for s in range(500)])
    c0 = np.array([(view.W - 1) / 2.0, (view.H - 1) / 2.0])
    shift = A[..., :2] @ c0 + A[..., 2] - c0 - view.margin            # where the image centre lands
    rot = np.degrees(np.arctan2(A[..., 1, 0], A[..., 0, 0]))
    assert shift.std() == pytest.approx(3.0, rel=0.06)
    assert rot.std() == pytest.approx(0.5, rel=0.06)
    assert abs(shift.mean()) < 0.15 and abs(rot.mean()) < 0.03


def test_lag_frame_shakes_with_its_observation_for_a_slow_process():
    """Image shift between each lag frame and its observation (1 s apart) on an empty scene: ~0 for a
    slow shake process, ~sqrt(2) x the shake per axis for an (almost) independent one."""
    sim = make_sim("smoke_only")
    rp = _params(12, _clean(camera_shake_px=[5.0, 5.0]))
    rms = {}
    for tau in (3000.0, 0.01):
        rp["camera_shake_tau_s"] = tau
        out = render_sequence(sim, rp, np.random.default_rng(3))
        d = [cv2.phaseCorrelate(out.lag_frames[k].astype(np.float32), out.frames[k].astype(np.float32))[0]
             for k in range(T)]
        rms[tau] = float(np.sqrt(np.mean(np.square(d))))
    assert rms[3000.0] < 0.5
    assert 0.6 * math.sqrt(2.0) * 5.0 < rms[0.01] < 1.4 * math.sqrt(2.0) * 5.0


# ---------------------------------------------------------------------- subgrid texture turbulence
def test_sgs_displacement_field_statistics():
    """D(x, z, t): off without u_sgs or gain, local velocity scale gain * u_sgs, zero mean, scales with dx."""
    from src.data.render import _sgs_displacement

    times = np.concatenate([TIMES, LAG_TIMES])
    tex = {"sgs_gain": 1.0, "sgs_seed": 5}
    full = lambda v: np.full((2 * T, NZ, NX), v, np.float32)
    assert _sgs_displacement(None, times, DX, tex)[0] is None
    assert _sgs_displacement(full(0.0), times, DX, tex)[0] is None
    assert _sgs_displacement(full(0.3), times, DX, {**tex, "sgs_gain": 0.0})[0] is None
    D, rec = _sgs_displacement(full(0.1), times, DX, tex)
    assert D.shape == (2 * T, NZ, NX, 2) and D.dtype == np.float32 and rec["applied"]
    np.testing.assert_array_equal(D, _sgs_displacement(full(0.1), times, DX, tex)[0])
    assert not np.array_equal(D, _sgs_displacement(full(0.1), times, DX, {**tex, "sgs_seed": 6})[0])
    # Velocity over the 1 s lag and displacement over 5 s, averaged over realisations.
    vel, disp5 = [], []
    for u in (0.05, 0.2, 0.6):
        v1, d5, means, rms = [], [], [], []
        for seed in range(20):
            D, _ = _sgs_displacement(full(u), times, DX, {"sgs_gain": 1.0, "sgs_seed": seed})
            v1.append(np.sqrt(((D[:T] - D[T:]) ** 2).sum(-1).mean()))
            d5.append(np.sqrt(((D[1:T] - D[:T - 1]) ** 2).sum(-1).mean()))
            means.append(D.mean(axis=(1, 2)))
            rms.append(np.sqrt((D ** 2).sum(-1).mean()))
        vel.append(np.mean(v1))
        disp5.append(np.mean(d5))
        assert np.abs(np.stack(means)).max() < 0.05 * np.mean(rms), u       # no net drift in any frame
        if u <= 0.2:                                                        # 1 s << correlation time
            assert vel[-1] == pytest.approx(u, rel=0.1), u
    assert vel[0] < vel[1] < vel[2] and disp5[0] < disp5[1] < disp5[2]
    # Velocity proportional to the gain; length scales with the grid spacing.
    D, _ = _sgs_displacement(full(0.05), times, DX, {"sgs_gain": 0.5, "sgs_seed": 0})
    D1, _ = _sgs_displacement(full(0.05), times, DX, {"sgs_gain": 1.0, "sgs_seed": 0})
    v = lambda d: np.sqrt(((d[:T] - d[T:]) ** 2).sum(-1).mean())
    assert v(D) == pytest.approx(0.5 * v(D1), rel=0.15)
    _, far = _sgs_displacement(full(0.05), times, 10 * DX, {"sgs_gain": 1.0, "sgs_seed": 0})
    _, near = _sgs_displacement(full(0.05), times, DX, {"sgs_gain": 1.0, "sgs_seed": 0})
    assert far["amplitude_m"] == pytest.approx(10 * near["amplitude_m"])
    # Local: no displacement where there is no subgrid motion ...
    u = full(0.3)
    u[:, :, : NX // 2] = 0.0
    D, _ = _sgs_displacement(u, times, DX, tex)
    assert np.abs(D[:, :, : NX // 2]).max() == 0.0 and np.abs(D[:, :, NX // 2:]).max() > 0.1
    # ... and quiet air (0.02 m/s) next to a turbulent region (0.5 m/s) each moves at its own
    # velocity, with a similar displacement size and its own correlation time (eddy size / u_sgs).
    u = full(0.5)
    u[:, :, : NX // 2] = 0.02
    sides = {"quiet": (slice(None), slice(4, NX // 2 - 4)), "turbulent": (slice(None), slice(NX // 2 + 4, NX - 4))}
    stats = {name: {"v": [], "size": [], "corr5": []} for name in sides}
    for seed in range(20):
        D, rec = _sgs_displacement(u, times, DX, {"sgs_gain": 1.0, "sgs_seed": seed})
        for name, (rows, cols) in sides.items():
            d = D[:, rows, cols]
            stats[name]["v"].append(np.sqrt(((d[:T] - d[T:]) ** 2).sum(-1).mean()))
            stats[name]["size"].append(np.sqrt((d ** 2).sum(-1).mean()))
            stats[name]["corr5"].append(np.mean([np.corrcoef(d[k].ravel(), d[k + 1].ravel())[0, 1] for k in range(T - 1)]))
    s = {name: {key: float(np.mean(v)) for key, v in st.items()} for name, st in stats.items()}
    assert s["quiet"]["v"] == pytest.approx(0.02, rel=0.15)
    assert s["turbulent"]["v"] == pytest.approx(0.5, rel=0.25)          # 1 s is not << its 3 s correlation time
    for name in sides:
        assert s[name]["size"] == pytest.approx(rec["amplitude_m"], rel=0.25), name
    assert s["quiet"]["corr5"] > 0.9 and s["turbulent"]["corr5"] < 0.6


def test_sgs_displacement_has_no_jumps_where_u_sgs_changes_in_time():
    """u_sgs changing between frames (a 5x step in time; lag and observation frames at different
    u_sgs): between consecutive frames in time order (lag and observation interleaved) D moves by
    ~ g u_sgs dt, never by a jump of the displacement size A, and stays zero-mean."""
    from src.data.render import _sgs_displacement

    times = np.concatenate([TIMES, LAG_TIMES])
    order = np.argsort(times, kind="stable")
    dt = np.diff(times[order])
    cases = {"step": np.where(times < 24.5, 0.02, 0.1),                     # 0.02 -> 0.1 m/s between the lag frame at 24 s and the observation at 25 s
             "lag_vs_obs": np.where(np.arange(2 * T) < T, 0.1, 0.02)}       # observations 0.1, lag frames 0.02 m/s
    for name, u_t in cases.items():
        u = np.repeat(u_t.astype(np.float32)[:, None, None], NZ, axis=1).repeat(NX, axis=2)
        u_mid = 0.5 * (u_t[order][1:] + u_t[order][:-1])
        ratio, largest = [], 0.0
        for seed in range(10):
            D, rec = _sgs_displacement(u, times, DX, {"sgs_gain": 1.0, "sgs_seed": seed})
            d = D[order].astype(np.float64)
            step = np.sqrt(((d[1:] - d[:-1]) ** 2).sum(-1).mean(axis=(1, 2)))      # rms |dD| per frame step
            ratio.append(step / (u_mid * dt))
            largest = max(largest, float(step.max()) / rec["amplitude_m"])
            rms = np.sqrt((d ** 2).sum(-1).mean())
            assert np.abs(d.mean(axis=(1, 2))).max() < 0.05 * rms, name          # no net drift in any frame
        ratio = np.mean(ratio, axis=0)
        # Every step moves at ~ g u_sgs (1-4 s steps, well inside the ~16 s correlation time at 0.1 m/s).
        assert ratio.min() > 0.75 and ratio.max() < 1.25, (name, ratio)
        assert largest < 0.75, name                                         # <= 0.4 m per step vs A ~ 1 m


def test_sgs_turbulence_in_rendered_frames():
    """A static fog layer with a static texture: identical lag and observation frames without subgrid
    motion (sgs_u None, zero, or gain 0, all rendering the same frames); with it the texture boils
    with an rms flow growing with u_sgs and no net drift."""
    rp = _params(4, _clean(texture={"gain": [0.6, 0.6], "sgs_gain": [1.0, 1.0]}, peak_opacity=[0.95, 0.95]))

    def fog():
        return make_sim("fog_only", lwc=_fog_layer(TIMES, top=150.0), lwc_lag=_fog_layer(LAG_TIMES, top=150.0),
                        texcoords=False)

    ref = render_sequence(fog(), rp, np.random.default_rng(0))
    assert ref.visibility["sgs_texture"]["applied"] is False
    off_gain = {**rp, "texture": {**rp["texture"], "sgs_gain": 0.0}}
    for sim, params in ((with_sgs(fog(), 0.0), rp), (with_sgs(fog(), 0.5, 0.5), off_gain)):
        out = render_sequence(sim, params, np.random.default_rng(0))
        np.testing.assert_array_equal(out.frames, ref.frames)
        np.testing.assert_array_equal(out.lag_frames, ref.lag_frames)
    dense = cv2.blur(ref.component_opacity["fog"][5].astype(np.float32), (9, 9)) > 0.85
    region = cv2.erode(dense.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert region.sum() > 1000
    rms = []
    for u in (0.0, 0.4, 1.5):
        out = ref if u == 0.0 else render_sequence(with_sgs(fog(), u, u), rp, np.random.default_rng(0))
        diff = np.abs(out.frames.astype(np.int16) - out.lag_frames.astype(np.int16))[:, region]
        flow = np.concatenate([_dis_flow(out.lag_frames[k], out.frames[k])[region] for k in range(T)])
        rms.append(float(np.sqrt((flow ** 2).sum(-1).mean())))
        if u == 0.0:
            assert diff.max() == 0
        else:
            assert out.visibility["sgs_texture"]["applied"] and diff.mean() > 0.5
            # No net drift (loose: the dense-fog band is only a few eddies tall; the field test
            # checks the zero mean tightly).
            assert np.abs(flow.mean(axis=0)).max() < 0.3 * rms[-1]
    assert rms[0] < rms[1] < rms[2] and rms[2] > 0.1


# ---------------------------------------------------------------------- parameters
def _flatten(d, prefix=""):
    """Numeric leaves of a params dict; optional blocks get a `key?` presence flag; strings
    become one-hot `key=value` entries."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out[key + "?"] = 1.0
            out.update(_flatten(v, key + "."))
        elif v is None:
            out[key + "?"] = 0.0
        elif isinstance(v, str):
            out[f"{key}={v}"] = 1.0
        elif isinstance(v, (bool, int, float)):
            out[key] = float(v)
        elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
            for i, x in enumerate(v):
                out[f"{key}[{i}]"] = float(x)
    return out


def test_sample_render_params_is_class_independent():
    """Same hints -> same distribution of every rendering choice for every category.

    sample_render_params never sees the class, so this documents the property:
    marginals (means of every numeric leaf, presence rates of optional blocks,
    frequencies of categorical choices) agree across per-category random streams.
    """
    hints = {"shadow": False, "strong_artifact": True, "fp_boost": False, "distant_cloud": True}
    n = 300
    stats = {}
    for ci, category in enumerate(CATEGORIES):
        rows = [_flatten(sample_render_params(child_rng(99, ci, i), CFG, hints)) for i in range(n)]
        keys = sorted(set().union(*rows))
        # one-hot / presence keys absent from a row mean 0; numeric leaves of absent blocks are NaN
        stats[category] = {k: np.array([r.get(k, 0.0 if ("=" in k or k.endswith("?")) else np.nan) for r in rows])
                           for k in keys}
    keys = set().union(*(set(s) for s in stats.values()))
    for k in sorted(keys):
        if k.endswith(".seed"):
            continue
        ref = stats[CATEGORIES[0]].get(k, np.zeros(n))
        for c in CATEGORIES[1:]:
            x = stats[c].get(k, np.zeros(n))
            a, b = ref[~np.isnan(ref)], x[~np.isnan(x)]
            if len(a) < 10 or len(b) < 10:
                continue
            se = math.sqrt(a.var() / len(a) + b.var() / len(b)) + 1e-9
            assert abs(a.mean() - b.mean()) < 4.5 * se + 1e-9, (k, c, a.mean(), b.mean())


def test_hints_only_switch_nuisances_on():
    off = _params(14, hints={})
    on = _params(14, hints={"shadow": True, "strong_artifact": True, "fp_boost": True, "distant_cloud": True})
    assert off["shadow"] is None and off["distant_cloud"] is None and off["strong_artifact"] is None
    assert on["shadow"] and on["distant_cloud"] and on["strong_artifact"] and on["fp_boost"]
    changed = {"shadow", "distant_cloud", "strong_artifact", "fp_boost", "hints", "camera_shake_px",
               "zoom_jitter", "exposure_flicker", "masks"}
    for k in off:
        if k not in changed:
            assert off[k] == on[k], k
    assert on["camera_shake_px"] >= off["camera_shake_px"] and on["zoom_jitter"] > 0.0
    assert on["masks"]["false_positive_prob"] == CFG["render"]["masks"]["false_positive_prob_boost"]
    assert {k: v for k, v in off["masks"].items() if k != "false_positive_prob"} == \
           {k: v for k, v in on["masks"].items() if k != "false_positive_prob"}


def test_v2_params_come_from_config_and_are_drawn_last():
    """camera_shake_tau_s and texture.sgs_gain come from the config (code defaults without it) and are
    drawn after every version-1 value: overriding them changes nothing else."""
    p = _params(30)
    assert p["version"] == 2
    lo, hi = CFG["render"].get("camera_shake_tau_s") or SHAKE_TAU_S_DEFAULT
    assert lo <= p["camera_shake_tau_s"] <= hi
    lo, hi = CFG["render"]["texture"].get("sgs_gain", SGS_GAIN_DEFAULT)
    assert lo <= p["texture"]["sgs_gain"] <= hi and isinstance(p["texture"]["sgs_seed"], int)
    q = _params(30, {"camera_shake_tau_s": [10.0, 10.0], "texture": {"sgs_gain": [0.0, 0.0]}})
    assert q["camera_shake_tau_s"] == 10.0 and q["texture"]["sgs_gain"] == 0.0

    def v1(d):
        out = {k: v for k, v in d.items() if k != "camera_shake_tau_s"}
        out["texture"] = {k: v for k, v in d["texture"].items() if k != "sgs_gain"}
        return out

    assert v1(p) == v1(q)
    # A configured 0 is a value, not a missing key (one draw either way: the stream is unchanged).
    z = _params(30, {"camera_shake_tau_s": 0.0})
    assert z["camera_shake_tau_s"] == 0.0 and v1(z) == v1(p) and z["texture"] == p["texture"]
    dumps_json(p)


def test_version1_params_render_with_independent_shake_and_no_subgrid_motion():
    """Stored version-1 params (no camera_shake_tau_s, no texture.sgs_*) render like tau 0 and gain 0,
    even for a simulation that carries sgs_u."""
    rp = _params(31, hints={})
    old = {k: v for k, v in rp.items() if k != "camera_shake_tau_s"}
    old["texture"] = {k: v for k, v in rp["texture"].items() if not k.startswith("sgs_")}
    off = {**rp, "camera_shake_tau_s": 0.0, "texture": {**rp["texture"], "sgs_gain": 0.0}}
    a = render_sequence(with_sgs(smoke_blob_sim(), 0.5), old, np.random.default_rng(1))
    b = render_sequence(smoke_blob_sim(), off, np.random.default_rng(1))
    np.testing.assert_array_equal(a.frames, b.frames)
    np.testing.assert_array_equal(a.lag_frames, b.lag_frames)
    np.testing.assert_array_equal(a.masks, b.masks)


def test_robustness_overrides_are_applied():
    rob = CFG["robustness"]
    for i in range(30):
        p = _params(100 + i, rob["shake_heavy"])
        assert 4.0 <= p["camera_shake_px"] <= 7.0 and 0.8 <= p["camera_rotation_deg"] <= 1.5
        p = _params(200 + i, rob["detector_masks"])
        assert p["masks"]["source"] == "detector"
        assert p["masks"]["threshold_contrast"] <= CFG["render"]["masks"]["threshold_contrast"][1]
        p = _params(300 + i, rob["bbox_masks"])
        assert p["masks"]["source"] == "bbox"
        p = _params(400 + i, rob["occlusion_heavy"])
        assert p["occlusion"] is not None and 0.3 <= p["occlusion"]["area_frac"] <= 0.45
        p = _params(500 + i, rob["render_shift"])
        assert p["jpeg"] and 12 <= p["jpeg_quality"] <= 25 and 0.35 <= p["contrast"] <= 0.55
        p = _params(600 + i, rob["zoom_out"])
        assert 0.35 <= p["zoom"] <= 0.55
    sources = {_params(700 + i)["masks"]["source"] for i in range(60)}
    assert sources == set(MASK_SOURCES)


def test_heavy_occlusion_removes_masked_pixels_and_covers_area():
    rp = _params(15, _clean(occlusion_prob=1.0, occlusion_area_frac=[0.3, 0.3]))
    out = render_sequence(fog_layer_sim(), rp, np.random.default_rng(0))
    occluded = out.component_opacity["fog"][0].astype(np.float32) == 0
    assert 0.2 < occluded[: out.visibility["floor_row"]].mean()
    assert all(m.mean() < 0.9 for m in out.masks)


# ---------------------------------------------------------------------- masks
def _disc_stack(n=6, h=96, w=96, r=15):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.stack([((xx - 30 - 5 * t) ** 2 + (yy - 48) ** 2 < r * r).astype(np.float32) * 0.2 for t in range(n)])


def test_degrade_masks_threshold_morphology_and_occluder():
    D = _disc_stack()
    clean = {"threshold_contrast": 0.05, "morph_px": 0, "boundary_noise": 0.0, "drop_component_prob": 0.0,
             "false_positive_prob": 0.0, "frame_dropout_prob": 0.0, "min_area_frac": 0.001, "threshold_jitter": 0.0}
    m = degrade_masks(D, np.random.default_rng(0), clean)
    np.testing.assert_array_equal(m, D > 0.05)
    grown = degrade_masks(D, np.random.default_rng(0), {**clean, "morph_px": 2})
    assert (grown.sum(axis=(1, 2)) > m.sum(axis=(1, 2))).all()
    occ = np.zeros(D.shape[1:])
    occ[:, :40] = 1.0
    m = degrade_masks(D, np.random.default_rng(0), clean, occluder=occ)
    assert not m[:, :, :40].any()
    dropped = degrade_masks(D, np.random.default_rng(0), {**clean, "frame_dropout_prob": 1.0})
    assert not dropped.any()
    fps = degrade_masks(np.zeros_like(D), np.random.default_rng(0), {**clean, "false_positive_prob": 1.0})
    assert fps.any(axis=(1, 2)).all()


def test_boundary_noise_is_small():
    D = _disc_stack(n=4)
    base = D > 0.05
    noisy = degrade_masks(D, np.random.default_rng(1), {"threshold_contrast": 0.05, "morph_px": 0,
                                                        "boundary_noise": 0.15, "drop_component_prob": 0.0,
                                                        "false_positive_prob": 0.0, "frame_dropout_prob": 0.0,
                                                        "threshold_jitter": 0.0})
    iou = (noisy & base).sum() / (noisy | base).sum()
    assert 0.75 < iou < 1.0


def test_boxes_from_masks():
    m = np.zeros((2, 50, 50), bool)
    m[0, 10:20, 5:15] = True
    m[0, 30:40, 30:45] = True
    b = boxes_from_masks(m, np.random.default_rng(0), jitter=0.0)
    np.testing.assert_array_equal(b, m)
    j = boxes_from_masks(m, np.random.default_rng(0), jitter=0.2)
    assert j[0].any() and not j[1].any()
    n, _ = cv2.connectedComponents(j[0].astype(np.uint8))
    assert n - 1 in (1, 2)


def _moving_textured_blob(n=12, h=128, w=128, shake=0.0, seed=0):
    rng = np.random.default_rng(seed)
    bg = cv2.GaussianBlur(rng.random((h + 40, w + 40)).astype(np.float32), (0, 0), 2.0)
    bg = 0.3 + 0.6 * (bg - bg.min()) / (bg.max() - bg.min())
    tex = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.5)
    yy, xx = np.mgrid[0:h, 0:w]
    frames, truth = [], []
    for t in range(n):
        dx, dy = (rng.normal(0, shake, 2) if shake else (0.0, 0.0))
        frame = bg[20 + int(round(dy)):20 + int(round(dy)) + h, 20 + int(round(dx)):20 + int(round(dx)) + w].copy()
        cx = 25 + 6 * t
        inside = ((xx - cx) / 16.0) ** 2 + ((yy - 64) / 12.0) ** 2 < 1.0
        alpha = 0.6 * inside * (0.7 + 0.6 * np.roll(tex, 6 * t, axis=1))
        frame = (1 - alpha) * frame + alpha * 0.85
        frame += rng.normal(0, 0.01, frame.shape)
        frames.append(np.clip(frame * 255, 0, 255).astype(np.uint8))
        truth.append(inside)
    return np.stack(frames), np.stack(truth)


@pytest.mark.parametrize("shake", [0.0, 2.0])
def test_detect_candidate_masks_finds_moving_textured_blob(shake):
    frames, truth = _moving_textured_blob(shake=shake)
    m = detect_candidate_masks(frames, np.arange(len(frames)) * 5.0)
    assert m.shape == truth.shape and m.dtype == bool
    ious = [(a & b).sum() / max((a | b).sum(), 1) for a, b in zip(m, truth)]
    assert np.median(ious) > 0.5
    assert (m & ~cv2.dilate(truth.astype(np.uint8).max(0), np.ones((15, 15), np.uint8)).astype(bool)).mean() < 0.01


def test_detect_candidate_masks_accepts_every_extractor_input_type():
    """The detector converts its input exactly like the feature extractor (float64 BGR, uint16, ...)."""
    frames, _ = _moving_textured_blob()
    ts = np.arange(len(frames)) * 5.0
    ref = detect_candidate_masks(frames, ts)
    bgr = np.repeat(frames[..., None], 3, axis=-1)
    for variant in (frames.astype(np.float64) / 255.0, bgr.astype(np.float64) / 255.0, bgr,
                    frames.astype(np.uint16) * 257):
        np.testing.assert_array_equal(detect_candidate_masks(variant, ts), ref)


def test_detect_candidate_masks_quiet_on_static_scene():
    frames, _ = _moving_textured_blob()
    static = np.repeat(frames[:1], 12, axis=0).astype(np.int16)
    static = np.clip(static + np.random.default_rng(0).normal(0, 2.5, static.shape), 0, 255).astype(np.uint8)
    m = detect_candidate_masks(static)
    assert m.mean() < 0.005


def test_detector_mask_source_on_rendered_frames():
    rp = _params(16, _clean(masks={"source_probs": {"detector": 1.0}}, texture={"gain": [0.4, 0.4]},
                            noise_sigma=[0.01, 0.01]))
    rp["peak_opacity"]["smoke"] = 0.7
    out = render_sequence(smoke_blob_sim(u=5.0), rp, np.random.default_rng(0))
    assert out.mask_source == "detector"
    share = np.asarray(out.visibility["mask_share"]["smoke"])
    assert np.median(share) > 0.4


def test_render_speed_budget():
    import time

    sim = smoke_blob_sim()
    rp = _params(17, hints={"shadow": True, "distant_cloud": True})
    render_sequence(sim, rp, np.random.default_rng(0))       # warm caches
    t0 = time.perf_counter()
    for i in range(3):
        render_sequence(sim, rp, np.random.default_rng(i))
    per = (time.perf_counter() - t0) / 3
    assert per < 1.0, per   # generous for CI; the measured budget is ~150 ms (see module report)


def test_every_tracer_distractor_regime_is_rendered_as_the_primary_group():
    """A neither scene's own tracer distractor (haze, dust, rain) gets the primary exposure range,
    not the faint background-haze one; background haze in other scenes stays background."""
    from types import SimpleNamespace

    from src.constants import NEITHER_REGIMES
    from src.data.render import PRIMARY_TRACER_REGIMES, _group_roles

    tracer_regimes = {"haze_drift", "dust_burst", "rain_shaft"}
    assert tracer_regimes <= set(NEITHER_REGIMES) and set(PRIMARY_TRACER_REGIMES) == tracer_regimes
    for regime in tracer_regimes:
        sim = SimpleNamespace(labels=(0, 0), category="neither", params={"distractor_regime": regime})
        assert _group_roles(sim)["tracer"] == "primary"
    for category, labels in (("neither", (0, 0)), ("fog_only", (0, 1))):
        sim = SimpleNamespace(labels=labels, category=category, params={"distractor_regime": "distant_cloud"})
        assert _group_roles(sim)["tracer"] == "background"


def test_stretch_attenuation_fades_only_sheared_texture():
    """render.texture.max_stretch fades a pair's texture by min(1, max / stretch); off when unset."""
    from src.data.render import _stretch_attenuation

    nz, nx, dx = 16, 20, 5.0
    z, x = np.meshgrid((np.arange(nz) + 0.5) * dx, (np.arange(nx) + 0.5) * dx, indexing="ij")
    tc = np.zeros((2, 4, nz, nx))
    tc[0] = np.stack([x, z, x, z])                         # frame 0: no deformation
    tc[1] = np.stack([x + 3.0 * z, z, x, z])               # frame 1: pair a sheared (stretch ~3.3), b untouched
    assert _stretch_attenuation(tc, dx, None) is None and _stretch_attenuation(tc, dx, 0) is None
    att = _stretch_attenuation(tc, dx, 1.5)
    assert att.shape == (2, 2, nz, nx) and att.dtype == np.float32
    np.testing.assert_allclose(att[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(att[1, 1], 1.0, atol=1e-6)
    expected = 1.5 / ((3.0 ** 2 + 2.0 + 3.0 * (3.0 ** 2 + 4.0) ** 0.5) / 2.0) ** 0.5   # 1.5 / largest singular value
    np.testing.assert_allclose(att[1, 0], expected, rtol=1e-6)
