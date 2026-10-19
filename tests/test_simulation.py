"""Coupled-simulation checks: steady ambient state, plumes, fog, budgets, scenarios, profiles."""

import collections
import copy

import numpy as np
import pytest

from src.cfd.lbm import LatticeUnits, LBMConfig, LBMSolver, ambient_maintaining_acceleration
from src.cfd.results import ALL_LAG_FIELDS, MissingLagFields, SimulationResult
from src.cfd.scenarios import (
    HOLDOUT_TAGS, SPLITS, build_spec, regime_weight, run_scenario, sample_scenario, scenario_seed,
)
from src.cfd.simulation import run_simulation
from src.constants import CATEGORIES, FOG_REGIMES, NEITHER_REGIMES, OPT_IN_REGIMES, SMOKE_REGIMES
from src.utils.config import REPO_ROOT, load_simulation_config

CFG = load_simulation_config()
FAR_CFG = load_simulation_config(REPO_ROOT / "configs" / "simulation_far.yaml")


def _short(cfg, n_frames=4):
    cfg = copy.deepcopy(cfg)
    cfg["time"]["n_frames"] = n_frames
    cfg["time"]["precursor_s"] = 10.0
    return cfg


def _calm(params, spinup=10.0):
    p = copy.deepcopy(params)
    p["wind"].update(speed_ms=0.0, gust_fraction=0.0, flip=False, ramp_factor=1.0)
    p["timing"]["spinup_s"] = spinup
    p["background_haze"] = None
    return p


def _find(category, regime_key, regime, cfg, split="train", limit=400):
    for i in range(limit):
        p = sample_scenario(category, split, i, cfg)
        if p[regime_key] == regime:
            return p
    raise AssertionError(f"no {regime} in {limit} draws of {category}")


def _hot_plume(cfg, flux=30.0):
    p = _calm(_find("smoke_only", "smoke_regime", "hot_plume", cfg))
    p["terrain"] = {"kind": "flat", "height_m": 0.0}
    p["atmosphere"]["lapse_dtheta_dz"] = 0.0     # neutral: a stable lapse traps the plume (physics)
    p["smoke"].update(buoyancy_flux_m3s3=flux, onset_s=-5.0, offset_s=None, intermittency=0.0,
                      source_x_frac=0.5, source_width_m=15.0, source_height_m=0.0,
                      emission_mg_m3_s=50.0, residual=None, on_hilltop=False)
    return p


# ------------------------------------------------------------------ ambient state
def test_ambient_state_is_steady_with_inversion_terrain_and_fog_layer():
    cfg = _short(CFG, n_frames=7)
    p = _calm(_find("fog_only", "fog_regime", "radiation_fog", cfg), spinup=60.0)
    p["terrain"] = {"kind": "hill", "height_m": 40.0, "center_frac": 0.5, "width_m": 50.0}
    p["fog"].update(inversion_dtheta_dz=0.06, surface_cooling_k_s=0.0, heterogeneity=0.0, top_undulation=0.0)
    p["mixing"]["diffusivity_m2s"] = 4.0
    spec = build_spec(p, cfg, dt=0.1)
    res = run_simulation(spec, cfg)
    fluid = ~res.solid
    assert np.abs(res.theta_anom[:, fluid]).max() < 0.01
    assert np.abs(res.ux[:, fluid]).max() < 0.01 and np.abs(res.uz[:, fluid]).max() < 0.01
    # The fog layer neither evaporates nor grows.
    assert abs(res.lwc[-1].sum() / res.lwc[0].sum() - 1.0) < 1e-3
    assert res.validity["valid"], res.validity["reasons"]


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_ambient_wind_profile_is_maintained_over_the_no_slip_floor(sign):
    nz = nx = 64
    units = LatticeUnits(5.0, 0.05)
    z = (np.arange(nz) + 0.5) * 5.0
    u_amb = sign * 5.0 * np.minimum((z / 50.0) ** 0.2, 1.3)
    nu, cs = 2.0, 0.15
    s = LBMSolver(LBMConfig(nx, nz, units, nu, smagorinsky_cs=cs), u_ambient=u_amb)
    a = ambient_maintaining_acceleration(u_amb, nu, 5.0, cs)
    ax = np.repeat(a[:, None], nx, axis=1)
    for _ in range(2000):
        s.step(ax, None)
    ux, uz = s.velocity()
    interior = ux[:, 10:54]
    assert np.all(np.sign(interior) == sign)                       # no reversed flow
    # Above the lowest cell the profile is held within a few per cent; the first cell
    # (2.5 m) sits in the wall's discretisation layer and is allowed to be slower.
    assert np.abs(interior[1:].mean(axis=1) - u_amb[1:]).max() < 0.05 * np.abs(u_amb).max()
    assert abs(interior[0].mean()) > 0.4 * abs(u_amb[0])
    assert np.abs(uz[:, 10:54]).max() < 0.1
    assert s.max_density_deviation() < 0.01


# ------------------------------------------------------------------ plumes
def test_hot_plume_rises_and_leaves_through_the_top():
    cfg = _short(CFG, n_frames=12)
    p = _hot_plume(cfg)
    p["smoke"]["onset_s"] = -200.0           # established plume (ignited 200 s before frame 0): the
    p["timing"]["spinup_s"] = 205.0          # starting-plume head has left the domain by then
    # Fixed mixing, so the test does not depend on which sample _find returns.
    p["mixing"].update(nu_m2s=2.3, smagorinsky_cs=0.15, diffusivity_m2s=1.0)
    res = run_scenario(p, cfg)
    assert res.validity["valid"], res.validity["reasons"]
    z = (np.arange(res.smoke.shape[1]) + 0.5) * res.dx
    heights = [(c.sum(axis=1) * z).sum() / c.sum() for c in res.smoke]
    assert heights[-1] > heights[0] + 10.0 or heights[0] > 80.0
    # The open top must not act as a lid: once the plume has reached it, smoke leaves
    # through the top about as fast as the source emits it (steady plume).
    nz, nx = res.smoke.shape[1:]
    emission_rate = res.validity["budgets"]["smoke"]["sources"] / (200.0 + 55.0)   # onset to last frame
    top_rate = float((np.maximum(res.uz[-1][-1], 0.0) * res.smoke[-1][-1]).sum() * res.dx)
    assert top_rate > 0.5 * emission_rate
    # No mushrooming ceiling layer spanning the domain under the top boundary.
    last = res.smoke[-1]

    def width(row):
        prof = last[row]
        return int((prof > 0.05 * prof.max()).sum()) if prof.max() > 0 else 0

    assert width(int(0.9 * nz)) < 0.75 * nx
    assert width(int(0.75 * nz)) <= 1.8 * max(width(int(0.5 * nz)), 1)


def test_smoldering_plume_stays_low_in_wind():
    cfg = _short(CFG, n_frames=6)
    p = _hot_plume(cfg, flux=0.05)
    p["wind"].update(speed_ms=5.0, sign=1.0)
    p["smoke"]["source_x_frac"] = 0.2
    p["timing"]["spinup_s"] = 40.0
    res = run_scenario(p, cfg)
    z = (np.arange(res.smoke.shape[1]) + 0.5) * res.dx
    c = res.smoke[-1]
    assert (c.sum(axis=1) * z).sum() / c.sum() < 60.0


# ------------------------------------------------------------------ fog
def test_established_fog_survives_and_patch_drifts_with_wind():
    cfg = _short(CFG, n_frames=6)
    p = _find("fog_only", "fog_regime", "fog_patch", cfg)
    p["wind"].update(speed_ms=3.0, sign=1.0, gust_fraction=0.0, flip=False)
    p["terrain"] = {"kind": "flat", "height_m": 0.0}
    p["fog"].update(center_frac=0.3, center_height_m=40.0, radius_m=50.0, ambient_rh=0.98, moist_inflow=False)
    p["timing"]["spinup_s"] = 10.0
    res = run_scenario(p, cfg)
    assert res.lwc[-1].sum() > 0.3 * res.lwc[0].sum()
    x = (np.arange(res.lwc.shape[2]) + 0.5) * res.dx
    cx = [(l.sum(axis=0) * x).sum() / l.sum() for l in res.lwc]
    assert cx[-1] > cx[0] + 20.0          # moved downwind (3 m/s for 25 s ~ 75 m)


@pytest.mark.parametrize("regime_key,regime,category", [("fog_regime", "advected_fog_bank", "fog_only"),
                                                         ("smoke_regime", "advected_smoke", "smoke_only")])
def test_advected_front_enters_on_time_and_crosses_the_view(regime_key, regime, category):
    """A front placed at 30 % of the view at the first frame is there, not already past the view.

    With 4 m/s for an 80 s spin-up the front starts ~270 m upwind of the
    domain: the inflow must wait for it instead of feeding the layer from t = 0.
    """
    cfg = _short(CFG, n_frames=12)
    p = _find(category, regime_key, regime, cfg)
    p["wind"].update(speed_ms=4.0, sign=-1.0, gust_fraction=0.0, flip=False, ramp_factor=1.0)
    p["terrain"] = {"kind": "flat", "height_m": 0.0}
    p["timing"]["spinup_s"] = 80.0
    (p["fog"] if category == "fog_only" else p["smoke"])["front_frac"] = 0.3
    res = run_scenario(p, cfg)
    field = res.lwc if category == "fog_only" else res.smoke
    thr = 0.02 if category == "fog_only" else 0.05 * float(field.max())
    view = field[:, :, 6:-6]
    cover = [float((view[t] > thr).any(axis=0).mean()) for t in range(view.shape[0])]
    assert 0.15 < cover[0] < 0.7
    assert cover[-1] > cover[0] + 0.25          # the front advances across the view


def test_supersaturated_inflow_condenses_into_fog():
    cfg = _short(CFG, n_frames=6)
    p = _find("fog_only", "fog_regime", "fog_patch", cfg)
    p["wind"].update(speed_ms=4.0, sign=1.0, gust_fraction=0.0, flip=False)
    p["terrain"] = {"kind": "flat", "height_m": 0.0}
    p["fog"].update(lwc_g_m3=1e-6, moist_inflow=True, inflow_rh=1.02, inflow_depth_m=60.0, ambient_rh=0.97)
    p["timing"]["spinup_s"] = 40.0
    spec = build_spec(p, cfg)
    res = run_simulation(spec, cfg)
    # Liquid appears where the supersaturated inflow enters (left side, low levels).
    assert res.lwc[-1][:8, :20].max() > 0.01


# ------------------------------------------------------------------ sampling
def test_sampling_is_deterministic_and_split_keyed():
    a = sample_scenario("smoke_only", "train", 3, CFG)
    b = sample_scenario("smoke_only", "train", 3, CFG)
    assert a == b
    seeds = {scenario_seed(CFG, s, c, 0) for s in SPLITS for c in CATEGORIES}
    assert len(seeds) == len(SPLITS) * len(CATEGORIES)
    assert sample_scenario("smoke_only", "val_select", 3, CFG)["seed"] != a["seed"]


def test_context_is_class_independent():
    speeds = {c: np.array([sample_scenario(c, "train", i, CFG)["wind"]["speed_ms"] for i in range(150)])
              for c in CATEGORIES}
    terrain = {c: np.mean([sample_scenario(c, "train", i, CFG)["terrain"]["kind"] != "flat" for i in range(150)])
               for c in CATEGORIES}
    spinups = {c: np.array([sample_scenario(c, "train", i, CFG)["timing"]["spinup_s"] for i in range(150)])
               for c in CATEGORIES}
    for c in CATEGORIES:
        assert abs(speeds[c].mean() - speeds["smoke_only"].mean()) < 0.6
        assert abs(spinups[c].mean() - spinups["fog_only"].mean()) < 8.0     # spin-up never depends on the class
        assert abs(terrain[c] - terrain["smoke_only"]) < 0.13
        assert speeds[c].max() <= 7.0
    # An accelerating wind is context too: drawn alike for every class, never beyond the core range.
    ramps = {c: np.array([sample_scenario(c, "train", i, CFG)["wind"]["ramp_factor"] for i in range(300)])
             for c in CATEGORIES}
    for c in CATEGORIES:
        assert 0.03 < np.mean(ramps[c] > 1.0) < 0.2
    for i in range(200):
        w = sample_scenario(CATEGORIES[i % 4], "train", i, CFG)["wind"]
        assert w["speed_ms"] * w["ramp_factor"] <= 7.0 + 1e-9


def test_holdout_recipes_are_reachable_and_never_in_training():
    for c in CATEGORIES:
        tags = set()
        for i in range(6):
            p = sample_scenario(c, "test_ood", i, CFG)
            tags.update(p["holdout_tags"])
            if "fast_wind" in p["holdout_tags"]:
                assert p["wind"]["speed_ms"] >= 7.0
            if "elevated_source" in p["holdout_tags"]:
                assert p["terrain"]["height_m"] >= 45.0
        assert tags == set(HOLDOUT_TAGS)
    for i in range(100):
        p = sample_scenario(CATEGORIES[i % 4], "train", i, CFG)
        assert p["wind"]["speed_ms"] <= 7.0 and p["terrain"]["height_m"] <= 40.0 and not p["holdout_tags"]


def _sampled_regimes(cfg, n):
    seen = set()
    for i in range(n):
        for c in CATEGORIES:
            p = sample_scenario(c, "train", i, cfg)
            for key in ("smoke_regime", "fog_regime", "distractor_regime"):
                if p[key]:
                    seen.add(p[key])
    return seen


def test_every_regime_can_be_sampled():
    # Updated for engine v2: the opt-in regimes (ridge_cloud, rain_shaft) have weight 0 in
    # the near profile and must never appear there; every other regime still must.
    seen = _sampled_regimes(CFG, 400)
    weights = CFG["scenarios"]["regime_weights"]
    enabled = {r for cat, regs in (("smoke_only", SMOKE_REGIMES), ("fog_only", FOG_REGIMES), ("neither", NEITHER_REGIMES))
               for r in regs if regime_weight(r, weights[cat]) > 0}
    assert enabled == (set(SMOKE_REGIMES) | set(FOG_REGIMES) | set(NEITHER_REGIMES)) - set(OPT_IN_REGIMES)
    assert seen == enabled


def test_every_regime_can_be_sampled_in_the_far_profile():
    assert _sampled_regimes(FAR_CFG, 300) == set(SMOKE_REGIMES) | set(FOG_REGIMES) | set(NEITHER_REGIMES)


def test_regime_weights_absent_means_never_sampled():
    assert regime_weight("ridge_cloud", {"fog_patch": 1.0}) == 0.0          # missing from a block
    assert regime_weight("fog_patch", {"fog_patch": 0.5}) == 0.5
    assert regime_weight("fog_patch", None) == 1.0                          # no block: uniform ...
    assert regime_weight("rain_shaft", None) == 0.0                         # ... except opt-in regimes


def test_near_field_lookalikes_are_not_under_sampled():
    """illumination_change, camera_artifact and distant_cloud each make >= ~12 % of neither scenes."""
    n = 1500
    counts = collections.Counter(sample_scenario("neither", "train", i, CFG)["distractor_regime"] for i in range(n))
    for regime in ("illumination_change", "camera_artifact", "distant_cloud"):
        assert counts[regime] / n >= 0.12, (regime, counts)
    assert counts["haze_drift"] / n > 0.25 and counts["dust_burst"] / n > 0.15


@pytest.mark.parametrize("profile", ["near", "far"])
def test_nuisance_hints_are_class_independent(profile):
    cfg = CFG if profile == "near" else FAR_CFG
    n = 400
    rates = {c: np.mean([[sample_scenario(c, "train", i, cfg)["render_hints"][h]
                          for h in ("shadow", "strong_artifact", "fp_boost", "distant_cloud")] for i in range(n)], axis=0)
             for c in CATEGORIES}
    nuis = cfg["scenarios"]["nuisance"]
    expected = np.array([nuis["shadow_prob"], nuis["strong_artifact_prob"], nuis["fp_boost_prob"], nuis["distant_cloud_prob"]])
    for c in CATEGORIES:
        assert np.abs(rates[c] - expected).max() < 0.07, (c, rates[c])


def test_run_is_deterministic():
    cfg = _short(CFG, n_frames=3)
    p = sample_scenario("both", "train", 1, cfg)
    p["timing"]["spinup_s"] = 10.0
    r1 = run_scenario(p, cfg)
    r2 = run_scenario(p, cfg)
    assert np.array_equal(r1.smoke, r2.smoke) and np.array_equal(r1.lwc, r2.lwc)
    assert np.array_equal(r1.texcoords, r2.texcoords)


# ------------------------------------------------------------------ engine v2: lag frames, SGS velocity
def test_profile_without_lag_frames_stores_none_and_round_trips(tmp_path):
    cfg = _short(CFG, n_frames=3)
    p = sample_scenario("smoke_only", "train", 2, cfg)
    p["timing"]["spinup_s"] = 10.0
    with_lag = run_scenario(p, cfg)
    assert with_lag.has_lag and with_lag.sgs_u_lag is not None and with_lag.sgs_u_lag.shape == with_lag.smoke.shape
    for value in (None, 0.0):
        cfg["time"]["flow_lag_s"] = value
        res = run_scenario(p, cfg)
        assert not res.has_lag
        assert all(getattr(res, name) is None for name in ALL_LAG_FIELDS)
        assert res.sgs_u is not None and res.sgs_u.shape == res.smoke.shape
        # The observation snapshots do not depend on whether lag snapshots are taken.
        assert np.array_equal(res.smoke, with_lag.smoke) and np.array_equal(res.texcoords, with_lag.texcoords)
        with pytest.raises(MissingLagFields, match="no lag snapshots"):
            res.field_at("smoke", lag=True)
        assert res.field_at("lwc") is res.lwc
    path = res.save(tmp_path / "nolag.npz")
    back = SimulationResult.load(path)
    assert not back.has_lag and all(getattr(back, name) is None for name in ALL_LAG_FIELDS)
    assert np.array_equal(back.smoke, res.smoke) and back.valid == res.valid
    np.testing.assert_allclose(back.sgs_u, res.sgs_u, rtol=2e-3, atol=1e-4)      # float16 on disk
    back_lag = SimulationResult.load(with_lag.save(tmp_path / "lag.npz"))
    assert back_lag.has_lag and np.array_equal(back_lag.smoke_lag, with_lag.smoke_lag)
    np.testing.assert_allclose(back_lag.sgs_u_lag, with_lag.sgs_u_lag, rtol=2e-3, atol=1e-4)
    # A record with lag fields but no lag times is inconsistent.
    with pytest.raises(ValueError):
        SimulationResult(sim_id="x", category="smoke_only", regime="r", labels=(1, 0), seed=0, params={},
                         dx=5.0, times=res.times, smoke=res.smoke, lwc=res.lwc, tracer=res.tracer, solid=res.solid,
                         valid=True, validity={}, texcoords=res.texcoords, tex_weights=res.tex_weights,
                         smoke_lag=res.smoke)


def test_sgs_velocity_is_larger_inside_a_plume_than_in_calm_ambient():
    cfg = _short(CFG, n_frames=3)
    p = _hot_plume(cfg)
    p["smoke"]["onset_s"] = -25.0
    p["timing"]["spinup_s"] = 30.0
    p["mixing"].update(nu_m2s=2.3, smagorinsky_cs=0.15, diffusivity_m2s=1.0)
    res = run_scenario(p, cfg)
    assert res.validity["finite"] and res.validity["max_mach"] < 0.25      # 3 frames: too few for label support
    assert np.isfinite(res.sgs_u).all() and res.sgs_u.min() >= 0.0
    assert np.all(res.sgs_u[:, res.solid] == 0.0)
    last_smoke, last_sgs = res.smoke[-1], res.sgs_u[-1]
    plume = last_smoke > 0.05 * last_smoke.max()
    calm = (last_smoke < 1e-4 * last_smoke.max()) & ~res.solid
    calm[:, 20:44] = False                                   # stay well away from the plume column
    assert plume.sum() > 10 and calm.sum() > 100
    assert last_sgs[plume].mean() > 5.0 * last_sgs[calm].mean()
    assert 0.0 < last_sgs.max() < 5.0                        # m/s: a fraction of the plume velocity


# ------------------------------------------------------------------ engine v2: far-field profile
def _far_short(n_frames=6, label_min_frames=4):
    cfg = copy.deepcopy(FAR_CFG)
    cfg["time"]["n_frames"] = n_frames
    cfg["time"]["precursor_s"] = 60.0
    cfg["validation"]["label_min_frames"] = label_min_frames     # the shortened window has fewer frames
    return cfg


def _find_far(category, regime_key, regime, cfg, limit=600):
    return _find(category, regime_key, regime, cfg, limit=limit)


def test_far_profile_samples_far_field_scales():
    cfg = FAR_CFG
    dx = cfg["grid"]["domain_width_m"] / cfg["grid"]["nx"]
    assert dx == 50.0 and cfg["time"]["frame_interval_s"] == 60.0 and cfg["time"]["flow_lag_s"] is None
    kinds = set()
    for i in range(80):
        for c in CATEGORIES:
            p = sample_scenario(c, "train", i, cfg)
            assert p["wind"]["speed_ms"] * p["wind"]["ramp_factor"] <= 12.0 + 1e-9
            assert p["timing"]["frame_interval_s"] == 60.0 and 120.0 <= p["timing"]["spinup_s"] <= 900.0
            assert 15.0 <= p["mixing"]["nu_m2s"] <= 40.0
            t = p["terrain"]
            kinds.add(t["kind"])
            if t["kind"] != "flat":
                assert 50.0 <= t["height_m"] <= 600.0 and t["width_m"] >= 250.0
            smoke = p["smoke"]
            if smoke and smoke["regime"] != "advected_smoke":
                assert 50.0 <= smoke["source_width_m"] <= 400.0
                assert smoke["onset_ramp_s"] == 36.0                 # 3 s x (60 s / 5 s)
            if p["fog"] and p["fog"]["regime"] not in ("steam_fog", "ridge_cloud"):
                assert 50.0 <= p["fog"]["fog_top_m"] <= 800.0
            haze = p["background_haze"]
            if haze:
                assert haze["texture_scale_m"] >= 300.0 and haze["layer_depth_m"] >= 400.0
    assert kinds == {"flat", "hill", "valley", "slope", "ridge"}
    for c in CATEGORIES:
        for i in range(4):
            p = sample_scenario(c, "test_ood", i, cfg)
            if "fast_wind" in p["holdout_tags"]:
                assert p["wind"]["speed_ms"] >= 12.0
            if "elevated_source" in p["holdout_tags"]:
                assert p["terrain"]["height_m"] >= 700.0


@pytest.mark.parametrize("category,regime_key,regime", [("smoke_only", "smoke_regime", "hot_plume"),
                                                         ("fog_only", "fog_regime", "advected_fog_bank")])
def test_short_far_run_validates(category, regime_key, regime):
    cfg = _far_short()
    p = _find_far(category, regime_key, regime, cfg)
    # Calm for the plume (its rise is then unambiguous; in a crosswind this coarse 2D source
    # makes a low, bent-over warm layer in both profiles), a light wind for the drifting bank.
    p["wind"].update(speed_ms=0.0 if category == "smoke_only" else 3.0, gust_fraction=0.0, ramp_factor=1.0, flip=False)
    p["terrain"] = {"kind": "flat", "height_m": 0.0}
    p["timing"]["spinup_s"] = 300.0
    if category == "smoke_only":
        p["smoke"].update(onset_s=-240.0, offset_s=None, source_x_frac=0.5, buoyancy_flux_m3s3=100.0)
    else:
        p["fog"]["front_frac"] = 0.4
    res = run_scenario(p, cfg)
    assert res.valid, res.validity["reasons"]
    assert res.dx == 50.0 and not res.has_lag
    assert np.allclose(np.diff(res.times), 60.0)
    assert res.sgs_u.shape == res.smoke.shape and res.sgs_u.min() >= 0.0
    if category == "smoke_only":
        z = (np.arange(res.smoke.shape[1]) + 0.5) * res.dx
        c = res.smoke[-1]
        assert (c.sum(axis=1) * z).sum() / c.sum() > 300.0          # a wildfire plume rises hundreds of metres
        assert res.validity["max_theta_anomaly_k"] <= FAR_CFG["physics"]["heat_cap_k"] + 1.0


def test_ridge_cloud_condenses_over_the_ridge():
    cfg = _far_short(n_frames=4, label_min_frames=3)
    p = _find_far("fog_only", "fog_regime", "ridge_cloud", cfg)
    p["terrain"] = {"kind": "ridge", "height_m": 450.0, "center_frac": 0.5, "width_m": 400.0,
                    "direction": 1.0, "lee_width_frac": 0.7}
    p["wind"].update(speed_ms=8.0, sign=1.0, gust_fraction=0.0, ramp_factor=1.0, flip=False)
    p["atmosphere"]["lapse_dtheta_dz"] = 0.003
    p["fog"].update(lcl_frac=0.4, moist_top_frac=2.0)
    p["timing"]["spinup_s"] = 300.0
    p["background_haze"] = None
    spec = build_spec(p, cfg)
    assert np.all(spec.ql_ref == 0.0) and np.all(spec.initial["ql"] == 0.0)      # nothing prescribed
    res = run_simulation(spec, cfg)
    assert res.valid, res.validity["reasons"]
    lwc = res.lwc[-1]
    assert lwc.max() > 0.05
    x = (np.arange(lwc.shape[1]) + 0.5) * res.dx
    crest = x[np.argmax(res.solid.sum(axis=0))]
    cx = (lwc.sum(axis=0) * x).sum() / lwc.sum()
    assert abs(cx - crest) < 600.0                                   # anchored over the ridge ...
    assert lwc[:, :14].max() < 1e-3                                  # ... not in the sub-saturated inflow
    assert res.validity["label_support"]["fog_visible_frames"] == res.n_frames


def test_rain_shaft_falls_from_cloud_base():
    cfg = _far_short(n_frames=4)
    p = _find_far("neither", "distractor_regime", "rain_shaft", cfg)
    p["wind"].update(speed_ms=2.0, sign=1.0, gust_fraction=0.0, ramp_factor=1.0, flip=False)
    p["terrain"] = {"kind": "flat", "height_m": 0.0}
    p["background_haze"] = None
    p["timing"]["spinup_s"] = 120.0
    p["distractor"].update(cloud_base_m=2200.0, fall_speed_ms=5.0, onset_s=-60.0, intermittency=0.0,
                           evaporation_rate_s=0.0, center_frac=0.5)
    res = run_scenario(p, cfg)
    assert res.valid, res.validity["reasons"]
    assert res.smoke.max() == 0.0 and res.lwc.max() == 0.0
    z = (np.arange(res.tracer.shape[1]) + 0.5) * res.dx
    heights = [float((c.sum(axis=1) * z).sum() / c.sum()) for c in res.tracer]
    assert all(b < a - 50.0 for a, b in zip(heights, heights[1:]))     # the curtain descends frame by frame
    lowest = [float(z[np.nonzero(c.max(axis=1) > 0.01 * c.max())[0].min()]) for c in res.tracer]
    assert lowest[-1] < lowest[0] - 400.0 and lowest[-1] < 1000.0
    assert "precip" in res.validity["budgets"] and abs(res.validity["mass_residuals"]["precip"]) < 1e-6


# ------------------------------------------------------------------ slow
@pytest.mark.slow
@pytest.mark.parametrize("category", CATEGORIES)
def test_categories_run_and_mostly_validate(category):
    cfg = _short(CFG, n_frames=12)
    valid = 0
    for i in range(4):
        p = sample_scenario(category, "train", i, cfg)
        res = run_scenario(p, cfg)
        assert res.validity["finite"]
        valid += res.valid
    assert valid >= 2


@pytest.mark.slow
@pytest.mark.parametrize("category", CATEGORIES)
def test_far_profile_categories_run_and_mostly_validate(category):
    cfg = copy.deepcopy(FAR_CFG)                    # the full 12-frame, one-frame-per-minute window
    valid = 0
    for i in range(3):
        res = run_scenario(sample_scenario(category, "train", i, cfg), cfg)
        assert res.validity["finite"] and not res.has_lag
        valid += res.valid
    assert valid >= 2


@pytest.mark.slow
def test_plume_rise_converges_with_grid_resolution():
    heights = {}
    for n in (32, 64, 128):
        cfg = _short(CFG, n_frames=7)
        cfg["grid"]["nx"] = cfg["grid"]["nz"] = n
        cfg["render"]["view_margin_cells"] = max(2, 6 * n // 64)
        cfg["lbm"]["sponge_width_cells"] = max(2, 6 * n // 64)
        p = _hot_plume(cfg, flux=20.0)
        p["smoke"]["source_width_m"] = 25.0       # at least two coarse cells
        p["mixing"].update(nu_m2s=3.0, smagorinsky_cs=0.15, diffusivity_m2s=2.0)
        spec = build_spec(p, cfg, dt=0.05 * 64 / n)
        res = run_simulation(spec, cfg)
        z = (np.arange(n) + 0.5) * res.dx
        c = res.smoke[3]                              # 15 s after onset + spin-up
        heights[n] = float((c.sum(axis=1) * z).sum() / c.sum())
    assert abs(heights[128] - heights[64]) < abs(heights[64] - heights[32]) + 5.0
