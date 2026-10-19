"""Coupled-simulation checks: steady ambient state, plumes, fog, budgets, scenarios."""

import copy

import numpy as np
import pytest

from src.cfd.lbm import LatticeUnits, LBMConfig, LBMSolver, ambient_maintaining_acceleration
from src.cfd.scenarios import (
    HOLDOUT_TAGS, SPLITS, build_spec, run_scenario, sample_scenario, scenario_seed,
)
from src.cfd.simulation import run_simulation
from src.constants import CATEGORIES, FOG_REGIMES, NEITHER_REGIMES, SMOKE_REGIMES
from src.utils.config import load_simulation_config

CFG = load_simulation_config()


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


def test_every_regime_can_be_sampled():
    seen = set()
    for i in range(400):
        for c in CATEGORIES:
            p = sample_scenario(c, "train", i, CFG)
            for key in ("smoke_regime", "fog_regime", "distractor_regime"):
                if p[key]:
                    seen.add(p[key])
    assert set(SMOKE_REGIMES) | set(FOG_REGIMES) | set(NEITHER_REGIMES) <= seen


def test_run_is_deterministic():
    cfg = _short(CFG, n_frames=3)
    p = sample_scenario("both", "train", 1, cfg)
    p["timing"]["spinup_s"] = 10.0
    r1 = run_scenario(p, cfg)
    r2 = run_scenario(p, cfg)
    assert np.array_equal(r1.smoke, r2.smoke) and np.array_equal(r1.lwc, r2.lwc)
    assert np.array_equal(r1.texcoords, r2.texcoords)


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
