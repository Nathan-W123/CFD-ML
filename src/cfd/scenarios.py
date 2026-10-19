"""Scenario sampling and assembly: parameters -> SimulationSpec -> SimulationResult.

Sampling is context-first, to keep label shortcuts out of the data:

1. Class-independent context: wind speed and direction, gusts, shear,
   mixing, terrain, atmosphere, spin-up, background haze and the camera
   nuisances handed to the renderer (moving shadow, strong camera artefacts,
   detector false-positive boost). These are drawn from the SAME
   distributions for every category, so e.g. |wind| or P(terrain) carries no
   information about the label.
2. A regime compatible with that context is chosen for the category (a calm
   radiation fog needs light wind, terrain fog needs a valley, a "camera
   artefact" negative needs a strong artefact to exist, ...), with the
   configured regime weights renormalised over the compatible regimes.
3. Component parameters (smoke source, fog layer, distractor) are sampled.

Seeds: every simulation's seed is derived from (base seed, split, category,
index, attempt), and every component draws from its own child stream, so no
two simulations in different splits share random streams and a rejected
simulation is replaced by re-drawing with the next attempt number.

Held-out (test_ood) recipes, never seen in train / val / test_iid:
* "fast_wind"        wind of 7-9 m/s (training wind is 0-7 m/s) in EVERY
                     category, so fast-moving fog has matched fast negatives;
* "elevated_source"  a tall hill (45-80 m; training terrain is <= 40 m) in
                     every category, with the smoke source on the hilltop
                     for smoke-bearing categories.

Where this fits: `scripts/generate_dataset.py` calls `sample_scenario` and
`run_scenario`; tests call `build_spec` with small step budgets.
Calls into: smoke, fog, simulation, thermo, scenario_util, runtime seeding.
Start reading at: sample_scenario.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.cfd import thermo
from src.cfd.fog import fog_component, fog_reference, fog_wind_range, sample_fog
from src.cfd.results import SimulationResult
from src.cfd.scenario_util import (
    Component, Grid, layer_profile, log_uniform, sample_noise_spec, smooth_noise, smooth_noise01,
    smooth_texture, uniform,
)
from src.cfd.simulation import SimulationSpec, SimulationUnstable, run_simulation
from src.cfd.smoke import (
    dust_component, sample_dust, sample_smoke, smoke_component, smoke_wind_range,
)
from src.constants import CATEGORIES, CATEGORY_INDEX, CATEGORY_LABELS, FOG_REGIMES, SMOKE_REGIMES
from src.utils.runtime import child_rng, derive_seed

SPLITS: Tuple[str, ...] = ("train", "val_select", "val_calib", "test_iid", "test_ood")
SPLIT_KEYS = {name: i + 1 for i, name in enumerate(SPLITS)}
HOLDOUT_TAGS: Tuple[str, ...] = ("fast_wind", "elevated_source")
_STREAM_CONTEXT, _STREAM_REGIME, _STREAM_SMOKE, _STREAM_FOG, _STREAM_DISTRACTOR = 1, 2, 3, 4, 5
CALM_ONLY_FOG = ("radiation_fog", "terrain_fog")


# ---------------------------------------------------------------------- helpers
def _sc(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return cfg.get("scenarios", {})


def _holdout(cfg: Dict[str, Any]) -> Dict[str, Any]:
    h = cfg.get("holdout", {})
    return {
        "fast_wind_ms": tuple(h.get("fast_wind", {}).get("wind_speed_ms", [7.0, 9.0])),
        "tall_terrain_m": tuple(h.get("elevated_source", {}).get("terrain_height_m", [45.0, 80.0])),
    }


def scenario_seed(cfg: Dict[str, Any], split: str, category: str, index: int, attempt: int = 0) -> int:
    return derive_seed(int(cfg["seed"]), SPLIT_KEYS[split], CATEGORY_INDEX[category], int(index), int(attempt))


def sim_id_for(split: str, category: str, index: int, attempt: int = 0) -> str:
    return f"{split}-{category}-{index:04d}-a{attempt}"


def _weighted_choice(rng: np.random.Generator, options: List[str], weights: Dict[str, float]) -> str:
    w = np.array([float(weights.get(o, 1.0)) for o in options], dtype=float)
    if not len(options) or w.sum() <= 0:
        raise ValueError("no compatible regime")
    return options[int(rng.choice(len(options), p=w / w.sum()))]


def _in_range(value: float, bounds) -> bool:
    return float(bounds[0]) - 1e-9 <= value <= float(bounds[1]) + 1e-9


# ---------------------------------------------------------------------- context
def sample_context(rng: np.random.Generator, cfg: Dict[str, Any], category: str,
                   holdout_tags: List[str]) -> Dict[str, Any]:
    """Class-independent conditions (the category only decides the humidity range)."""
    common = _sc(cfg).get("common", {})
    hold = _holdout(cfg)
    fast = "fast_wind" in holdout_tags
    core_wind = common.get("wind_speed_ms", [0.0, hold["fast_wind_ms"][0]])
    speed = uniform(rng, hold["fast_wind_ms"] if fast else (core_wind[0], min(core_wind[1], hold["fast_wind_ms"][0])))
    wind = {
        "speed_ms": speed,
        "sign": 1.0 if rng.random() < 0.5 else -1.0,
        "shear_exponent": uniform(rng, common.get("wind_shear_exponent", [0.1, 0.3])),
        "gust_fraction": uniform(rng, common.get("wind_gust_fraction", [0.0, 0.3])),
        "gusts": sample_noise_spec(rng, common.get("wind_gust_period_s", [8.0, 40.0])),
        "flip": bool(speed < 2.0 and rng.random() < float(common.get("wind_direction_flip_prob", 0.05))),
    }
    # An accelerating wind over the observation window, for every category alike
    # (fog_dissipating may use it as its clearing mechanism, it never creates it).
    # The end-of-window speed stays inside the split's wind range, so the ramp
    # never reaches the fast_wind hold-out outside test_ood.
    ramp_drawn = rng.random() < float(common.get("wind_ramp_prob", 0.1))
    ramp = uniform(rng, common.get("wind_ramp_factor", [1.3, 2.0]))
    cap = hold["fast_wind_ms"][1] if fast else hold["fast_wind_ms"][0]
    wind["ramp_factor"] = float(min(ramp, max(1.0, cap / speed))) if ramp_drawn and speed > 0.0 else 1.0
    mixing = {
        "nu_m2s": uniform(rng, common.get("eddy_viscosity_m2s", [1.5, 3.0])),
        "smagorinsky_cs": uniform(rng, common.get("smagorinsky_cs", [0.1, 0.2])),
        "diffusivity_m2s": log_uniform(rng, common.get("scalar_diffusivity_m2s", [0.3, 4.0])),
    }
    if "elevated_source" in holdout_tags:
        terrain = {"kind": "hill", "height_m": uniform(rng, hold["tall_terrain_m"]),
                   "center_frac": uniform(rng, (0.3, 0.7)), "width_m": uniform(rng, (35.0, 70.0))}
    elif rng.random() < float(common.get("terrain_prob", 0.35)):
        kind = ["hill", "valley", "slope"][int(rng.integers(0, 3))]
        terrain = {"kind": kind, "height_m": uniform(rng, common.get("terrain_height_m", [5.0, 40.0])),
                   "center_frac": uniform(rng, (0.25, 0.75)),
                   "width_m": uniform(rng, (30.0, 80.0) if kind == "hill" else (50.0, 110.0)),
                   "direction": 1.0 if rng.random() < 0.5 else -1.0}
    else:
        terrain = {"kind": "flat", "height_m": 0.0}
    fog_present = CATEGORY_LABELS[category][1] == 1
    atmosphere = {
        "theta_ground_k": uniform(rng, common.get("theta_ground_k", [272.0, 300.0])),
        "lapse_dtheta_dz": uniform(rng, common.get("lapse_dtheta_dz", [-0.002, 0.03])),
        # Non-fog scenes stay well below saturation so lifted air cannot condense
        # into unlabelled cloud; fog scenes set their own humidity in fog.py.
        "relative_humidity": uniform(rng, (0.6, 0.95) if fog_present else common.get("relative_humidity_dry", [0.3, 0.7])),
    }
    spin = common.get("spinup_s", cfg.get("time", {}).get("spinup_s", [10.0, 60.0]))
    timing = {
        "spinup_s": uniform(rng, spin),
        "frame_interval_s": float(cfg.get("time", {}).get("frame_interval_s", 5.0)),
        "n_frames": int(cfg.get("time", {}).get("n_frames", 12)),
    }
    nuis = _sc(cfg).get("nuisance", {})
    hints = {
        "shadow": bool(rng.random() < float(nuis.get("shadow_prob", 0.25))),
        "strong_artifact": bool(rng.random() < float(nuis.get("strong_artifact_prob", 0.15))),
        "fp_boost": bool(rng.random() < float(nuis.get("fp_boost_prob", 0.2))),
        "distant_cloud": bool(rng.random() < float(nuis.get("distant_cloud_prob", 0.15))),
    }
    haze = None
    if rng.random() < float(common.get("background_haze_prob", 0.3)):
        haze = _sample_haze(rng, background=True)
    return {"wind": wind, "mixing": mixing, "terrain": terrain, "atmosphere": atmosphere,
            "timing": timing, "render_hints": hints, "background_haze": haze}


def _sample_haze(rng: np.random.Generator, background: bool, block: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    block = block or {}
    return {
        "background": bool(background),
        "inflow_mg_m3": uniform(rng, (0.05, 1.0) if background else block.get("inflow_mg_m3", [0.2, 4.0])),
        "layer_base_m": uniform(rng, (0.0, 100.0)),
        "layer_depth_m": uniform(rng, block.get("layer_depth_m", [40.0, 250.0])),
        "patchiness": uniform(rng, (0.2, 0.8)),
        "texture_seed": int(rng.integers(0, 2**31 - 1)),
        "texture_scale_m": uniform(rng, (30.0, 120.0)),
        "modulation": sample_noise_spec(rng, (8.0, 40.0)),
    }


# ---------------------------------------------------------------------- regimes
def compatible_smoke_regimes(context: Dict[str, Any], cfg: Dict[str, Any], tags: List[str]) -> List[str]:
    speed = context["wind"]["speed_ms"]
    out = []
    for r in SMOKE_REGIMES:
        if "elevated_source" in tags and r == "advected_smoke":
            continue
        if "fast_wind" in tags or _in_range(speed, smoke_wind_range(r, cfg)):
            out.append(r)
    return out


def compatible_fog_regimes(context: Dict[str, Any], cfg: Dict[str, Any], tags: List[str]) -> List[str]:
    speed = context["wind"]["speed_ms"]
    kind = context["terrain"]["kind"]
    out = []
    for r in FOG_REGIMES:
        if r == "terrain_fog" and kind != "valley":
            continue
        if r == "steam_fog" and kind != "flat":
            continue
        if "fast_wind" in tags:
            if r in CALM_ONLY_FOG:
                continue
        elif not _in_range(speed, fog_wind_range(r, cfg)):
            continue
        out.append(r)
    return out


def compatible_neither_regimes(context: Dict[str, Any], cfg: Dict[str, Any], tags: List[str]) -> List[str]:
    hints = context["render_hints"]
    speed = context["wind"]["speed_ms"]
    dust_range = _sc(cfg).get("neither", {}).get("dust_burst", {}).get("wind_speed_ms", [1.5, 9.0])
    out = ["haze_drift"]
    if "fast_wind" in tags or _in_range(speed, dust_range):
        out.append("dust_burst")
    if hints["shadow"]:
        out.append("illumination_change")
    if hints["strong_artifact"]:
        out.append("camera_artifact")
    if hints["distant_cloud"]:
        out.append("distant_cloud")
    return out


def _ood_tag(category: str, index: int) -> List[str]:
    return [HOLDOUT_TAGS[index % len(HOLDOUT_TAGS)]]


# ---------------------------------------------------------------------- sampling
def sample_scenario(category: str, split: str, index: int, cfg: Dict[str, Any], attempt: int = 0) -> Dict[str, Any]:
    """All parameters of one simulation (JSON-able); deterministic in its arguments."""
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    seed = scenario_seed(cfg, split, category, index, attempt)
    tags = _ood_tag(category, index) if split == "test_ood" else []
    context = sample_context(child_rng(seed, _STREAM_CONTEXT), cfg, category, tags)
    rng = child_rng(seed, _STREAM_REGIME)
    weights = _sc(cfg).get("regime_weights", {})
    smoke_label, fog_label = CATEGORY_LABELS[category]
    smoke = fog = distractor = None
    smoke_regime = fog_regime = None
    if fog_label:
        options = compatible_fog_regimes(context, cfg, tags)
        fog_regime = _weighted_choice(rng, options, weights.get("fog_only", {}))
        fog = sample_fog(fog_regime, child_rng(seed, _STREAM_FOG), cfg, context)
    if smoke_label:
        options = compatible_smoke_regimes(context, cfg, tags)
        smoke_regime = _weighted_choice(rng, options, weights.get("smoke_only", {}))
        smoke = sample_smoke(smoke_regime, child_rng(seed, _STREAM_SMOKE), cfg, {**context, "holdout_tags": tags})
    if category == "neither":
        options = compatible_neither_regimes(context, cfg, tags)
        regime = _weighted_choice(rng, options, weights.get("neither", {}))
        drng = child_rng(seed, _STREAM_DISTRACTOR)
        if regime == "haze_drift":
            distractor = {"regime": regime, "haze": _sample_haze(drng, background=False,
                                                                 block=_sc(cfg).get("neither", {}).get("haze_drift", {}))}
        elif regime == "dust_burst":
            distractor = sample_dust(drng, cfg, context)
        else:
            distractor = {"regime": regime}
    else:
        regime = "+".join(r for r in (smoke_regime, fog_regime) if r)
    return {
        "sim_id": sim_id_for(split, category, index, attempt),
        "category": category,
        "split": split,
        "index": int(index),
        "attempt": int(attempt),
        "seed": int(seed),
        "labels": [int(smoke_label), int(fog_label)],
        "regime": regime,
        "smoke_regime": smoke_regime,
        "fog_regime": fog_regime,
        "distractor_regime": distractor["regime"] if distractor else None,
        "holdout_tags": tags,
        **context,
        "smoke": smoke,
        "fog": fog,
        "distractor": distractor,
    }


# ---------------------------------------------------------------------- assembly
def terrain_heights(terrain: Dict[str, Any], x: np.ndarray, view_x) -> np.ndarray:
    kind = terrain["kind"]
    if kind == "flat":
        return np.zeros_like(x)
    H = float(terrain["height_m"])
    xc = view_x(terrain["center_frac"])
    w = float(terrain["width_m"])
    if kind == "hill":
        return H * np.exp(-((x - xc) / w) ** 2)
    if kind == "valley":
        return H * (1.0 - np.exp(-((x - xc) / w) ** 2))
    if kind == "slope":
        s = terrain.get("direction", 1.0) * (x - xc) / w
        return H * 0.5 * (1.0 + np.tanh(s))
    raise ValueError(f"unknown terrain kind {kind!r}")


def build_grid(params: Dict[str, Any], cfg: Dict[str, Any]) -> Grid:
    g = cfg.get("grid", {})
    nx, nz = int(g.get("nx", 64)), int(g.get("nz", 64))
    dx = float(g.get("domain_width_m", 320.0)) / nx
    margin = int(cfg.get("render", {}).get("view_margin_cells", cfg.get("lbm", {}).get("sponge_width_cells", 6)))
    probe = Grid(nx, nz, dx, np.zeros((nz, nx), dtype=bool), margin)
    h = terrain_heights(params["terrain"], probe.x, probe.view_x)
    ncells = np.rint(h / dx).astype(int)
    solid = np.arange(nz)[:, None] < ncells[None, :]
    return Grid(nx, nz, dx, solid, margin, wind_sign=float(params["wind"]["sign"]))


def wind_function(params: Dict[str, Any], grid: Grid):
    """Mean ambient wind profile u(z, t) in m/s: power law, gusts, optional ramp or slow reversal."""
    w = params["wind"]
    U, sign = float(w["speed_ms"]), float(w["sign"])
    prof = np.minimum((grid.z / 50.0) ** float(w["shear_exponent"]), 1.3)
    gust_f = float(w["gust_fraction"])
    first = float(params["timing"]["spinup_s"])
    span = (params["timing"]["n_frames"] - 1) * params["timing"]["frame_interval_s"]
    ramp = float(w.get("ramp_factor", 1.0))
    flip = bool(w.get("flip", False))

    def wind(t: float) -> np.ndarray:
        s = min(max((t - first) / max(span, 1.0), 0.0), 1.0)
        mult = 1.0 + gust_f * smooth_noise(t, w["gusts"])
        if ramp != 1.0:
            mult *= 1.0 + (ramp - 1.0) * s
        direction = sign * (math.cos(math.pi * s) if flip else 1.0)
        return direction * U * max(mult, 0.0) * prof

    return wind


def haze_component(haze: Dict[str, Any], grid: Grid) -> Component:
    comp = Component()
    profile = layer_profile(grid.z, haze["layer_base_m"], haze["layer_base_m"] + haze["layer_depth_m"], soft_m=20.0)
    tex = smooth_texture(grid, haze["texture_seed"], scale_m=haze["texture_scale_m"], contrast=haze["patchiness"])
    c0 = haze["inflow_mg_m3"]
    init = c0 * profile[:, None] * tex
    mod = haze["modulation"]

    def initial(fields):
        fields["tracer"] += np.where(grid.fluid, init, 0.0)

    def inflow(t, ghosts):
        side = "left" if grid.wind_sign >= 0 else "right"
        level = c0 * (1.0 - 0.5 * haze["patchiness"] + haze["patchiness"] * smooth_noise01(t, mod))
        ghosts.setdefault("tracer", {})[side] = level * profile

    comp.initial.append(initial)
    comp.inflow.append(inflow)
    return comp


def expected_peak_speed(params: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """A conservative estimate of the largest flow speed (m/s), used to pick dt:
    max_z u_amb (1 + 1.5 gust) combined with the plume rise 1.66 F^(1/3)."""
    w = params["wind"]
    U = float(w["speed_ms"]) * 1.3 * float(w.get("ramp_factor", 1.0)) * (1.0 + 1.5 * float(w["gust_fraction"]))
    plume = 0.0
    smoke = params.get("smoke")
    if smoke and smoke.get("buoyancy_flux_m3s3", 0.0) > 0:
        plume = 1.66 * float(smoke["buoyancy_flux_m3s3"]) ** (1.0 / 3.0)
    fog = params.get("fog")
    if fog and fog["regime"] == "steam_fog":
        plume = max(plume, 1.0 + 0.3 * fog["surface_excess_k"])
    return math.sqrt(U ** 2 + plume ** 2) + 1.0


def choose_dt(params: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[float, int]:
    t = cfg.get("time", {})
    dt = float(t.get("dt_s", 0.1))
    dx = float(cfg.get("grid", {}).get("domain_width_m", 320.0)) / int(cfg.get("grid", {}).get("nx", 64))
    target = float(cfg.get("lbm", {}).get("target_lattice_speed", 0.08))
    speed = expected_peak_speed(params, cfg)
    halvings = 0
    while speed * dt / dx > target and halvings < int(t.get("max_dt_halvings", 3)):
        dt *= 0.5
        halvings += 1
    return dt, halvings


def build_spec(params: Dict[str, Any], cfg: Dict[str, Any], dt: Optional[float] = None,
               halvings: int = 0) -> SimulationSpec:
    """Turn sampled parameters into a runnable SimulationSpec."""
    if dt is None:
        dt, halvings = choose_dt(params, cfg)
    phys = cfg.get("physics", {})
    lbm_cfg = cfg.get("lbm", {})
    rho_air = float(phys.get("rho_air", 1.2))
    theta0 = float(phys.get("theta0_k", 288.0))
    p_surface = float(phys.get("p0_pa", 101325.0))
    grid = build_grid(params, cfg)
    nz, nx = grid.nz, grid.nx
    p = thermo.pressure_profile(grid.z, p_surface, theta0)
    pi = thermo.exner(p)
    fog = params.get("fog")
    theta_ref, qv_ref, ql_ref = fog_reference(fog, grid, params["atmosphere"], p, pi, rho_air)
    wind = wind_function(params, grid)

    components: List[Component] = []
    first = float(params["timing"]["spinup_s"])
    if fog is not None:
        window = (int(params["timing"]["n_frames"]) - 1) * float(params["timing"]["frame_interval_s"])
        components.append(fog_component(fog, grid, (theta_ref, qv_ref, ql_ref), p, pi, rho_air, wind, first, window))
    if params.get("smoke") is not None:
        components.append(smoke_component(params["smoke"], grid, first, theta0,
                                          float(phys.get("heat_cap_k", 30.0)), wind))
    distractor = params.get("distractor")
    if distractor is not None:
        if distractor["regime"] == "haze_drift":
            components.append(haze_component(distractor["haze"], grid))
        elif distractor["regime"] == "dust_burst":
            components.append(dust_component(distractor, grid, first))
    if params.get("background_haze") is not None:
        components.append(haze_component(params["background_haze"], grid))

    initial = {
        "theta": np.repeat(theta_ref[:, None], nx, axis=1),
        "qv": np.repeat(qv_ref[:, None], nx, axis=1),
        "ql": np.repeat(ql_ref[:, None], nx, axis=1),
        "smoke": np.zeros((nz, nx)),
        "tracer": np.zeros((nz, nx)),
    }
    for comp in components:
        for fn in comp.initial:
            fn(initial)
    for name in initial:
        initial[name] = np.where(grid.fluid, initial[name], 0.0 if name != "theta" else theta_ref[:, None])

    zeros_z, zeros_x = np.zeros(nz), np.zeros(nx)
    base_ghosts = {
        "theta": {"left": theta_ref, "right": theta_ref, "top": np.full(nx, theta_ref[-1])},
        "qv": {"left": qv_ref, "right": qv_ref, "top": np.full(nx, qv_ref[-1])},
        "ql": {"left": ql_ref, "right": ql_ref, "top": zeros_x},
        "smoke": {"left": zeros_z, "right": zeros_z, "top": zeros_x},
        "tracer": {"left": zeros_z, "right": zeros_z, "top": zeros_x},
    }
    inflow_fns = [fn for comp in components for fn in comp.inflow]
    source_fns = [fn for comp in components for fn in comp.sources]

    def inflow(t: float):
        ghosts = {name: dict(sides) for name, sides in base_ghosts.items()}
        for fn in inflow_fns:
            fn(t, ghosts)
        return ghosts

    def sources(t: float, fields):
        rates: Dict[str, Any] = {}
        for fn in source_fns:
            fn(t, fields, rates)
        return rates

    settling = max([c.settling for c in components] + [0.0])
    timing = params["timing"]
    frame_times = np.arange(int(timing["n_frames"])) * float(timing["frame_interval_s"])
    info = {}
    for comp in components:
        info.update(comp.info)
    mixing = params["mixing"]
    nu = float(mixing["nu_m2s"])
    cs = float(mixing.get("smagorinsky_cs", lbm_cfg.get("smagorinsky_cs", 0.12)))
    tcfg = cfg.get("time", {})
    return SimulationSpec(
        sim_id=params["sim_id"], category=params["category"], regime=params["regime"],
        labels=tuple(params["labels"]), seed=int(params["seed"]), params=params,
        nx=nx, nz=nz, dx=grid.dx, dt=float(dt),
        nu=nu, diffusivity=float(mixing["diffusivity_m2s"]), smagorinsky_cs=cs,
        sponge_width=int(lbm_cfg.get("sponge_width_cells", 6)),
        sponge_rate=float(lbm_cfg.get("sponge_rate_s", 0.8)),
        solid=grid.solid, theta_ref=theta_ref, qv_ref=qv_ref, ql_ref=ql_ref,
        initial=initial, wind=wind, inflow=inflow, sources=sources,
        tracer_settling=float(settling), spinup_s=first, frame_times=frame_times,
        flow_lag_s=float(tcfg.get("flow_lag_s", 1.0)),
        precursor_s=float(tcfg.get("precursor_s", 30.0)),
        tex_period_s=float(phys.get("texture_period_s", 30.0)),
        top_sponge_theta_rate=float(lbm_cfg.get("top_sponge_theta_rate_s", 0.5)),
        theta0=theta0, p_surface=p_surface,
        max_lattice_speed=float(lbm_cfg.get("max_lattice_speed", 0.14)),
        max_view_density_deviation=float(lbm_cfg.get("max_view_density_deviation", 0.05)),
        view_margin=grid.view_margin,
        dt_halvings=int(halvings), extra={"components": info},
    )


def run_scenario(params: Dict[str, Any], cfg: Dict[str, Any]) -> SimulationResult:
    """Run with automatic dt reduction; raises SimulationUnstable when every dt fails."""
    dt, halvings = choose_dt(params, cfg)
    max_h = int(cfg.get("time", {}).get("max_dt_halvings", 3))
    errors = []
    while True:
        spec = build_spec(params, cfg, dt=dt, halvings=halvings)
        try:
            return run_simulation(spec, cfg)
        except SimulationUnstable as exc:
            errors.append(str(exc))
            if halvings >= max_h:
                raise SimulationUnstable("; ".join(errors)) from exc
            dt *= 0.5
            halvings += 1
