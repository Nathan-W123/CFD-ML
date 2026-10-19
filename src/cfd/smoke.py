"""Smoke regimes: emission sources, heating, schedules and advected smoke layers.

Smoke is a transported aerosol (mg/m^3) emitted by a ground or near-ground
source that may also heat the air (the buoyancy comes out of the flow solver,
it is not prescribed). A source is specified by its integrated buoyancy flux
per metre F (m^3/s^3), spread over at least 2 x 2 cells starting one cell
above the ground (the no-slip cell itself barely moves); the heating rate
fades out as the local temperature anomaly approaches `physics.heat_cap_k`,
which keeps the plume within the Boussinesq range. Five regimes:

* hot_plume       large F, light wind: a rising plume
* smoldering      small F: the plume drifts and hugs the ground
* crosswind       moderate F in strong wind: a bent-over plume
* pulsing         emission and heat switching on and off (intermittent burning)
* advected_smoke  no source in the scene: a cooled remnant of an upwind plume
                  carried in through the upwind boundary as an elevated, puffy
                  layer with a small residual warmth (unlike passive haze,
                  which is broad, smooth and neutral — an explicit modelling
                  assumption that keeps the two from being identical)

Every regime randomises source position, width, height, emission rate,
heating, onset time relative to the first frame (most plumes are already
established, some ignite during the window, some switch off), intermittency
and an optional residual smoke puff. Nothing forces smoke to rise.

The same machinery provides the "dust_burst" distractor (an unheated,
settling ground emission, possibly from a moving vehicle), which lives on the
tracer field.

Where this fits: `scenarios.sample_scenario` calls `sample_smoke` /
`sample_dust`; `scenarios.build_spec` calls `smoke_component` /
`dust_component` and merges the returned callables into the SimulationSpec.
Calls into: numpy only (plus the shared helpers in scenarios_util).
"""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from src.cfd import thermo
from src.cfd.scenario_util import (
    Component, Grid, activity_window, drift_start, layer_profile, log_uniform, smooth_noise01,
    smooth_texture, sample_noise_spec, source_mask, uniform,
)

SMOKE_DEFAULTS = {
    "hot_plume": {"buoyancy_flux_m3s3": [5.0, 70.0], "wind_speed_ms": [0.0, 5.0]},
    "smoldering": {"buoyancy_flux_m3s3": [0.02, 0.5], "wind_speed_ms": [0.0, 7.0]},
    "crosswind": {"buoyancy_flux_m3s3": [2.0, 40.0], "wind_speed_ms": [3.5, 9.0]},
    "pulsing": {"buoyancy_flux_m3s3": [1.0, 40.0], "pulse_period_s": [8.0, 40.0], "duty_cycle": [0.2, 0.7],
                "wind_speed_ms": [0.0, 7.0]},
    "advected_smoke": {"layer_base_m": [0.0, 120.0], "layer_depth_m": [20.0, 150.0],
                       "inflow_mg_m3": [0.5, 15.0], "intermittency": [0.3, 0.8],
                       "residual_warmth_k": [0.1, 0.5], "wind_speed_ms": [1.0, 9.0]},
}


def smoke_wind_range(regime: str, cfg: Dict[str, Any]):
    block = cfg.get("scenarios", {}).get("smoke", {}).get(regime, SMOKE_DEFAULTS[regime])
    return tuple(block.get("wind_speed_ms", SMOKE_DEFAULTS[regime]["wind_speed_ms"]))


def sample_smoke(regime: str, rng: np.random.Generator, cfg: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Sample the smoke component of a scenario (JSON-able dict)."""
    sc = cfg.get("scenarios", {}).get("smoke", {})
    block = dict(SMOKE_DEFAULTS[regime])
    block.update(sc.get(regime, {}))
    out: Dict[str, Any] = {"regime": regime}
    first = float(context["timing"]["spinup_s"])
    if regime == "advected_smoke":
        out.update(
            layer_base_m=uniform(rng, block["layer_base_m"]),
            layer_depth_m=uniform(rng, block["layer_depth_m"]),
            inflow_mg_m3=uniform(rng, block["inflow_mg_m3"]),
            intermittency=uniform(rng, block["intermittency"]),
            residual_warmth_k=uniform(rng, block["residual_warmth_k"]),
            front_frac=uniform(rng, (0.25, 0.85)),   # how far into the view the layer reaches at t=0
            modulation=sample_noise_spec(rng, (5.0, 25.0)),
            texture_seed=int(rng.integers(0, 2**31 - 1)),
            buoyancy_flux_m3s3=0.0,
        )
        return out
    elevated = "elevated_source" in context.get("holdout_tags", [])
    width = uniform(rng, sc.get("source_width_m", [10.0, 25.0]))
    out.update(
        buoyancy_flux_m3s3=log_uniform(rng, block["buoyancy_flux_m3s3"]),
        emission_mg_m3_s=uniform(rng, sc.get("emission_mg_m3_s", [5.0, 100.0])),
        source_x_frac=uniform(rng, sc.get("source_x_frac", [0.1, 0.9])),
        source_width_m=width,
        source_height_m=0.0 if elevated else uniform(rng, sc.get("source_height_m", [0.0, 10.0])),
        on_hilltop=bool(elevated),
        intermittency=uniform(rng, (0.0, 0.5)),
        # Puffing of a flaming source, ~1.5 / sqrt(D) Hz (D = source width); the
        # 1 s optical-flow lag sees it, the 5 s observation spacing aliases it.
        flicker=sample_noise_spec(rng, (0.6 * math.sqrt(width) / 1.5, 2.0 * math.sqrt(width) / 1.5)),
    )
    # Onset relative to the first frame, inside the (class-independent) spin-up: mostly
    # established plumes, some ignitions in view. The spin-up is never lengthened for
    # smoke; doing so made spin-up predict the smoke label (folded AUC 0.74 in the audit).
    spin = float(context["timing"]["spinup_s"])
    if rng.random() < 0.75 and spin > 25.0:
        onset = uniform(rng, (-(spin - 5.0), -20.0))
    else:
        onset = uniform(rng, (-min(20.0, spin - 5.0), 25.0))
    out["onset_s"] = onset
    out["offset_s"] = uniform(rng, (15.0, 50.0)) if rng.random() < 0.15 else None
    if regime == "pulsing":
        out.update(pulse_period_s=uniform(rng, block["pulse_period_s"]),
                   duty_cycle=uniform(rng, block["duty_cycle"]),
                   pulse_phase=uniform(rng, (0.0, 1.0)))
    out["residual"] = None
    if rng.random() < 0.35:
        out["residual"] = {
            "mg_m3": uniform(rng, (0.3, 3.0)),
            "height_m": uniform(rng, (40.0, 160.0)),
            "radius_m": uniform(rng, (25.0, 60.0)),
            "dx_m": uniform(rng, (-60.0, 60.0)),
        }
    out["first_frame_s"] = first
    return out


def heating_rate(buoyancy_flux: float, n_cells: int, dx: float, theta0: float = 288.0) -> float:
    """Volumetric heating (K/s) that gives integrated buoyancy flux F over n_cells cells of dx^2."""
    if n_cells <= 0:
        return 0.0
    return buoyancy_flux * theta0 / (thermo.G * n_cells * dx * dx)


def smoke_component(smoke: Dict[str, Any], grid: Grid, first_frame_s: float,
                    theta0: float = 288.0, heat_cap_k: float = 30.0, wind_fn=None) -> Component:
    """Initial fields, inflow and sources for the smoke component."""
    comp = Component()
    if smoke["regime"] == "advected_smoke":
        _advected(smoke, grid, comp, first_frame_s, wind_fn)
        return comp
    xc = grid.view_x(smoke["source_x_frac"])
    if smoke.get("on_hilltop"):
        xc = grid.hilltop_x()
    mask = source_mask(grid, xc, smoke["source_width_m"], smoke["source_height_m"], lift_cells=1, rows=2)
    t_on = first_frame_s + float(smoke["onset_s"])
    t_off = None if smoke.get("offset_s") is None else first_frame_s + float(smoke["offset_s"])
    emission = float(smoke["emission_mg_m3_s"])
    heating = heating_rate(float(smoke["buoyancy_flux_m3s3"]), int(mask.sum()), grid.dx, theta0)
    inter = float(smoke["intermittency"])
    flicker = smoke["flicker"]
    pulsing = smoke["regime"] == "pulsing"
    period = float(smoke.get("pulse_period_s", 1.0))
    duty = float(smoke.get("duty_cycle", 1.0))
    phase = float(smoke.get("pulse_phase", 0.0))

    def activity(t: float) -> float:
        a = activity_window(t, t_on, t_off, ramp_s=3.0)
        if a <= 0.0:
            return 0.0
        if pulsing:
            # Smoothed square wave: on for `duty` of each period.
            s = ((t / period + phase) % 1.0)
            edge = min(0.15, 0.5 * duty, 0.5 * (1.0 - duty))
            if s < duty:
                a *= min(1.0, s / edge, (duty - s) / edge) if edge > 0 else 1.0
            else:
                a = 0.0
        return a * (1.0 - inter + inter * smooth_noise01(t, flicker))

    def sources(t, fields, rates):
        a = activity(t)
        if a <= 0.0:
            return
        rates["smoke"] = rates.get("smoke", 0.0) + a * emission * mask
        if heating > 0.0:
            fade = np.clip(1.0 - fields["theta_anom"] / heat_cap_k, 0.0, 1.0)
            rates["theta"] = rates.get("theta", 0.0) + a * heating * mask * fade

    comp.sources.append(sources)
    res = smoke.get("residual")
    if res:
        xx, zz = grid.xx, grid.zz
        blob = res["mg_m3"] * np.exp(-((xx - (xc + res["dx_m"])) ** 2 + (zz - res["height_m"]) ** 2)
                                     / (2.0 * res["radius_m"] ** 2))

        def initial(fields):
            fields["smoke"] += np.where(grid.fluid, blob, 0.0)

        comp.initial.append(initial)
    comp.info["source_x_m"] = float(xc)
    comp.info["source_cells"] = int(mask.astype(bool).sum())
    comp.info["source_heating_k_s"] = float(heating)
    return comp


def _advected(smoke: Dict[str, Any], grid: Grid, comp: Component, first_frame_s: float = 0.0,
              wind_fn=None) -> None:
    base, depth = float(smoke["layer_base_m"]), float(smoke["layer_depth_m"])
    c_in = float(smoke["inflow_mg_m3"])
    inter = float(smoke["intermittency"])
    mod = smoke["modulation"]
    profile = layer_profile(grid.z, base, base + depth, soft_m=12.0)
    tex = smooth_texture(grid, smoke["texture_seed"], scale_m=40.0, contrast=0.5)
    upwind_left = grid.wind_sign >= 0
    # Layer present from the upwind boundary to `front_frac` of the view at the first frame.
    target = grid.view_x(smoke["front_frac"] if upwind_left else 1.0 - smoke["front_frac"])
    u_drift = 0.0
    if wind_fn is not None:
        k_mid = int(np.clip((base + 0.5 * depth) / grid.dx, 0, grid.nz - 1))
        u_drift = 0.85 * float(wind_fn(0.0)[k_mid])
    front_x = drift_start(target, u_drift, first_frame_s)
    xx = grid.xx
    reach = (xx <= front_x) if upwind_left else (xx >= front_x)
    edge = np.clip(1.0 - np.abs(xx - front_x) / 30.0, 0.0, 1.0)
    init = c_in * profile[:, None] * tex * np.where(reach, 1.0, edge)

    def initial(fields):
        fields["smoke"] += np.where(grid.fluid, init, 0.0)

    warmth = float(smoke.get("residual_warmth_k", 0.0))

    def initial_warmth(fields):
        fields["theta"] += np.where(grid.fluid, warmth * profile[:, None] * np.where(reach, 1.0, edge), 0.0)

    x_b = 0.0 if upwind_left else grid.nx * grid.dx
    s_dir = 1.0 if upwind_left else -1.0

    def inflow(t, ghosts):
        # The layer enters only once its front (front_x at t = 0, drifting at
        # u_drift) has reached the upwind boundary.
        w_b = 0.5 * (1.0 + math.tanh(s_dir * (front_x + u_drift * t - x_b) / 15.0))
        if w_b < 1e-6:
            return
        puff = smooth_noise01(t, mod)
        level = c_in * (1.0 - inter + inter * puff)
        side = "left" if upwind_left else "right"
        ghosts.setdefault("smoke", {})[side] = w_b * level * profile
        th = ghosts["theta"][side]
        ghosts["theta"][side] = th + w_b * warmth * (0.5 + puff) * profile

    comp.initial.append(initial)
    comp.initial.append(initial_warmth)
    comp.inflow.append(inflow)


# ------------------------------------------------------------------ dust (neither)
def sample_dust(rng: np.random.Generator, cfg: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    block = cfg.get("scenarios", {}).get("neither", {}).get("dust_burst", {})
    moving = rng.random() < 0.4
    return {
        "regime": "dust_burst",
        "emission_mg_m3_s": uniform(rng, block.get("emission_mg_m3_s", [10.0, 120.0])),
        "duration_s": uniform(rng, block.get("duration_s", [2.0, 12.0])),
        "settling_ms": uniform(rng, block.get("settling_ms", [0.1, 0.6])),
        "onset_s": uniform(rng, (-8.0, 30.0)),          # relative to the first frame
        "source_x_frac": uniform(rng, (0.1, 0.9)),
        "source_width_m": uniform(rng, (5.0, 25.0)),
        "vehicle_speed_ms": uniform(rng, (2.0, 10.0)) * (1 if rng.random() < 0.5 else -1) if moving else 0.0,
        "flicker": sample_noise_spec(rng, (1.0, 6.0)),
    }


def dust_component(dust: Dict[str, Any], grid: Grid, first_frame_s: float) -> Component:
    comp = Component()
    t_on = first_frame_s + float(dust["onset_s"])
    t_off = t_on + float(dust["duration_s"])
    x0 = grid.view_x(dust["source_x_frac"])
    speed = float(dust["vehicle_speed_ms"])
    width = float(dust["source_width_m"])
    emission = float(dust["emission_mg_m3_s"])
    flicker = dust["flicker"]
    lo, hi = grid.view_x(0.0), grid.view_x(1.0)

    def sources(t, fields, rates):
        a = activity_window(t, t_on, t_off, ramp_s=1.0)
        if a <= 0.0:
            return
        x = x0 + speed * (t - t_on)
        if x < lo - width or x > hi + width:
            return
        mask = source_mask(grid, x, width, 0.0, lift_cells=0, rows=1)
        a *= 0.6 + 0.4 * smooth_noise01(t, flicker)
        rates["tracer"] = rates.get("tracer", 0.0) + a * emission * mask

    comp.sources.append(sources)
    comp.settling = float(dust["settling_ms"])
    return comp
