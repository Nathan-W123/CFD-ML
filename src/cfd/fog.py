"""Fog regimes: moist reference profiles, fog layers, banks, patches and steam fog.

Fog is liquid water (q_l) in saturated air, produced and destroyed by the
saturation adjustment in thermo.py. It is never smoke relabelled: the
scenarios below set up humidity and temperature so that condensation,
evaporation and transport of liquid water produce the fog the camera sees.

* radiation_fog      an established near-ground fog layer under an inversion,
                     light wind, slow surface cooling
* advected_fog_bank  colder, saturated air carrying fog in through the upwind
                     boundary into clearer air (the front evaporates by mixing)
* terrain_fog        valley fog over uneven terrain with a cold layer on the
                     slopes that drains downhill (cold-air drainage)
* fog_patch          a localised saturated patch in sub-saturated air,
                     drifting and evaporating at its edges (optionally fed by
                     moist inflow, so it can also grow)
* fog_dissipating    an established layer eroded by dry inflow, surface
                     warming (sunrise) or a strengthening wind
* steam_fog          cold air over a warm, wet surface strip: rising wisps of
                     condensate anchored at a source (a deliberate hard
                     negative for "rises from a fixed source")
* ridge_cloud        orographic cap / low cloud: humid, near-neutral to weakly
                     stable air carried over a ridge or hill condenses where it
                     is lifted (no liquid is prescribed; the saturation
                     adjustment makes it). The ambient humidity is set so that
                     air lifted by `lcl_frac` of the ridge height saturates;
                     the cloud base sits on the windward slope and the cloud
                     stays anchored over the crest while the air moves through
                     it (a stationary cloud in moving air); it evaporates where
                     the air descends in the lee near the ground and, the
                     stratification being close to moist-neutral, trails
                     downstream aloft as a cloud deck. Needs a hill or ridge
                     and wind.

Fog starts established (the 60 s window is far too short to form a fog bank
from clear air) and evolves through transport, mixing, cooling and warming.
(The ridge cloud is the exception: it forms during the spin-up, within one
advective time over the ridge, from the class-independent wind.)
The air around established fog is near saturation (RH 0.92-0.99; drier air
only where a regime calls for it, e.g. dry inflow eroding a fog), and fog air
starts close to neutrally buoyant (saturated at about the ambient
temperature), so banks neither slump nor lift as an artefact of the set-up.

Scales: physical ranges (layer tops, inversions, patch sizes, depths,
periods) come from the profile config, regime blocks overriding the shared
`scenarios.fog` keys; numerical widths (front ramps, layer softness, minimum
thicknesses, texture) scale with dx; the dissipation onset scales with the
frame interval (see scenario_util).

Where this fits: `scenarios.sample_scenario` calls `sample_fog`;
`scenarios.build_spec` calls `fog_reference` (moist ambient profiles) and
`fog_component` (initial fields, inflow, sources).
Calls into: thermo, scenario_util.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

from src.cfd import thermo
from src.cfd.scenario_util import (
    Component, Grid, block_range, drift_start, layer_profile, sample_noise_spec, smooth_noise,
    smooth_profile_noise, smooth_texture, time_scale, uniform,
)

DRIFT_FACTOR = 0.85  # drifting features move a little slower than the free-stream wind at their height

FOG_WIND = {
    "radiation_fog": (0.0, 3.5),
    "advected_fog_bank": (1.0, 9.0),
    "terrain_fog": (0.0, 4.0),
    "fog_patch": (0.0, 7.0),
    "fog_dissipating": (0.0, 9.0),
    "steam_fog": (0.0, 5.0),
    "ridge_cloud": (3.0, 15.0),
}
AMBIENT_LAYER_REGIMES = ("radiation_fog", "terrain_fog", "fog_dissipating")
RIDGE_TERRAIN = ("hill", "ridge")
RIDGE_DEFAULTS = {
    "min_terrain_height_m": 15.0,      # a resolvable obstacle (near-field default: 3 cells)
    "lcl_frac": [0.3, 0.7],            # lift (fraction of the ridge height) that saturates the inflow
    "moist_top_frac": [1.5, 3.0],      # humid layer depth in ridge heights
    "rh_above": [0.3, 0.7],            # drier air above the humid layer
    "max_lapse_dtheta_dz": 0.004,      # K/m: near-neutral / weakly stable, so the flow goes over (Fr > ~1)
}


def ridge_cloud_possible(context: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    """Terrain requirement of ridge_cloud: a hill or ridge at least min_terrain_height_m tall."""
    block = cfg.get("scenarios", {}).get("fog", {}).get("ridge_cloud", {})
    terrain = context["terrain"]
    return (terrain["kind"] in RIDGE_TERRAIN
            and float(terrain.get("height_m", 0.0)) >= float(block.get("min_terrain_height_m",
                                                                       RIDGE_DEFAULTS["min_terrain_height_m"])))


def fog_wind_range(regime: str, cfg: Dict[str, Any]):
    block = cfg.get("scenarios", {}).get("fog", {}).get(regime, {})
    return tuple(block.get("wind_speed_ms", FOG_WIND[regime]))


def steam_supersaturation(theta_ground: float, excess: float, rh: float, p: float = 101325.0) -> float:
    """Largest supersaturation (kg/kg) along the mixing line between surface air saturated at
    T + excess and ambient air at relative humidity rh: > 0 means mixing produces fog."""
    T_a = theta_ground * float(thermo.exner(p))
    T_s = T_a + excess
    q_a = rh * float(thermo.saturation_mixing_ratio(T_a, p))
    q_s = float(thermo.saturation_mixing_ratio(T_s, p))
    f = np.linspace(0.0, 1.0, 101)
    T_m = f * T_s + (1 - f) * T_a
    q_m = f * q_s + (1 - f) * q_a
    return float((q_m - thermo.saturation_mixing_ratio(T_m, p)).max())


def sample_fog(regime: str, rng: np.random.Generator, cfg: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    fc = cfg.get("scenarios", {}).get("fog", {})
    block = fc.get(regime, {})
    ts = time_scale(context.get("timing", {}).get("frame_interval_s", 5.0))
    # A regime block may override the shared liquid-water and layer-top ranges.
    lwc = uniform(rng, block_range(block, "lwc_g_m3", fc.get("lwc_g_m3", [0.05, 0.4])))
    top = uniform(rng, block_range(block, "fog_top_m", fc.get("fog_top_m", [40.0, 160.0])))
    rh_outside = fc.get("rh_outside", [0.92, 0.99])
    out: Dict[str, Any] = {
        "regime": regime,
        "lwc_g_m3": lwc,
        "fog_top_m": top,
        "heterogeneity": uniform(rng, (0.1, 0.5)),
        "top_undulation": uniform(rng, (0.0, 0.25)),
        "texture_seed": int(rng.integers(0, 2**31 - 1)),
        "top_seed": int(rng.integers(0, 2**31 - 1)),
        "rh_above": uniform(rng, rh_outside if regime != "fog_dissipating" else (0.8, 0.97)),
        # Height of the capping inversion above the fog top (physical, from the profile).
        "inversion_offset_m": float(fc.get("inversion_offset_m", 20.0)),
    }
    if regime == "radiation_fog":
        out.update(inversion_dtheta_dz=uniform(rng, block.get("inversion_dtheta_dz", [0.01, 0.06])),
                   surface_cooling_k_s=uniform(rng, block.get("surface_cooling_k_s", [0.0002, 0.002])))
    elif regime == "advected_fog_bank":
        out.update(front_frac=uniform(rng, block.get("front_x_frac", [0.1, 0.6])),
                   bank_dtheta_k=uniform(rng, (-0.5, 0.3)),
                   domain_rh=uniform(rng, rh_outside),
                   modulation=sample_noise_spec(rng, block.get("modulation_period_s", [10.0, 40.0])),
                   inversion_dtheta_dz=uniform(rng, block.get("inversion_dtheta_dz", [0.0, 0.03])))
    elif regime == "terrain_fog":
        out.update(cold_layer_k=uniform(rng, block.get("cold_layer_k", [0.5, 3.0])),
                   cold_layer_depth_m=uniform(rng, block.get("cold_layer_depth_m", [8.0, 20.0])),
                   fill_frac=uniform(rng, block.get("fill_frac", [0.5, 1.1])),
                   inversion_dtheta_dz=uniform(rng, block.get("inversion_dtheta_dz", [0.01, 0.05])),
                   surface_cooling_k_s=uniform(rng, block.get("surface_cooling_k_s", [0.0003, 0.002])))
    elif regime == "fog_patch":
        out.update(center_frac=uniform(rng, (0.15, 0.85)),
                   center_height_m=uniform(rng, block.get("center_height_m", [0.0, 90.0])),
                   radius_m=uniform(rng, block.get("patch_radius_m", [20.0, 70.0])),
                   aspect=uniform(rng, (0.4, 1.0)),
                   ambient_rh=uniform(rng, block.get("ambient_rh", rh_outside)),
                   patch_dtheta_k=uniform(rng, (-0.3, 0.1)),
                   moist_inflow=bool(rng.random() < 0.3),
                   inflow_rh=uniform(rng, (0.98, 1.01)),
                   inflow_depth_m=uniform(rng, block.get("inflow_depth_m", [20.0, 100.0])),
                   inversion_dtheta_dz=uniform(rng, block.get("inversion_dtheta_dz", [0.0, 0.03])))
    elif regime == "fog_dissipating":
        # "wind_ramp" only where the class-independent context already accelerates the wind.
        mechs = ["dry_inflow", "surface_warming"]
        if float(context.get("wind", {}).get("ramp_factor", 1.0)) > 1.0:
            mechs.append("wind_ramp")
        mech = mechs[int(rng.integers(0, len(mechs)))]
        out.update(mechanism=mech,
                   # relative to the first frame (near-field seconds x the profile's time scale)
                   mechanism_onset_s=uniform(rng, (-30.0 * ts, 10.0 * ts)),
                   inversion_dtheta_dz=uniform(rng, block.get("inversion_dtheta_dz", [0.005, 0.04])),
                   dry_inflow_rh=uniform(rng, block.get("dry_inflow_rh", [0.4, 0.85])),
                   surface_warming_k_s=uniform(rng, block.get("surface_warming_k_s", [0.002, 0.02])))
    elif regime == "steam_fog":
        # Re-draw until mixing surface air with ambient air actually supersaturates.
        for _ in range(50):
            theta_g = uniform(rng, block.get("theta_ground_k", [263.0, 280.0]))
            excess = uniform(rng, block.get("surface_excess_k", [8.0, 15.0]))
            rh = uniform(rng, block.get("ambient_rh", [0.85, 0.98]))
            ss = steam_supersaturation(theta_g, excess, rh)
            if ss >= 5e-5:
                break
        out.update(surface_excess_k=excess,
                   strip_center_frac=uniform(rng, (0.2, 0.8)),
                   strip_width_m=uniform(rng, block.get("strip_width_m", [20.0, 120.0])),
                   exchange_time_s=uniform(rng, block.get("exchange_time_s", [2.0, 6.0])),
                   theta_ground_k=theta_g,
                   ambient_rh=rh,
                   mixing_supersaturation=ss)
    elif regime == "ridge_cloud":
        rb = dict(RIDGE_DEFAULTS)
        rb.update(block)
        out.update(lcl_frac=uniform(rng, rb["lcl_frac"]),
                   moist_top_frac=uniform(rng, rb["moist_top_frac"]),
                   rh_above=uniform(rng, rb["rh_above"]),
                   max_lapse_dtheta_dz=float(rb["max_lapse_dtheta_dz"]))
    else:
        raise ValueError(f"unknown fog regime {regime!r}")
    return out


def _ql_from_lwc(lwc_g_m3: float, rho_air: float) -> float:
    return lwc_g_m3 / (rho_air * 1000.0)


def _layer_lwc_shape(z: np.ndarray, top: float, soft_min: float = 4.0) -> np.ndarray:
    """Vertical LWC shape of a fog layer: grows with height, peaks near 0.8 top, soft top
    (softness 8 % of the top, at least soft_min; callers pass 0.8 dx)."""
    top = np.maximum(np.asarray(top, dtype=float), 1.0)
    s = np.clip(z / top, 0.0, None)
    return (0.35 + 0.65 * np.clip(s / 0.8, 0.0, 1.0)) * 0.5 * (1.0 - np.tanh((z - top) / np.maximum(soft_min, 0.08 * top)))


def _lifted_pressure(z: np.ndarray, p: np.ndarray, dz: float) -> np.ndarray:
    """Reference pressure at z + dz (log-linear in z, extrapolated above the top)."""
    lnp = np.log(p)
    zl = z + dz
    out = np.interp(zl, z, lnp)
    slope = (lnp[-1] - lnp[-2]) / (z[-1] - z[-2]) if len(z) > 1 else -1.2e-4
    above = zl > z[-1]
    out[above] = lnp[-1] + (zl[above] - z[-1]) * slope
    return np.exp(out)


def ridge_cloud_reference(fog: Dict[str, Any], grid: Grid, atm: Dict[str, Any], p: np.ndarray,
                          pi: np.ndarray):
    """(theta, q_v) profiles for ridge_cloud: air at every level of the humid layer saturates
    when lifted dry-adiabatically by lcl_frac x the ridge height (theta conserved, so the
    lifted temperature is theta(z) Exner(p(z + dz))); above it the air is drier. Every
    level stays sub-saturated at rest (at most 99.5 % RH), so all condensate comes from lifting."""
    z = grid.z
    lapse = min(max(float(atm["lapse_dtheta_dz"]), 0.0), float(fog["max_lapse_dtheta_dz"]))
    theta = atm["theta_ground_k"] + lapse * z
    H = max(float(grid.ground_height.max()), grid.dx)
    dz_c = float(fog["lcl_frac"]) * H
    T = theta * pi
    qs = thermo.saturation_mixing_ratio(T, p)
    p_lift = _lifted_pressure(z, p, dz_c)
    T_lift = theta * thermo.exner(p_lift)
    q_moist = np.minimum(thermo.saturation_mixing_ratio(T_lift, p_lift), 0.995 * qs)
    q_dry = float(fog["rh_above"]) * qs
    top = float(fog["moist_top_frac"]) * H
    inside = layer_profile(z, 0.0, top, soft_m=2.0 * grid.dx)
    qv = np.minimum(inside * q_moist + (1.0 - inside) * q_dry, 0.995 * qs)
    fog["lift_to_saturation_m"] = dz_c
    fog["moist_top_m"] = top
    return theta, qv


def fog_reference(fog: Optional[Dict[str, Any]], grid: Grid, atm: Dict[str, Any], p: np.ndarray,
                  pi: np.ndarray, rho_air: float):
    """Ambient (reference) profiles theta_ref, qv_ref, ql_ref (each (nz,)) for the scenario.

    Without fog the atmosphere block alone sets them; fog regimes add an
    inversion and, for ambient fog layers, a saturated layer holding liquid.
    """
    z = grid.z
    dx = grid.dx
    theta = atm["theta_ground_k"] + atm["lapse_dtheta_dz"] * z
    rh = np.full_like(z, atm["relative_humidity"])
    ql = np.zeros_like(z)
    if fog is not None:
        regime = fog["regime"]
        if regime == "ridge_cloud":
            theta, qv = ridge_cloud_reference(fog, grid, atm, p, pi)
            return theta, qv, ql
        if regime == "steam_fog":
            theta = fog["theta_ground_k"] + max(atm["lapse_dtheta_dz"], 0.0) * z
            rh[:] = fog["ambient_rh"]
        else:
            top = fog["fog_top_m"]
            if regime == "terrain_fog":
                # Valley fog fills the valley and usually extends above it (the terrain is
                # only 5-40 m deep, shared with every category); the cold pool on the slopes
                # (fog_component) drives the drainage that distinguishes it.
                top = max(fog["fog_top_m"], fog["fill_frac"] * float(grid.ground_height.max()))
                fog["effective_top_m"] = top
            inv = fog.get("inversion_dtheta_dz", 0.0)
            h_inv = top + float(fog.get("inversion_offset_m", 20.0))
            theta = atm["theta_ground_k"] + inv * np.minimum(z, h_inv) + max(atm["lapse_dtheta_dz"], 0.0) * np.maximum(z - h_inv, 0.0)
            above = fog["rh_above"]
            if regime in AMBIENT_LAYER_REGIMES:
                inside = layer_profile(z, 0.0, top, soft_m=max(0.6 * dx, 0.05 * top))
                rh = above + (1.0 - above) * inside
                ql = _ql_from_lwc(fog["lwc_g_m3"], rho_air) * _layer_lwc_shape(z, top, 0.8 * dx)
            elif regime == "advected_fog_bank":
                inside = layer_profile(z, 0.0, top, soft_m=2.0 * dx)
                rh = above + (fog["domain_rh"] - above) * inside
            elif regime == "fog_patch":
                rh[:] = fog["ambient_rh"]
    T = theta * pi
    qv = thermo.mixing_ratio_from_rh(np.minimum(rh, 1.0), T, p)
    return theta, qv, ql


def fog_component(fog: Dict[str, Any], grid: Grid, refs, p: np.ndarray, pi: np.ndarray,
                  rho_air: float, wind_fn, first_frame_s: float = 0.0, window_s: float = 55.0) -> Component:
    regime = fog["regime"]
    theta_ref, qv_ref, ql_ref = refs
    comp = Component()
    dx = grid.dx
    # Numerical widths in cells (texture 50 m, fronts 15 m, softness 8 m at dx = 5 m).
    tex = smooth_texture(grid, fog["texture_seed"], scale_m=10.0 * dx, contrast=fog["heterogeneity"])
    ql_max = _ql_from_lwc(fog["lwc_g_m3"], rho_air)
    surface = grid.surface_cells()

    if regime == "ridge_cloud":
        # Nothing prescribed: the humid reference inflow condenses where the ridge lifts it.
        comp.info["lift_to_saturation_m"] = float(fog.get("lift_to_saturation_m", 0.0))
        comp.info["moist_top_m"] = float(fog.get("moist_top_m", 0.0))
        return comp

    if regime in AMBIENT_LAYER_REGIMES:
        top = fog.get("effective_top_m", fog["fog_top_m"])
        undul = 1.0 + fog["top_undulation"] * smooth_profile_noise(grid.nx, fog["top_seed"], scale_cells=12.0)
        top_x = np.maximum(top * undul, dx)
        ql0 = ql_max * _layer_lwc_shape(grid.zz, top_x[None, :], 0.8 * dx) * tex

        def initial(fields):
            # Only the liquid changes (heterogeneity, undulating top): the reference layer is
            # already saturated, and the diagnosis re-equilibrates every cell at t = 0.
            fields["ql"][...] = np.where(grid.fluid, ql0, 0.0)

        comp.initial.append(initial)
        if regime in ("radiation_fog", "terrain_fog"):
            rate = -fog["surface_cooling_k_s"] * surface

            def cooling(t, fields, rates):
                rates["theta"] = rates.get("theta", 0.0) + rate

            comp.sources.append(cooling)
        if regime == "terrain_fog":
            cold = grid.near_surface(fog["cold_layer_depth_m"]) & (grid.ground_height[None, :] > 2.0 * grid.dx)
            depth = np.clip(1.0 - grid.height_above_ground / fog["cold_layer_depth_m"], 0.0, 1.0)

            def cold_init(fields):
                fields["theta"] -= np.where(cold, fog["cold_layer_k"] * depth, 0.0)

            comp.initial.append(cold_init)
        if regime == "fog_dissipating":
            mech = fog["mechanism"]
            t_mech = first_frame_s + float(fog.get("mechanism_onset_s", 0.0))
            if mech == "dry_inflow":
                T_ref = theta_ref * pi
                qv_dry = thermo.mixing_ratio_from_rh(fog["dry_inflow_rh"], T_ref, p)

                def dry(t, ghosts):
                    if t < t_mech:
                        return
                    side = "left" if grid.wind_sign >= 0 else "right"
                    ghosts.setdefault("qv", {})[side] = qv_dry
                    ghosts.setdefault("ql", {})[side] = np.zeros(grid.nz)

                comp.inflow.append(dry)
            elif mech == "surface_warming":
                rate = fog["surface_warming_k_s"] * surface

                def warming(t, fields, rates):
                    if t >= t_mech:
                        rates["theta"] = rates.get("theta", 0.0) + rate

                comp.sources.append(warming)
            # "wind_ramp": the context's accelerating wind (scenarios.wind_function) clears the fog.
        return comp

    if regime == "advected_fog_bank":
        top = fog["fog_top_m"]
        layer = layer_profile(grid.z, 0.0, top, soft_m=1.6 * dx)
        theta_bank = theta_ref + fog["bank_dtheta_k"] * layer
        qv_bank = np.where(layer > 0.05, thermo.saturation_mixing_ratio(theta_bank * pi, p), qv_ref)
        ql_bank = ql_max * _layer_lwc_shape(grid.z, top, 0.8 * dx)
        upwind_left = grid.wind_sign >= 0
        target = grid.view_x(fog["front_frac"] if upwind_left else 1.0 - fog["front_frac"])
        k_mid = int(np.clip(0.5 * top / grid.dx, 0, grid.nz - 1))
        u_drift = DRIFT_FACTOR * float(wind_fn(0.0)[k_mid])
        front_x = drift_start(target, u_drift, first_frame_s)
        dist = (front_x - grid.xx) if upwind_left else (grid.xx - front_x)
        inside = 0.5 * (1.0 + np.tanh(dist / (3.0 * dx)))

        def initial(fields):
            fields["theta"] += inside * (theta_bank - theta_ref)[:, None]
            fields["qv"][...] = fields["qv"] + inside * (qv_bank - qv_ref)[:, None]
            fields["ql"][...] = np.where(grid.fluid, inside * ql_bank[:, None] * tex, 0.0)

        mod = fog["modulation"]
        x_b = 0.0 if upwind_left else grid.nx * grid.dx
        s_dir = 1.0 if upwind_left else -1.0

        def inflow(t, ghosts):
            # The bank enters only once its front (front_x at t = 0, drifting at
            # u_drift) has reached the upwind boundary; until then the inflow is ambient.
            w_b = 0.5 * (1.0 + math.tanh(s_dir * (front_x + u_drift * t - x_b) / (3.0 * dx)))
            if w_b < 1e-6:
                return
            side = "left" if upwind_left else "right"
            level = 0.85 + 0.15 * np.tanh(smooth_noise(t, mod))
            for name, bank in (("theta", theta_bank), ("qv", qv_bank), ("ql", level * ql_bank)):
                sides = ghosts.setdefault(name, {})
                base = sides.get(side)
                base = np.zeros_like(bank) if base is None else base
                sides[side] = base + w_b * (bank - base)

        comp.initial.append(initial)
        comp.inflow.append(inflow)
        return comp

    if regime == "fog_patch":
        target = grid.view_x(fog["center_frac"])
        zc = float(grid.ground_height[int(np.clip(target / grid.dx, 0, grid.nx - 1))]) + fog["center_height_m"]
        a = fog["radius_m"]
        b = a * fog["aspect"]
        k_c = int(np.clip(zc / grid.dx, 0, grid.nz - 1))
        u_drift = DRIFT_FACTOR * float(wind_fn(0.0)[k_c])
        # A patch is placed at the middle of the observation window, so in wind it
        # crosses the view during the window instead of leaving it early.
        x0 = drift_start(target, u_drift, first_frame_s + 0.5 * window_s)
        r2 = ((grid.xx - x0) / a) ** 2 + ((grid.zz - zc) / b) ** 2
        blob = np.where(grid.fluid, np.exp(-1.5 * r2) * tex, 0.0)

        def initial(fields):
            fields["theta"] += fog["patch_dtheta_k"] * blob
            T = fields["theta"] * pi[:, None]
            qs = thermo.saturation_mixing_ratio(T, p[:, None])
            sat = blob > 0.05
            fields["qv"][...] = np.where(sat, qs, fields["qv"])
            fields["ql"][...] = ql_max * blob

        comp.initial.append(initial)
        # The part of the patch still upwind of the domain enters through the boundary.
        x_b = 0.0 if u_drift >= 0 else grid.nx * grid.dx
        side_b = "left" if u_drift >= 0 else "right"
        zprof = (grid.z - zc) / b
        T_ref = theta_ref * pi
        qs_ref = thermo.saturation_mixing_ratio(T_ref + fog["patch_dtheta_k"] * pi, p)

        def patch_inflow(t, ghosts):
            xc_t = x0 + u_drift * t
            cut = np.exp(-1.5 * (((x_b - xc_t) / a) ** 2 + zprof ** 2))
            if cut.max() < 1e-3:
                return
            ghosts["theta"][side_b] = ghosts["theta"][side_b] + fog["patch_dtheta_k"] * cut
            ghosts["qv"][side_b] = np.where(cut > 0.05, qs_ref, ghosts["qv"][side_b])
            ghosts["ql"][side_b] = ghosts["ql"][side_b] + ql_max * cut

        comp.inflow.append(patch_inflow)
        if fog["moist_inflow"]:
            T_ref = theta_ref * pi
            moist = layer_profile(grid.z, 0.0, fog["inflow_depth_m"], soft_m=1.6 * dx)
            qv_in = qv_ref + moist * (fog["inflow_rh"] * thermo.saturation_mixing_ratio(T_ref, p) - qv_ref)

            def inflow(t, ghosts):
                side = "left" if grid.wind_sign >= 0 else "right"
                ghosts.setdefault("qv", {})[side] = qv_in

            comp.inflow.append(inflow)
        return comp

    if regime == "steam_fog":
        xc = grid.view_x(fog["strip_center_frac"])
        strip = surface & (np.abs(grid.xx - xc) <= 0.5 * fog["strip_width_m"])
        theta_s = fog["theta_ground_k"] + fog["surface_excess_k"]
        T_s = theta_s * pi
        qv_s = thermo.saturation_mixing_ratio(T_s, p)[:, None]
        tau = fog["exchange_time_s"]

        def exchange(t, fields, rates):
            rates["theta"] = rates.get("theta", 0.0) + np.where(strip, (theta_s - fields["theta"]) / tau, 0.0)
            rates["qv"] = rates.get("qv", 0.0) + np.where(strip, np.maximum(qv_s - fields["qv"], 0.0) / tau, 0.0)

        comp.sources.append(exchange)
        comp.info["strip_cells"] = int(strip.sum())
        return comp

    raise ValueError(f"unknown fog regime {regime!r}")
