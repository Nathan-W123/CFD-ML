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

Fog starts established (the 60 s window is far too short to form a fog bank
from clear air) and evolves through transport, mixing, cooling and warming.
The air around established fog is near saturation (RH 0.92-0.99; drier air
only where a regime calls for it, e.g. dry inflow eroding a fog), and fog air
starts close to neutrally buoyant (saturated at about the ambient
temperature), so banks neither slump nor lift as an artefact of the set-up.

Where this fits: `scenarios.sample_scenario` calls `sample_fog`;
`scenarios.build_spec` calls `fog_reference` (moist ambient profiles) and
`fog_component` (initial fields, inflow, sources).
Calls into: thermo, scenario_util.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.cfd import thermo
from src.cfd.scenario_util import (
    Component, Grid, drift_start, layer_profile, sample_noise_spec, smooth_noise, smooth_profile_noise,
    smooth_texture, uniform,
)

DRIFT_FACTOR = 0.85  # drifting features move a little slower than the free-stream wind at their height

FOG_WIND = {
    "radiation_fog": (0.0, 3.5),
    "advected_fog_bank": (1.0, 9.0),
    "terrain_fog": (0.0, 4.0),
    "fog_patch": (0.0, 7.0),
    "fog_dissipating": (0.0, 9.0),
    "steam_fog": (0.0, 5.0),
}
AMBIENT_LAYER_REGIMES = ("radiation_fog", "terrain_fog", "fog_dissipating")


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
    lwc = uniform(rng, fc.get("lwc_g_m3", [0.05, 0.4]))
    top = uniform(rng, fc.get("fog_top_m", [40.0, 160.0]))
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
    }
    if regime == "radiation_fog":
        out.update(inversion_dtheta_dz=uniform(rng, block.get("inversion_dtheta_dz", [0.01, 0.06])),
                   surface_cooling_k_s=uniform(rng, block.get("surface_cooling_k_s", [0.0002, 0.002])))
    elif regime == "advected_fog_bank":
        out.update(front_frac=uniform(rng, block.get("front_x_frac", [0.1, 0.6])),
                   bank_dtheta_k=uniform(rng, (-0.5, 0.3)),
                   domain_rh=uniform(rng, rh_outside),
                   modulation=sample_noise_spec(rng, (10.0, 40.0)),
                   inversion_dtheta_dz=uniform(rng, (0.0, 0.03)))
    elif regime == "terrain_fog":
        out.update(cold_layer_k=uniform(rng, block.get("cold_layer_k", [0.5, 3.0])),
                   cold_layer_depth_m=uniform(rng, (8.0, 20.0)),
                   fill_frac=uniform(rng, (0.5, 1.1)),
                   inversion_dtheta_dz=uniform(rng, (0.01, 0.05)),
                   surface_cooling_k_s=uniform(rng, (0.0003, 0.002)))
    elif regime == "fog_patch":
        out.update(center_frac=uniform(rng, (0.15, 0.85)),
                   center_height_m=uniform(rng, (0.0, 90.0)),
                   radius_m=uniform(rng, block.get("patch_radius_m", [20.0, 70.0])),
                   aspect=uniform(rng, (0.4, 1.0)),
                   ambient_rh=uniform(rng, block.get("ambient_rh", rh_outside)),
                   patch_dtheta_k=uniform(rng, (-0.3, 0.1)),
                   moist_inflow=bool(rng.random() < 0.3),
                   inflow_rh=uniform(rng, (0.98, 1.01)),
                   inflow_depth_m=uniform(rng, (20.0, 100.0)),
                   inversion_dtheta_dz=uniform(rng, (0.0, 0.03)))
    elif regime == "fog_dissipating":
        mech = ["dry_inflow", "surface_warming", "wind_ramp"][int(rng.integers(0, 3))]
        out.update(mechanism=mech,
                   mechanism_onset_s=uniform(rng, (-30.0, 10.0)),   # relative to the first frame
                   inversion_dtheta_dz=uniform(rng, (0.005, 0.04)),
                   dry_inflow_rh=uniform(rng, block.get("dry_inflow_rh", [0.4, 0.85])),
                   surface_warming_k_s=uniform(rng, block.get("surface_warming_k_s", [0.002, 0.02])),
                   ramp_factor=uniform(rng, (1.3, 2.0)))
    elif regime == "steam_fog":
        # Re-draw until mixing surface air with ambient air actually supersaturates.
        for _ in range(50):
            theta_g = uniform(rng, (263.0, 280.0))
            excess = uniform(rng, block.get("surface_excess_k", [8.0, 15.0]))
            rh = uniform(rng, block.get("ambient_rh", [0.85, 0.98]))
            ss = steam_supersaturation(theta_g, excess, rh)
            if ss >= 5e-5:
                break
        out.update(surface_excess_k=excess,
                   strip_center_frac=uniform(rng, (0.2, 0.8)),
                   strip_width_m=uniform(rng, block.get("strip_width_m", [20.0, 120.0])),
                   exchange_time_s=uniform(rng, (2.0, 6.0)),
                   theta_ground_k=theta_g,
                   ambient_rh=rh,
                   mixing_supersaturation=ss)
    else:
        raise ValueError(f"unknown fog regime {regime!r}")
    return out


def _ql_from_lwc(lwc_g_m3: float, rho_air: float) -> float:
    return lwc_g_m3 / (rho_air * 1000.0)


def _layer_lwc_shape(z: np.ndarray, top: float) -> np.ndarray:
    """Vertical LWC shape of a fog layer: grows with height, peaks near 0.8 top, soft top."""
    top = np.maximum(np.asarray(top, dtype=float), 1.0)
    s = np.clip(z / top, 0.0, None)
    return (0.35 + 0.65 * np.clip(s / 0.8, 0.0, 1.0)) * 0.5 * (1.0 - np.tanh((z - top) / np.maximum(4.0, 0.08 * top)))


def fog_reference(fog: Optional[Dict[str, Any]], grid: Grid, atm: Dict[str, Any], p: np.ndarray,
                  pi: np.ndarray, rho_air: float):
    """Ambient (reference) profiles theta_ref, qv_ref, ql_ref (each (nz,)) for the scenario.

    Without fog the atmosphere block alone sets them; fog regimes add an
    inversion and, for ambient fog layers, a saturated layer holding liquid.
    """
    z = grid.z
    theta = atm["theta_ground_k"] + atm["lapse_dtheta_dz"] * z
    rh = np.full_like(z, atm["relative_humidity"])
    ql = np.zeros_like(z)
    if fog is not None:
        regime = fog["regime"]
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
            h_inv = top + 20.0
            theta = atm["theta_ground_k"] + inv * np.minimum(z, h_inv) + max(atm["lapse_dtheta_dz"], 0.0) * np.maximum(z - h_inv, 0.0)
            above = fog["rh_above"]
            if regime in AMBIENT_LAYER_REGIMES:
                inside = layer_profile(z, 0.0, top, soft_m=max(3.0, 0.05 * top))
                rh = above + (1.0 - above) * inside
                ql = _ql_from_lwc(fog["lwc_g_m3"], rho_air) * _layer_lwc_shape(z, top)
            elif regime == "advected_fog_bank":
                inside = layer_profile(z, 0.0, top, soft_m=10.0)
                rh = above + (fog["domain_rh"] - above) * inside
            elif regime == "fog_patch":
                rh[:] = fog["ambient_rh"]
    T = theta * pi
    qv = thermo.mixing_ratio_from_rh(np.minimum(rh, 1.0), T, p)
    return theta, qv, ql


def fog_component(fog: Dict[str, Any], grid: Grid, refs, p: np.ndarray, pi: np.ndarray,
                  rho_air: float, wind_fn, first_frame_s: float = 0.0) -> Component:
    regime = fog["regime"]
    theta_ref, qv_ref, ql_ref = refs
    comp = Component()
    tex = smooth_texture(grid, fog["texture_seed"], scale_m=50.0, contrast=fog["heterogeneity"])
    ql_max = _ql_from_lwc(fog["lwc_g_m3"], rho_air)
    surface = grid.surface_cells()

    if regime in AMBIENT_LAYER_REGIMES:
        top = fog.get("effective_top_m", fog["fog_top_m"])
        undul = 1.0 + fog["top_undulation"] * smooth_profile_noise(grid.nx, fog["top_seed"], scale_cells=12.0)
        top_x = np.maximum(top * undul, 5.0)
        ql0 = ql_max * _layer_lwc_shape(grid.zz, top_x[None, :]) * tex

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
            # "wind_ramp" is applied by scenarios through the wind function.
        return comp

    if regime == "advected_fog_bank":
        top = fog["fog_top_m"]
        layer = layer_profile(grid.z, 0.0, top, soft_m=8.0)
        theta_bank = theta_ref + fog["bank_dtheta_k"] * layer
        qv_bank = np.where(layer > 0.05, thermo.saturation_mixing_ratio(theta_bank * pi, p), qv_ref)
        ql_bank = ql_max * _layer_lwc_shape(grid.z, top)
        upwind_left = grid.wind_sign >= 0
        target = grid.view_x(fog["front_frac"] if upwind_left else 1.0 - fog["front_frac"])
        k_mid = int(np.clip(0.5 * top / grid.dx, 0, grid.nz - 1))
        u_drift = DRIFT_FACTOR * float(wind_fn(0.0)[k_mid])
        front_x = drift_start(target, u_drift, first_frame_s)
        dist = (front_x - grid.xx) if upwind_left else (grid.xx - front_x)
        inside = 0.5 * (1.0 + np.tanh(dist / 15.0))

        def initial(fields):
            fields["theta"] += inside * (theta_bank - theta_ref)[:, None]
            fields["qv"][...] = fields["qv"] + inside * (qv_bank - qv_ref)[:, None]
            fields["ql"][...] = np.where(grid.fluid, inside * ql_bank[:, None] * tex, 0.0)

        mod = fog["modulation"]

        def inflow(t, ghosts):
            side = "left" if upwind_left else "right"
            level = 0.85 + 0.15 * np.tanh(smooth_noise(t, mod))
            ghosts.setdefault("theta", {})[side] = theta_bank
            ghosts.setdefault("qv", {})[side] = qv_bank
            ghosts.setdefault("ql", {})[side] = level * ql_bank

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
        x0 = drift_start(target, u_drift, first_frame_s)
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
            moist = layer_profile(grid.z, 0.0, fog["inflow_depth_m"], soft_m=8.0)
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
