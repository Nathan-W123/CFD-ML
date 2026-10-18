"""Numerical and data-quality checks that decide whether a simulation is kept.

Numerical checks (per run): finite fields; non-negative concentrations (and a
negligible amount of mass added by clipping); lattice speed / Mach number
within the low-Mach limit; bounded in-view density deviation (weak
compressibility); closed mass budgets for every transported scalar
(smoke, tracer, total water q_t', theta_l'); a small compressibility
correction (see scalars.py); temperature anomalies small enough for the
Boussinesq approximation. Total water is conserved by construction: q_t is
transported and phase change happens only in the diagnosis.

Label checks (per run): a sequence labelled smoke (fog) must show smoke
(fog) with a non-trivial optical depth inside the camera view in several
frames, and a sequence labelled "no fog" must not contain fog-like liquid
water. Otherwise the label would not be supported by what the camera sees
(e.g. a plume blown out of view during spin-up) and the run is rejected.

Where this fits: `simulation.run_simulation` calls `summarize_validity` at
the end of a run; the dataset generator drops runs with valid == False and
logs the reasons.
Calls into: numpy and thermo (fog optics).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping

import numpy as np

from src.cfd import thermo

_DEFAULTS = {
    "max_mass_residual": 1.0e-6,
    "max_clipped_fraction": 1.0e-6,
    "max_density_deviation": 0.05,
    "max_compressibility_fraction": 0.05,
    "max_theta_anomaly_frac": 0.1,
    "max_mach": 0.25,
    "label_line_of_sight_m": 100.0,
    "label_tau_min": 0.1,
    "label_min_cells": 4,
    "label_min_frames": 4,
    "absent_tau_max": 0.05,
}
CONCENTRATIONS = ("qv", "ql", "smoke", "tracer")
COMPRESSIBILITY_CHECKED = ("smoke", "tracer")


def validation_settings(cfg: Mapping[str, Any]) -> Dict[str, float]:
    out = dict(_DEFAULTS)
    out.update(cfg.get("validation", {}) if cfg else {})
    return out


def view_slices(nz: int, nx: int, margin: int):
    """Rows and columns of the camera view: the domain without the sponge layers."""
    return slice(0, max(1, nz - margin)), slice(margin, max(margin + 1, nx - margin))


def label_support(smoke: np.ndarray, lwc: np.ndarray, solid: np.ndarray, margin: int,
                  settings: Mapping[str, float], smoke_extinction: float = 0.0076) -> Dict[str, Any]:
    """Frames in which smoke / fog are visible inside the view, at a nominal line of sight."""
    T, nz, nx = smoke.shape
    rows, cols = view_slices(nz, nx, margin)
    los = settings["label_line_of_sight_m"]
    fluid = ~solid[rows, cols]
    tau_s = smoke_extinction * smoke[:, rows, cols] * los
    tau_f = thermo.fog_extinction(lwc[:, rows, cols]) * los
    out: Dict[str, Any] = {}
    for name, tau in (("smoke", tau_s), ("fog", tau_f)):
        cells = ((tau >= settings["label_tau_min"]) & fluid[None]).sum(axis=(1, 2))
        visible = cells >= settings["label_min_cells"]
        out[f"{name}_visible_frames"] = int(visible.sum())
        out[f"{name}_max_tau"] = float(tau.max()) if tau.size else 0.0
    return out


def plume_unsteadiness(smoke_frames: np.ndarray, uz_frames: np.ndarray, min_conc: float = 0.5) -> float:
    """Temporal std / mean of the mean vertical velocity in smoke-laden cells (0 without smoke)."""
    means = []
    for c, w in zip(smoke_frames, uz_frames):
        m = c > min_conc
        if m.sum() >= 4:
            means.append(float(w[m].mean()))
    if len(means) < 3:
        return 0.0
    means = np.asarray(means)
    return float(means.std() / (np.abs(means).mean() + 0.1))


def summarize_validity(*, fields, transport, solid, max_lattice_speed, max_density_deviation,
                       max_theta_anomaly, spec, smoke_frames, lwc_frames, tracer_frames,
                       uz_frames, cfg) -> Dict[str, Any]:
    """The validity record stored with every simulation (see docs/design.md section 5)."""
    s = validation_settings(cfg)
    reasons: List[str] = []
    fluid = ~solid

    finite = all(np.isfinite(f).all() for f in fields.values())
    if not finite:
        reasons.append("non-finite fields")
    mins = {name: float(fields[name][fluid].min()) if fluid.any() else 0.0 for name in CONCENTRATIONS}
    nonnegative = all(v >= 0.0 for v in mins.values())
    if not nonnegative:
        reasons.append("negative concentration")

    residuals = {}
    clipped = {}
    compress = {}
    for name, budget in transport.budgets.items():
        if name.startswith("tex_"):
            continue  # texture coordinates are not conserved quantities
        current = transport.total(name, fields[name])
        residuals[name] = float(budget.residual(current))
        scale = max(budget.scale(), 1e-300)
        clipped[name] = float(budget.clipped / scale)
        compress[name] = float(abs(budget.compressibility) / scale)
        if abs(residuals[name]) > s["max_mass_residual"]:
            reasons.append(f"mass budget residual {residuals[name]:.2e} for {name}")
        if clipped[name] > s["max_clipped_fraction"]:
            reasons.append(f"clipped fraction {clipped[name]:.2e} for {name}")
        if name in COMPRESSIBILITY_CHECKED and compress[name] > s["max_compressibility_fraction"]:
            reasons.append(f"compressibility correction {compress[name]:.2e} for {name}")

    max_mach = max_lattice_speed / math.sqrt(1.0 / 3.0)
    if max_mach > s["max_mach"]:
        reasons.append(f"Mach {max_mach:.3f} > {s['max_mach']}")
    if max_density_deviation > s["max_density_deviation"]:
        reasons.append(f"density deviation {max_density_deviation:.3f}")
    theta_frac = max_theta_anomaly / float(getattr(spec, "theta0", 288.0))
    if theta_frac > s["max_theta_anomaly_frac"]:
        reasons.append(f"theta anomaly {max_theta_anomaly:.1f} K beyond the Boussinesq range")

    margin = int(cfg.get("render", {}).get("view_margin_cells", spec.sponge_width)) if cfg else spec.sponge_width
    support = label_support(smoke_frames, lwc_frames, solid, margin, s,
                            float(cfg.get("render", {}).get("smoke_mass_extinction_m2_mg", 0.0076)) if cfg else 0.0076)
    smoke_label, fog_label = int(spec.labels[0]), int(spec.labels[1])
    min_frames = int(s["label_min_frames"])
    if smoke_label and support["smoke_visible_frames"] < min_frames:
        reasons.append("smoke label unsupported: plume not visible in enough frames")
    if fog_label and support["fog_visible_frames"] < min_frames:
        reasons.append("fog label unsupported: fog not visible in enough frames")
    if not smoke_label and support["smoke_max_tau"] > s["absent_tau_max"]:
        reasons.append("smoke present in a no-smoke scenario")
    if not fog_label and support["fog_max_tau"] > s["absent_tau_max"]:
        reasons.append("condensate present in a no-fog scenario")

    return {
        "finite": bool(finite),
        "nonnegative": bool(nonnegative),
        "min_values": mins,
        "max_lattice_speed": float(max_lattice_speed),
        "max_mach": float(max_mach),
        "max_density_deviation": float(max_density_deviation),
        "mass_residuals": residuals,
        "clipped_fraction": clipped,
        "compressibility_fraction": compress,
        "max_theta_anomaly_k": float(max_theta_anomaly),
        "plume_unsteadiness": plume_unsteadiness(smoke_frames, uz_frames),
        "budgets": {name: b.as_dict() for name, b in transport.budgets.items() if not name.startswith("tex_")},
        "label_support": support,
        "dt_used": float(spec.dt),
        "dt_halvings": int(spec.dt_halvings),
        "reasons": reasons,
        "valid": not reasons,
    }
