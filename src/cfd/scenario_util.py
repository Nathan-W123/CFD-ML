"""Shared building blocks for scenario construction (grid geometry, noise, schedules).

Scenario parameters are sampled once (`scenarios.sample_scenario`) and stored
as plain JSON. Everything time-dependent (gusts, intermittency, pulsing) is a
deterministic function of stored phases and periods, so a SimulationSpec can
be rebuilt from the stored parameters alone.

Profiles (scale-agnostic code): a profile is a full simulation config (the
near-field 320 m / 5 m-cell / 5 s profile in configs/simulation.yaml, the
far-field 3.2 km / 50 m-cell / one-frame-per-minute profile in
configs/simulation_far.yaml). Physical ranges (sizes, heights, depths,
periods) come from the config, with the near-field values as code defaults.
What is left in code follows two rules: numerical widths (edge ramps, tanh
fronts, layer softness, minimum thicknesses) scale with the cell size dx
(written as multiples of dx that equal the old near-field metres at
dx = 5 m), and schedule times defined relative to the observation window
(onsets, switch-offs, activity ramps) scale with the frame interval through
`time_scale` (1 in the near-field profile).

Where this fits: used by smoke.py, fog.py and scenarios.py.
Calls into: numpy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


REFERENCE_FRAME_INTERVAL_S = 5.0   # the near-field profile the built-in schedule times were written for


def grid_dx(cfg: Dict[str, Any]) -> float:
    """Cell size (m) of a profile."""
    g = cfg.get("grid", {})
    return float(g.get("domain_width_m", 320.0)) / int(g.get("nx", 64))


def time_scale(frame_interval_s: float) -> float:
    """Factor applied to window-relative schedule times: frame interval / 5 s (1 near field, 12 far field)."""
    return float(frame_interval_s) / REFERENCE_FRAME_INTERVAL_S


def block_range(block: Dict[str, Any], key: str, default: Sequence[float]) -> Sequence[float]:
    """A [lo, hi] range from a config block, falling back to the near-field default."""
    value = block.get(key) if block else None
    return default if value is None else value


def uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return float(lo + (hi - lo) * rng.random())


def log_uniform(rng: np.random.Generator, bounds: Sequence[float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_noise_spec(rng: np.random.Generator, period_range: Sequence[float], n_modes: int = 4) -> Dict[str, List[float]]:
    """Periods, phases and weights of a smooth random signal (see smooth_noise)."""
    lo, hi = float(period_range[0]), float(period_range[1])
    periods = [float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) for _ in range(n_modes)]
    phases = [float(rng.uniform(0.0, 2 * math.pi)) for _ in range(n_modes)]
    weights = [float(rng.uniform(0.5, 1.0)) for _ in range(n_modes)]
    return {"periods": periods, "phases": phases, "weights": weights}


def smooth_noise(t: float, spec: Dict[str, List[float]]) -> float:
    """A smooth signal with zero mean and unit standard deviation (sum of sinusoids)."""
    w = np.asarray(spec["weights"], dtype=float)
    val = sum(wi * math.sin(2 * math.pi * t / p + ph) for wi, p, ph in zip(w, spec["periods"], spec["phases"]))
    return float(val / math.sqrt(0.5 * float((w ** 2).sum())))


def smooth_noise01(t: float, spec: Dict[str, List[float]]) -> float:
    """smooth_noise squashed into (0, 1)."""
    return 0.5 + 0.5 * math.tanh(smooth_noise(t, spec))


def activity_window(t: float, t_on: float, t_off: Optional[float], ramp_s: float = 2.0) -> float:
    """0 before t_on, ramps to 1 over ramp_s, back to 0 over ramp_s after t_off."""
    if t < t_on:
        return 0.0
    a = min(1.0, (t - t_on) / ramp_s) if ramp_s > 0 else 1.0
    if t_off is not None and t > t_off:
        a *= max(0.0, 1.0 - (t - t_off) / ramp_s) if ramp_s > 0 else 0.0
    return a


def layer_profile(z: np.ndarray, base: float, top: float, soft_m: float = 10.0) -> np.ndarray:
    """Smooth top-hat in z: ~1 between base and top, falling off over soft_m (base 0 touches the ground)."""
    z = np.asarray(z, dtype=float)
    up = 0.5 * (1.0 + np.tanh((z - base) / soft_m)) if base > 0 else np.ones_like(z)
    down = 0.5 * (1.0 - np.tanh((z - top) / soft_m))
    return up * down


@dataclass
class Grid:
    """Geometry of one simulation: cell centres, terrain and the camera view."""

    nx: int
    nz: int
    dx: float
    solid: np.ndarray            # (nz, nx) bool
    view_margin: int             # cells cropped at left, right and top (sponges)
    wind_sign: float = 1.0       # +1: wind blows towards +x (enters on the left)

    def __post_init__(self):
        self.solid = np.asarray(self.solid, dtype=bool)
        self.fluid = ~self.solid
        self.z = (np.arange(self.nz) + 0.5) * self.dx
        self.x = (np.arange(self.nx) + 0.5) * self.dx
        self.zz, self.xx = np.meshgrid(self.z, self.x, indexing="ij")
        # First fluid row of each column (terrain is a height field, solid below).
        self.k_ground = np.argmax(self.fluid, axis=0)
        self.ground_height = self.k_ground * self.dx
        self.height_above_ground = self.zz - self.ground_height[None, :]

    def view_x(self, frac: float) -> float:
        """x (m) at fraction `frac` across the camera view (sponges excluded)."""
        m = self.view_margin
        return (m + float(frac) * (self.nx - 2 * m)) * self.dx

    def hilltop_x(self) -> float:
        i = int(np.argmax(self.ground_height[self.view_margin:self.nx - self.view_margin])) + self.view_margin
        return float(self.x[i])

    def surface_cells(self) -> np.ndarray:
        """(nz, nx) bool: the first fluid cell above the ground or terrain in each column."""
        out = np.zeros((self.nz, self.nx), dtype=bool)
        out[self.k_ground, np.arange(self.nx)] = True
        return out & self.fluid

    def near_surface(self, depth_m: float) -> np.ndarray:
        """Fluid cells within depth_m of the local ground (vertical distance)."""
        return self.fluid & (self.height_above_ground < depth_m)


@dataclass
class Component:
    """What one physical component contributes to a SimulationSpec."""

    initial: List[Callable[[Dict[str, np.ndarray]], None]] = field(default_factory=list)
    inflow: List[Callable[[float, Dict[str, Dict[str, np.ndarray]]], None]] = field(default_factory=list)
    sources: List[Callable[[float, Dict[str, np.ndarray], Dict[str, Any]], None]] = field(default_factory=list)
    settling: float = 0.0           # tracer settling velocity, m/s
    precip_settling: float = 0.0    # > 0: the component emits into the separate "precip" scalar
    info: Dict[str, Any] = field(default_factory=dict)


def drift_start(target_x: float, u_drift: float, travel_s: float) -> float:
    """Where (x, m) a feature drifting at u_drift must start so it is at target_x after travel_s.

    Features that drift with the wind (fog patches, bank fronts, advected smoke
    fronts) are placed by their position at the FIRST FRAME, not at t = 0;
    otherwise long spin-ups in strong wind carry them out of the view before
    the camera starts. The start can lie upwind of the domain; the caller
    then feeds the part outside the domain in through the inflow boundary.
    """
    return float(target_x - u_drift * travel_s)


def source_mask(grid: Grid, x_center: float, width: float, height_above_ground: float,
                lift_cells: int = 1, rows: int = 2) -> np.ndarray:
    """(nz, nx) float mask of a ground-based source: columns within width/2 of x_center
    (at least two), `rows` rows starting `lift_cells` + height/dx above the local ground."""
    half = max(0.5 * width, grid.dx)
    cols = np.nonzero(np.abs(grid.x - x_center) <= half + 1e-9)[0]
    if cols.size == 0:
        cols = np.array([int(np.clip(round(x_center / grid.dx - 0.5), 0, grid.nx - 1))])
    mask = np.zeros((grid.nz, grid.nx))
    off = int(lift_cells) + int(round(height_above_ground / grid.dx))
    for i in cols:
        k0 = int(grid.k_ground[i]) + off
        for k in range(k0, min(k0 + rows, grid.nz)):
            if grid.fluid[k, i]:
                mask[k, i] = 1.0
    return mask


def smooth_texture(grid: Grid, seed: int, scale_m: float = 40.0, contrast: float = 0.5,
                   n_modes: int = 12) -> np.ndarray:
    """Positive multiplicative texture (mean ~1) made of random plane waves of wavelength ~scale_m."""
    rng = np.random.default_rng(int(seed))
    acc = np.zeros((grid.nz, grid.nx))
    for _ in range(n_modes):
        lam = scale_m * math.exp(rng.uniform(-0.5, 0.5))
        ang = rng.uniform(0.0, 2 * math.pi)
        kx, kz = 2 * math.pi / lam * math.cos(ang), 2 * math.pi / lam * math.sin(ang)
        acc += np.sin(kx * grid.xx + kz * grid.zz + rng.uniform(0.0, 2 * math.pi))
    acc /= math.sqrt(0.5 * n_modes)
    return np.clip(1.0 + contrast * acc, 0.15, None)


def smooth_profile_noise(n: int, seed: int, scale_cells: float = 8.0, n_modes: int = 6) -> np.ndarray:
    """1D smooth noise (zero mean, ~unit std) of length n."""
    rng = np.random.default_rng(int(seed))
    s = np.arange(n, dtype=float)
    acc = np.zeros(n)
    for _ in range(n_modes):
        lam = scale_cells * math.exp(rng.uniform(-0.5, 0.7))
        acc += np.sin(2 * math.pi * s / lam + rng.uniform(0.0, 2 * math.pi))
    return acc / math.sqrt(0.5 * n_modes)
