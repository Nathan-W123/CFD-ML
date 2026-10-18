"""Finite-volume advection-diffusion of the transported scalars, with exact budgets.

Every scalar (potential temperature, smoke, water vapour, fog liquid water,
distractor tracer) obeys

    dC/dt + div(u C) = div(K grad C) + S

on the LBM grid. Advection uses face fluxes (face velocity = mean of the two
adjacent cell-centre velocities; zero through walls and terrain) with a
minmod-limited MUSCL reconstruction (or first-order upwind), forward Euler in
time; both are positivity preserving for the CFL numbers the LBM allows
(|u| dt/dx <= 0.17 against a limit of 0.5). Diffusion is explicit with no
flux through walls or open boundaries. Settling scalars (dust) fall at their
settling velocity relative to the air and deposit on the ground and terrain.

The LBM flow is only weakly compressible: its discrete divergence is tiny in
the interior (~1e-6 per step) but reaches ~1e-2 at boundary corners and
terrain edges, where the purely conservative form lets a scalar pile up
without bound (a uniform 288 K field drifted by 50 K at an outflow corner).
The update therefore uses the advective form of the incompressible equation,
flux divergence minus `C div(u)`: a uniform field stays uniform and every
update is a convex combination of neighbours (bounded, non-negative). The
`C div(u)` term is booked in the budget as `compressibility`, so the budget
still closes to round-off and its size is reported as a validity metric.

`MassBudget` accumulates what entered and left through the open boundaries,
what settled out, what sources added, what phase change moved between
scalars, the compressibility correction and what (if anything)
non-negativity clipping added.

Where this fits: the scalar half of the coupled solver; `simulation.py` calls
`step` after each LBM step, then `add_source` and `record_conversion`.
Calls into: numpy and numba (single-threaded kernels); LatticeUnits from lbm.
Start reading at: ScalarTransport.step, then _transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional

import numba
import numpy as np

from src.cfd.lbm import LatticeUnits

MAX_CFL = 0.5
MAX_DIFFUSION_NUMBER = 0.2
_LATERAL_MODES = {"open": 0, "periodic": 1, "wall": 2}
_TOP_MODES = {"open": 0, "wall": 2}
_TINY = 1e-300


@dataclass(frozen=True)
class ScalarSpec:
    name: str
    diffusivity: float               # m^2/s
    nonnegative: bool = True         # theta is the only signed-free exception
    settling_velocity: float = 0.0   # m/s, > 0 falls relative to the air


@dataclass
class MassBudget:
    """Cumulative mass accounting for one scalar, in field-units * m^2 (per metre depth)."""

    initial: float = 0.0
    sources: float = 0.0
    inflow: float = 0.0
    outflow: float = 0.0
    deposited: float = 0.0
    conversion: float = 0.0
    compressibility: float = 0.0     # C div(u) correction of the weakly compressible flow
    clipped: float = 0.0

    def expected(self) -> float:
        return (self.initial + self.sources + self.inflow - self.outflow - self.deposited
                + self.conversion + self.compressibility + self.clipped)

    def scale(self) -> float:
        """A magnitude for relative residuals: everything that ever entered or existed."""
        return (abs(self.initial) + abs(self.sources) + abs(self.inflow) + abs(self.outflow)
                + self.deposited + abs(self.conversion) + abs(self.compressibility) + self.clipped)

    def residual(self, current: float) -> float:
        """(current - expected) relative to `scale()` (0 when nothing was ever present)."""
        scale = self.scale()
        if scale <= _TINY:
            return 0.0
        return (current - self.expected()) / scale

    def as_dict(self) -> Dict[str, float]:
        return {
            "initial": self.initial, "sources": self.sources, "inflow": self.inflow,
            "outflow": self.outflow, "deposited": self.deposited,
            "conversion": self.conversion, "compressibility": self.compressibility,
            "clipped": self.clipped,
        }


@numba.njit(cache=True, inline="always")
def _minmod(a, b):
    if a * b <= 0.0:
        return 0.0
    if abs(a) < abs(b):
        return a
    return b


@numba.njit(cache=True, fastmath=True)
def _transport(c, uf, wf, ws, kd, solid, muscl, lateral, top,
               gl, gr, gt, has_gl, has_gr, has_gt, nonneg, c_new, acc):
    """One forward-Euler step of advection + diffusion + settling.

    c      (nz, nx) field                      uf (nz, nx+1) lattice x-face velocity
    wf     (nz+1, nx) lattice z-face velocity  ws settling (lattice, > 0 down)
    kd     K dt / dx^2                         solid (nz, nx) bool
    gl/gr  (nz,) ghost values left/right       gt (nx,) ghost values at the top
    acc    [inflow, outflow, deposited, clipped, compressibility] in cell-sum units
           (multiplied by dx^2 by the caller)
    """
    nz, nx = c.shape
    fx = np.zeros((nz, nx + 1))
    fz = np.zeros((nz + 1, nx))

    # ---- x faces -------------------------------------------------------
    for k in range(nz):
        for j in range(nx + 1):
            u = uf[k, j]
            jl = j - 1
            jr = j
            if lateral == 1:
                if j == nx:
                    continue  # periodic: face nx is face 0
                jl = (j - 1) % nx
            left_out = jl < 0
            right_out = jr >= nx
            if (not left_out and solid[k, jl]) or (not right_out and solid[k, jr]):
                continue
            flux = 0.0
            if u != 0.0:
                if left_out or right_out:
                    # Open boundary face (wall faces carry u = 0 and never reach here).
                    inside = jr if left_out else jl
                    entering = (left_out and u > 0.0) or (right_out and u < 0.0)
                    if entering:
                        if left_out:
                            ghost = gl[k] if has_gl else c[k, inside]
                        else:
                            ghost = gr[k] if has_gr else c[k, inside]
                        flux = u * ghost
                    else:
                        # Outflow: the face value is the boundary cell (zero gradient outside,
                        # so the limited slope vanishes).
                        flux = u * c[k, inside]
                    net_in = flux if left_out else -flux
                    if entering:
                        acc[0] += net_in
                    else:
                        acc[1] -= net_in
                else:
                    cl = c[k, jl]
                    cr = c[k, jr]
                    if u > 0.0:
                        cf = cl
                        if muscl:
                            jll = jl - 1
                            if lateral == 1:
                                jll = jll % nx
                            if jll >= 0 and not solid[k, jll]:
                                cf = cl + 0.5 * _minmod(cl - c[k, jll], cr - cl)
                    else:
                        cf = cr
                        if muscl:
                            jrr = jr + 1
                            if lateral == 1:
                                jrr = jrr % nx
                            if jrr < nx and not solid[k, jrr]:
                                cf = cr - 0.5 * _minmod(c[k, jrr] - cr, cr - cl)
                    flux = u * cf
            if not (left_out or right_out):
                flux -= kd * (c[k, jr] - c[k, jl])
            fx[k, j] = flux
            if lateral == 1 and j == 0:
                fx[k, nx] = flux

    # ---- z faces -------------------------------------------------------
    for m in range(nz + 1):
        for i in range(nx):
            kb = m - 1   # cell below the face
            ka = m       # cell above the face
            below_out = kb < 0
            above_out = ka >= nz
            below_solid = (not below_out) and solid[kb, i]
            above_solid = (not above_out) and solid[ka, i]
            if above_solid:
                continue
            if below_out or below_solid:
                # Floor or terrain top: only settling deposition crosses it.
                if not above_out and ws > 0.0:
                    flux = -ws * c[ka, i]
                    fz[m, i] = flux
                    acc[2] += -flux
                continue
            if above_out:
                if top == 2:
                    continue  # wall: nothing crosses (settling scalars rest on it)
                w = wf[m, i] - ws
                if w < 0.0:
                    ghost = gt[i] if has_gt else c[kb, i]
                    flux = w * ghost
                    acc[0] -= flux
                else:
                    flux = w * c[kb, i]
                    acc[1] += flux
                fz[m, i] = flux
                continue
            w = wf[m, i] - ws
            cb = c[kb, i]
            ca = c[ka, i]
            if w > 0.0:
                cf = cb
                if muscl and kb - 1 >= 0 and not solid[kb - 1, i]:
                    cf = cb + 0.5 * _minmod(cb - c[kb - 1, i], ca - cb)
            else:
                cf = ca
                if muscl and ka + 1 < nz and not solid[ka + 1, i]:
                    cf = ca - 0.5 * _minmod(c[ka + 1, i] - ca, ca - cb)
            fz[m, i] = w * cf - kd * (ca - cb)

    # ---- update ----------------------------------------------------------
    for k in range(nz):
        for i in range(nx):
            if solid[k, i]:
                c_new[k, i] = c[k, i]
                continue
            # Discrete divergence of the AIR velocity (settling excluded, so deposition
            # stays a sink); the faces are the ones used for the fluxes above.
            div = (uf[k, i + 1] - uf[k, i]) + (wf[k + 1, i] - wf[k, i])
            corr = c[k, i] * div
            acc[4] += corr
            v = c[k, i] - (fx[k, i + 1] - fx[k, i]) - (fz[k + 1, i] - fz[k, i]) + corr
            if nonneg and v < 0.0:
                acc[3] += -v
                v = 0.0
            c_new[k, i] = v


class ScalarTransport:
    """Transport of several scalars by one velocity field; see the module docstring."""

    def __init__(self, nx: int, nz: int, units: LatticeUnits, solid: Optional[np.ndarray],
                 specs: List[ScalarSpec], scheme: str = "muscl",
                 lateral: str = "open", top: str = "open"):
        if scheme not in ("muscl", "upwind"):
            raise ValueError(f"scheme must be 'muscl' or 'upwind', got {scheme!r}")
        if lateral not in _LATERAL_MODES:
            raise ValueError(f"lateral must be one of {sorted(_LATERAL_MODES)}, got {lateral!r}")
        if top not in _TOP_MODES:
            raise ValueError(f"top must be one of {sorted(_TOP_MODES)}, got {top!r}")
        self.nx, self.nz, self.units = int(nx), int(nz), units
        if solid is None:
            solid = np.zeros((nz, nx), dtype=bool)
        self.solid = np.ascontiguousarray(solid, dtype=np.bool_)
        if self.solid.shape != (nz, nx):
            raise ValueError(f"solid has shape {self.solid.shape}, expected {(nz, nx)}")
        self.fluid = ~self.solid
        self.specs: Dict[str, ScalarSpec] = {s.name: s for s in specs}
        self.scheme = scheme
        self.lateral = lateral
        self.top = top
        self.cell_area = units.dx * units.dx
        for spec in specs:
            d = spec.diffusivity * units.dt / units.dx ** 2
            if d > MAX_DIFFUSION_NUMBER:
                raise ValueError(f"diffusion number {d:.3f} for {spec.name!r} exceeds {MAX_DIFFUSION_NUMBER}")
            if spec.settling_velocity < 0.0:
                raise ValueError(f"settling velocity of {spec.name!r} must be >= 0")
        self.budgets: Dict[str, MassBudget] = {s.name: MassBudget() for s in specs}
        self._empty_nz = np.zeros(nz)
        self._empty_nx = np.zeros(nx)
        self._buf = np.empty((nz, nx))

    # ------------------------------------------------------------------
    def face_velocities(self, ux: np.ndarray, uz: np.ndarray):
        """Lattice face velocities (uf (nz, nx+1), wf (nz+1, nx)) from cell-centre m/s."""
        s = self.units.dt / self.units.dx
        ul = np.where(self.fluid, ux, 0.0) * s
        wl = np.where(self.fluid, uz, 0.0) * s
        nz, nx = self.nz, self.nx
        uf = np.zeros((nz, nx + 1))
        uf[:, 1:nx] = 0.5 * (ul[:, :-1] + ul[:, 1:])
        if self.lateral == "open":
            uf[:, 0] = ul[:, 0]
            uf[:, nx] = ul[:, nx - 1]
        elif self.lateral == "periodic":
            uf[:, 0] = uf[:, nx] = 0.5 * (ul[:, nx - 1] + ul[:, 0])
        wf = np.zeros((nz + 1, nx))
        wf[1:nz, :] = 0.5 * (wl[:-1, :] + wl[1:, :])
        if self.top == "open":
            wf[nz, :] = wl[nz - 1, :]
        # No flow through faces touching a solid cell.
        solid = self.solid
        uf[:, 1:nx][solid[:, :-1] | solid[:, 1:]] = 0.0
        wf[1:nz, :][solid[:-1, :] | solid[1:, :]] = 0.0
        return np.ascontiguousarray(uf), np.ascontiguousarray(wf)

    def max_cfl(self, ux: np.ndarray, uz: np.ndarray) -> float:
        s = self.units.dt / self.units.dx
        fl = self.fluid
        if not fl.any():
            return 0.0
        return float((np.abs(ux[fl]) + np.abs(uz[fl])).max() * s)

    def total(self, name: str, field: np.ndarray) -> float:
        return float(field[self.fluid].sum() * self.cell_area)

    def start_budget(self, fields: Mapping[str, np.ndarray]) -> None:
        for name in self.specs:
            self.budgets[name] = MassBudget(initial=self.total(name, fields[name]))

    def residuals(self, fields: Mapping[str, np.ndarray]) -> Dict[str, float]:
        return {name: self.budgets[name].residual(self.total(name, fields[name])) for name in self.specs}

    # ------------------------------------------------------------------
    def step(self, fields: Dict[str, np.ndarray], ux: np.ndarray, uz: np.ndarray,
             inflow: Optional[Mapping[str, Mapping[str, np.ndarray]]] = None) -> None:
        """Advance every field one dt in place (advection, diffusion, settling)."""
        cfl = self.max_cfl(ux, uz)
        if not cfl <= MAX_CFL:
            raise ValueError(f"advective CFL {cfl:.3f} exceeds {MAX_CFL}")
        uf, wf = self.face_velocities(ux, uz)
        s_lat = self.units.dt / self.units.dx
        inflow = inflow or {}
        lat = _LATERAL_MODES[self.lateral]
        top = _TOP_MODES[self.top]
        for name, spec in self.specs.items():
            c = fields[name]
            if c.shape != (self.nz, self.nx):
                raise ValueError(f"field {name!r} has shape {c.shape}, expected {(self.nz, self.nx)}")
            ghosts = inflow.get(name, {})
            gl = ghosts.get("left")
            gr = ghosts.get("right")
            gt = ghosts.get("top")
            acc = np.zeros(5)
            _transport(
                np.ascontiguousarray(c, dtype=np.float64), uf, wf,
                spec.settling_velocity * s_lat,
                spec.diffusivity * self.units.dt / self.units.dx ** 2,
                self.solid, self.scheme == "muscl", lat, top,
                self._empty_nz if gl is None else np.ascontiguousarray(gl, dtype=np.float64),
                self._empty_nz if gr is None else np.ascontiguousarray(gr, dtype=np.float64),
                self._empty_nx if gt is None else np.ascontiguousarray(gt, dtype=np.float64),
                gl is not None, gr is not None, gt is not None,
                spec.nonnegative, self._buf, acc,
            )
            c[...] = self._buf
            b = self.budgets[name]
            b.inflow += acc[0] * self.cell_area
            b.outflow += acc[1] * self.cell_area
            b.deposited += acc[2] * self.cell_area
            b.clipped += acc[3] * self.cell_area
            b.compressibility += acc[4] * self.cell_area

    def add_source(self, name: str, field: np.ndarray, rate: np.ndarray) -> None:
        """field += dt * rate on fluid cells (rate in field-units / s); accounted as sources."""
        inc = np.where(self.fluid, rate, 0.0) * self.units.dt
        field += inc
        self.budgets[name].sources += float(inc.sum() * self.cell_area)
        spec = self.specs[name]
        if spec.nonnegative:
            neg = field < 0.0
            if neg.any():
                self.budgets[name].clipped += float(-field[neg].sum() * self.cell_area)
                field[neg] = 0.0

    def record_conversion(self, name: str, delta: np.ndarray) -> None:
        """Account an in-place change `delta` (field units) made by the caller, e.g. phase change."""
        self.budgets[name].conversion += float(np.where(self.fluid, delta, 0.0).sum() * self.cell_area)
