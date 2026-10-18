"""D2Q9 lattice-Boltzmann flow solver with Guo forcing and Smagorinsky BGK.

Solves weakly compressible (low-Mach) flow in a vertical x-z section, which
approximates the incompressible Navier-Stokes equations with a body force:
the Boussinesq buoyancy computed by the caller. One step is

    moments -> BGK collision (local tau from the Smagorinsky model, Guo
    forcing source) -> sponge relaxation -> pull streaming with boundaries.

Collision: single-relaxation-time BGK in its regularised form (Latt &
Chopard 2006, the default): before relaxing, the non-equilibrium part is
projected onto its hydrodynamic (second-order Hermite) component, which
removes the non-hydrodynamic "ghost" modes that make plain BGK blow up at
the low viscosities (tau -> 0.5) this coarse grid needs for unsteady,
turbulent-looking plumes. Plain BGK (`collision="bgk"`) is kept for tests.
Grid Reynolds number, not dt, governs BGK stability (u dt/dx over
nu dt/dx^2 is independent of dt), so halving dt only helps the Mach limit.

Boundaries (see docs/design.md section 1):
* bottom: no-slip wall at z = 0 by half-way bounce-back; `solid` cells
  (terrain) bounce back the same way;
* left/right: "open" (inflow/outflow from the ambient wind, either direction),
  "periodic" or "wall";
* top: "open" (buoyant plumes leave freely) or "wall".

At an open boundary the populations that would stream in from outside the
domain are set to an equilibrium:
* lateral inflow (the ambient wind enters through this side, |u_amb| above a
  0.5 m/s deadband): feq(rho of the boundary node, u_amb) — velocity inlet;
* lateral outflow, or calm ambient air (|u_amb| below the deadband):
  feq(1, u of the boundary node) — pressure outlet, which lets plumes
  entrain air through the sides in calm conditions;
* top: feq(1, u of the boundary node) — pressure outlet, so rising plumes
  leave freely (their vertical velocity is never damped).
The boundary node's density and velocity are its complete pre-collision
moments of the current step. A sponge layer next to the open sides relaxes
the populations towards the ambient equilibrium at a rate given in 1/s (so
it does not change when dt is halved); at the top only density and the
horizontal velocity are relaxed, never w.

Where this fits: the flow half of the coupled solver; `simulation.py` calls
`set_ambient` and `step` once per time step and reads `velocity()`.
Calls into: numpy and numba only (single-threaded kernels).
Start reading at: LatticeUnits, then LBMSolver.step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numba
import numpy as np

CS2 = 1.0 / 3.0

# Direction i: (ex, ez). z (row index) points UP.
E = np.array(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
    dtype=np.int64,
)
W = np.array([4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4, dtype=np.float64)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int64)
# Contiguous float copies of the lattice vectors for the kernels.
EXF = np.ascontiguousarray(E[:, 0], dtype=np.float64)
EZF = np.ascontiguousarray(E[:, 1], dtype=np.float64)
EXI = np.ascontiguousarray(E[:, 0])
EZI = np.ascontiguousarray(E[:, 1])

_LATERAL_MODES = {"open": 0, "periodic": 1, "wall": 2}
_TOP_MODES = {"open": 0, "wall": 2}


@dataclass(frozen=True)
class LatticeUnits:
    """Conversion between physical (m, s) and lattice units for spacing dx and step dt."""

    dx: float
    dt: float

    def velocity_to_lattice(self, u):
        return u * (self.dt / self.dx)

    def velocity_to_physical(self, u_lat):
        return u_lat * (self.dx / self.dt)

    def viscosity_to_lattice(self, nu):
        return nu * self.dt / (self.dx * self.dx)

    def acceleration_to_lattice(self, a):
        return a * (self.dt * self.dt / self.dx)

    def tau(self, nu: float) -> float:
        """BGK relaxation time for kinematic viscosity `nu` (m^2/s)."""
        return 0.5 + float(self.viscosity_to_lattice(nu)) / CS2


@dataclass
class LBMConfig:
    nx: int
    nz: int
    units: LatticeUnits
    nu: float                     # base kinematic (eddy) viscosity, m^2/s
    smagorinsky_cs: float = 0.12  # 0 disables the subgrid model
    sponge_width: int = 6         # cells next to each open boundary
    sponge_rate: float = 0.8      # relaxation rate at the boundary itself, 1/s
    calm_deadband: float = 0.5    # m/s; lateral sides are pressure outlets below this ambient speed
    lateral: str = "open"         # "open" | "periodic" | "wall"
    top: str = "open"             # "open" | "wall"
    collision: str = "regularized"  # "regularized" | "bgk"


@numba.njit(cache=True, inline="always")
def _feq(i, rho, ux, uz, ex, ez, w):
    eu = ex[i] * ux + ez[i] * uz
    uu = ux * ux + uz * uz
    return w[i] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)


@numba.njit(cache=True, fastmath=True)
def _collide(f, rho, ux, uz, ax, az, solid, tau0, cs_smag, sponge, u_amb_lat, top_sponge_rows,
             regularized, ex, ez, w, f_post):
    """Moments + BGK collision (Smagorinsky tau, Guo forcing, sponge) into f_post.

    Writes the pre-collision density and half-force-corrected velocity of
    every cell to rho, ux, uz (the open boundaries of `_stream` read them).
    """
    nz, nx, nq = f.shape
    feq = np.empty(9)
    smag = 18.0 * math.sqrt(2.0) * cs_smag * cs_smag
    for k in range(nz):
        for i in range(nx):
            if solid[k, i]:
                rho[k, i] = 1.0
                ux[k, i] = 0.0
                uz[k, i] = 0.0
                for q in range(nq):
                    f_post[k, i, q] = f[k, i, q]
                continue
            r = 0.0
            mx = 0.0
            mz = 0.0
            for q in range(nq):
                fq = f[k, i, q]
                r += fq
                mx += fq * ex[q]
                mz += fq * ez[q]
            u = mx / r + 0.5 * ax[k, i]
            v = mz / r + 0.5 * az[k, i]
            rho[k, i] = r
            ux[k, i] = u
            uz[k, i] = v
            fx = r * ax[k, i]
            fz = r * az[k, i]
            # Non-equilibrium momentum flux (with the Guo half-force correction).
            pxx = 0.0
            pxz = 0.0
            pzz = 0.0
            for q in range(nq):
                feq[q] = _feq(q, r, u, v, ex, ez, w)
                d = f[k, i, q] - feq[q]
                pxx += ex[q] * ex[q] * d
                pxz += ex[q] * ez[q] * d
                pzz += ez[q] * ez[q] * d
            pxx += u * fx
            pzz += v * fz
            pxz += 0.5 * (u * fz + v * fx)
            tau = tau0
            if smag > 0.0:
                qn = math.sqrt(pxx * pxx + 2.0 * pxz * pxz + pzz * pzz)
                tau = 0.5 * (tau0 + math.sqrt(tau0 * tau0 + smag * qn / r))
            omega = 1.0 / tau
            pref = 1.0 - 0.5 * omega
            for q in range(nq):
                eu = ex[q] * u + ez[q] * v
                # Guo source: w_q [ (e - u)/cs2 + (e.u) e / cs4 ] . F
                sx = 3.0 * (ex[q] - u) + 9.0 * eu * ex[q]
                sz = 3.0 * (ez[q] - v) + 9.0 * eu * ez[q]
                src = w[q] * (sx * fx + sz * fz)
                if regularized:
                    # Hermite projection: f_neq = w / (2 cs^4) Q : Pi_neq, Q = e e - cs^2 I.
                    qxx = ex[q] * ex[q] - 1.0 / 3.0
                    qzz = ez[q] * ez[q] - 1.0 / 3.0
                    qxz = ex[q] * ez[q]
                    # Guo's velocity carries half the force, so the non-equilibrium part
                    # also has a first moment -F/2: keep its first-order Hermite term.
                    fneq = (4.5 * w[q] * (qxx * pxx + 2.0 * qxz * pxz + qzz * pzz)
                            - 1.5 * w[q] * (ex[q] * fx + ez[q] * fz))
                    f_post[k, i, q] = feq[q] + (1.0 - omega) * fneq + pref * src
                else:
                    f_post[k, i, q] = f[k, i, q] - omega * (f[k, i, q] - feq[q]) + pref * src
            s = sponge[k, i]
            if s > 0.0:
                ut = u_amb_lat[k]
                vt = 0.0
                if k >= nz - top_sponge_rows:
                    # Top sponge: keep the vertical velocity so plumes leave through the top.
                    vt = v
                for q in range(nq):
                    target = _feq(q, 1.0, ut, vt, ex, ez, w)
                    f_post[k, i, q] -= s * (f_post[k, i, q] - target)


@numba.njit(cache=True)
def _stream(f_post, rho, ux, uz, solid, u_amb_lat, deadband, lateral, top, ex, ez, w, opp, f_new,
            rho_new, jx_new, jz_new):
    """Pull streaming with bounce-back walls and open/periodic boundaries.

    Also accumulates the density and momentum of the streamed populations
    (rho_new, jx_new, jz_new), so no separate moments pass is needed.
    """
    nz, nx, nq = f_post.shape
    for k in range(nz):
        for i in range(nx):
            if solid[k, i]:
                for q in range(nq):
                    f_new[k, i, q] = f_post[k, i, q]
                rho_new[k, i] = 1.0
                jx_new[k, i] = 0.0
                jz_new[k, i] = 0.0
                continue
            for q in range(nq):
                ks = k - ez[q]
                isrc = i - ex[q]
                # 1. Floor and terrain: half-way bounce-back.
                if ks < 0:
                    f_new[k, i, q] = f_post[k, i, opp[q]]
                    continue
                if 0 <= ks < nz and 0 <= isrc < nx and solid[ks, isrc]:
                    f_new[k, i, q] = f_post[k, i, opp[q]]
                    continue
                # 2. Lateral sides.
                out_lateral = isrc < 0 or isrc >= nx
                if out_lateral:
                    if lateral == 1:
                        isrc = isrc % nx
                        out_lateral = False
                    elif lateral == 2:
                        f_new[k, i, q] = f_post[k, i, opp[q]]
                        continue
                out_top = ks >= nz
                if out_top and top == 2:
                    f_new[k, i, q] = f_post[k, i, opp[q]]
                    continue
                if out_lateral or out_top:
                    # 3. Open boundary (see the module docstring): velocity inlet where the
                    #    ambient wind enters, pressure outlet everywhere else. The outlet
                    #    velocity is extrapolated from the next node inward (using the
                    #    boundary node's own velocity fed back on itself and blew up at
                    #    outflow corners); flow back in through the top outlet is cut.
                    kn = k - 1 if (out_top and k > 0) else k
                    inn = i
                    if out_lateral:
                        inn = i + 1 if isrc < 0 else i - 1
                        if inn < 0 or inn >= nx:
                            inn = i
                    if solid[kn, inn]:
                        kn = k
                        inn = i
                    rb = 1.0
                    ub = ux[kn, inn]
                    vb = uz[kn, inn]
                    if out_top and vb < 0.0:
                        vb = 0.0
                    if out_lateral and not out_top:
                        ua = u_amb_lat[k]
                        entering = (isrc < 0 and ua > deadband) or (isrc >= nx and ua < -deadband)
                        if entering:
                            rb = rho[k, i]
                            ub = ua
                            vb = 0.0
                    f_new[k, i, q] = _feq(q, rb, ub, vb, ex, ez, w)
                    continue
                f_new[k, i, q] = f_post[ks, isrc, q]
            r = 0.0
            mx = 0.0
            mz = 0.0
            for q in range(nq):
                fq = f_new[k, i, q]
                r += fq
                mx += fq * ex[q]
                mz += fq * ez[q]
            rho_new[k, i] = r
            jx_new[k, i] = mx
            jz_new[k, i] = mz


class LBMSolver:
    """D2Q9 BGK solver on an (nz, nx) grid; see the module docstring."""

    def __init__(self, cfg: LBMConfig, solid: Optional[np.ndarray] = None,
                 u_ambient: Optional[np.ndarray] = None):
        if cfg.lateral not in _LATERAL_MODES:
            raise ValueError(f"lateral must be one of {sorted(_LATERAL_MODES)}, got {cfg.lateral!r}")
        if cfg.top not in _TOP_MODES:
            raise ValueError(f"top must be one of {sorted(_TOP_MODES)}, got {cfg.top!r}")
        if cfg.collision not in ("regularized", "bgk"):
            raise ValueError(f"collision must be 'regularized' or 'bgk', got {cfg.collision!r}")
        self.cfg = cfg
        nz, nx = cfg.nz, cfg.nx
        self.tau0 = cfg.units.tau(cfg.nu)
        if not self.tau0 > 0.5:
            raise ValueError(f"tau = {self.tau0} must exceed 0.5 (nu = {cfg.nu} m^2/s)")
        if solid is None:
            solid = np.zeros((nz, nx), dtype=bool)
        solid = np.ascontiguousarray(solid, dtype=np.bool_)
        if solid.shape != (nz, nx):
            raise ValueError(f"solid has shape {solid.shape}, expected {(nz, nx)}")
        self.solid = solid
        self.fluid = ~solid
        self._lateral = _LATERAL_MODES[cfg.lateral]
        self._top = _TOP_MODES[cfg.top]
        self._u_amb_lat = np.zeros(nz)
        if u_ambient is not None:
            self.set_ambient(u_ambient)
        self._sponge, self._top_rows = self._build_sponge()
        self.rho = np.ones((nz, nx))
        self.ux_lat = np.where(self.fluid, self._u_amb_lat[:, None], 0.0)
        self.uz_lat = np.zeros((nz, nx))
        # Populations are stored cell-contiguous, f[k, i, q]: the kernels loop over the
        # nine directions of one cell innermost.
        self.f = np.empty((nz, nx, 9))
        for q in range(9):
            eu = E[q, 0] * self.ux_lat + E[q, 1] * self.uz_lat
            uu = self.ux_lat ** 2 + self.uz_lat ** 2
            self.f[:, :, q] = W[q] * self.rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * uu)
        self._f_post = np.empty_like(self.f)
        self._f_new = np.empty_like(self.f)
        self._rho_new = np.ones((nz, nx))
        self._jx = np.zeros((nz, nx))
        self._jz = np.zeros((nz, nx))
        self._zero = np.zeros((nz, nx))
        self.steps = 0

    def _build_sponge(self) -> Tuple[np.ndarray, int]:
        cfg = self.cfg
        nz, nx = cfg.nz, cfg.nx
        width = int(cfg.sponge_width)
        sponge = np.zeros((nz, nx))
        if width <= 0 or cfg.sponge_rate <= 0.0:
            return sponge, 0
        per_step = min(0.5, cfg.sponge_rate * cfg.units.dt)
        ramp = per_step * ((width - np.arange(width)) / width) ** 2
        top_rows = 0
        if cfg.lateral == "open":
            for d in range(min(width, nx)):
                sponge[:, d] = np.maximum(sponge[:, d], ramp[d])
                sponge[:, nx - 1 - d] = np.maximum(sponge[:, nx - 1 - d], ramp[d])
        if cfg.top == "open":
            top_rows = min(width, nz)
            for d in range(top_rows):
                sponge[nz - 1 - d, :] = np.maximum(sponge[nz - 1 - d, :], ramp[d])
        sponge[self.solid] = 0.0
        return sponge, top_rows

    def set_ambient(self, u_ambient: np.ndarray) -> None:
        """Ambient horizontal wind profile (nz,), m/s; used at inflow and in the sponges."""
        u = np.asarray(u_ambient, dtype=np.float64)
        if u.shape != (self.cfg.nz,):
            raise ValueError(f"u_ambient has shape {u.shape}, expected ({self.cfg.nz},)")
        self._u_amb_lat[:] = self.cfg.units.velocity_to_lattice(u)

    def step(self, ax: Optional[np.ndarray] = None, az: Optional[np.ndarray] = None) -> None:
        """Advance one time step with body acceleration (ax, az) in m/s^2 (None = 0)."""
        units = self.cfg.units
        axl = self._zero if ax is None else np.ascontiguousarray(units.acceleration_to_lattice(ax), dtype=np.float64)
        azl = self._zero if az is None else np.ascontiguousarray(units.acceleration_to_lattice(az), dtype=np.float64)
        _collide(self.f, self.rho, self.ux_lat, self.uz_lat, axl, azl, self.solid, self.tau0,
                 float(self.cfg.smagorinsky_cs), self._sponge, self._u_amb_lat, self._top_rows,
                 self.cfg.collision == "regularized", EXF, EZF, W, self._f_post)
        deadband = float(self.cfg.units.velocity_to_lattice(self.cfg.calm_deadband))
        _stream(self._f_post, self.rho, self.ux_lat, self.uz_lat, self.solid, self._u_amb_lat,
                deadband, self._lateral, self._top, EXI, EZI, W, OPP, self._f_new,
                self._rho_new, self._jx, self._jz)
        self.f, self._f_new = self._f_new, self.f
        # Velocity reported for the new state, half-force corrected with this step's force
        # (the next collision recomputes it with the next force).
        self.rho, self._rho_new = self._rho_new, self.rho
        np.divide(self._jx, self.rho, out=self.ux_lat)
        np.divide(self._jz, self.rho, out=self.uz_lat)
        self.ux_lat += 0.5 * axl
        self.uz_lat += 0.5 * azl
        self.ux_lat[self.solid] = 0.0
        self.uz_lat[self.solid] = 0.0
        self.steps += 1

    def velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Physical velocity (ux, uz) in m/s at cell centres; zero inside solids."""
        units = self.cfg.units
        return units.velocity_to_physical(self.ux_lat), units.velocity_to_physical(self.uz_lat)

    def max_lattice_speed(self) -> float:
        speed = np.sqrt(self.ux_lat ** 2 + self.uz_lat ** 2)
        return float(speed[self.fluid].max()) if self.fluid.any() else 0.0

    def max_mach(self) -> float:
        return self.max_lattice_speed() / math.sqrt(CS2)

    def total_mass(self) -> float:
        return float(self.rho[self.fluid].sum())

    def max_density_deviation(self, region: Optional[Tuple[slice, slice]] = None) -> float:
        """max |rho - 1| over fluid cells, optionally only inside `region` (rows, cols)."""
        dev = np.abs(self.rho - 1.0)
        fluid = self.fluid
        if region is not None:
            dev = dev[region]
            fluid = fluid[region]
        return float(dev[fluid].max()) if fluid.any() else 0.0

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.f).all())


def ambient_maintaining_acceleration(u_amb: np.ndarray, nu: float, dx: float,
                                     smagorinsky_cs: float = 0.0) -> np.ndarray:
    """Body acceleration (nz,) in m/s^2 that makes the ambient profile u_amb(z) a steady state.

    Over a no-slip floor with an eddy viscosity, a power-law wind profile is not
    a solution: a boundary layer would grow from the inflow side and slow the
    near-ground air by tens of per cent across the view. Real near-ground wind
    is maintained by unresolved turbulent momentum transport; this force plays
    that role: a = -d/dz(nu_eff du/dz), discretised like the LBM sees it
    (half-way bounce-back ghost u(-1) = -u(0) at the floor, zero gradient at
    the top) with nu_eff = nu + (Cs dx)^2 |du/dz| when the Smagorinsky model is on.
    Perturbations (plumes, fog) still feel the no-slip wall.
    """
    u = np.asarray(u_amb, dtype=np.float64)
    ext = np.concatenate(([-u[0]], u, [u[-1]]))
    shear = np.diff(ext) / dx                       # (nz+1,) at faces -1/2 .. nz-1/2
    nu_face = nu + (smagorinsky_cs * dx) ** 2 * np.abs(shear)
    flux = nu_face * shear                          # nu du/dz at faces
    return -(flux[1:] - flux[:-1]) / dx
