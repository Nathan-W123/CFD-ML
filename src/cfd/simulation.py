"""The coupled flow + scalar + moisture time loop for one scenario.

Prognostic variables (all on the LBM grid):

* LBM populations (velocity, weakly compressible pressure);
* theta_l' = theta_l - theta_l_ref(z): liquid-water potential temperature
  anomaly, theta_l = theta - L_v / (c_p Exner) q_l;
* q_t' = q_t - q_t_ref(z): total-water anomaly, q_t = q_v + q_l;
* smoke and tracer concentrations (mg/m^3);
* four passive texture coordinates (two pairs, see results.py).

theta_l and q_t are conserved by condensation and evaporation, so phase
change needs no bookkeeping and water is conserved by transport alone. The
temperature, vapour and liquid water are diagnosed every step by the
saturation adjustment of thermo.py starting from (theta_l, q_t, q_l = 0).
Transporting ANOMALIES about the z-only ambient profiles (with the
background-advection terms -w d theta_l_ref/dz and -w d q_t_ref/dz) keeps the
ambient state exactly steady: diffusing the full stratified profiles with
no-flux walls would warm the ground by ~1 K in two minutes and drive spurious
convection in every scene.

Each step:

1. diagnose theta, q_v, q_l; buoyancy relative to the reference profiles,
   b = g [(theta - theta_ref)/theta0 + 0.608 (q_v - q_v_ref) - (q_l - q_l_ref)];
2. one LBM step with body force = buoyancy + gust acceleration (dU/dt of the
   ambient wind, applied everywhere: imposing gusts only through the open
   boundaries launched pressure waves that blew the solver up) + the
   ambient-maintaining force (keeps the wind profile steady over the no-slip
   floor). A domain-wide stochastic turbulence forcing was tried and removed:
   in 2D the injected energy cascades to large scales instead of dissipating,
   and modes drifting slower than the fluid resonate (20-70 m/s blow-ups).
   Unsteadiness comes from gusts, the low base viscosity, source puffing and
   plume instabilities, and sub-grid texture from the advected coordinates;
3. transport of every scalar; 4. sources (emission, heating, surface fluxes,
   buoyancy relaxation in the top sponge); 5. checks; snapshots are stored at
   each observation time and at its lag partner (t - flow_lag_s).

Before t = 0 a source-free precursor spins up the ambient flow (no impulsive
start in the observed window); the scenario's initial fields are applied at
t = 0, and the first observation is at spinup_s.

A run whose lattice speed or in-view density deviation leaves the low-Mach
regime raises `SimulationUnstable`; `scenarios.run_scenario` retries it with
a halved time step.

Where this fits: `scenarios.build_spec` builds a `SimulationSpec`;
`run_simulation` turns it into a `SimulationResult` that the renderer reads.
Calls into: lbm, scalars, thermo, validation.
Start reading at: run_simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import numpy as np

from src.cfd import thermo
from src.cfd.lbm import LatticeUnits, LBMConfig, LBMSolver, ambient_maintaining_acceleration
from src.cfd.results import SimulationResult
from src.cfd.scalars import ScalarSpec, ScalarTransport
from src.cfd.validation import summarize_validity, view_slices

TRANSPORTED = ("thetal", "qt", "smoke", "tracer", "tex_xa", "tex_za", "tex_xb", "tex_zb")
TEX_NAMES = ("tex_xa", "tex_za", "tex_xb", "tex_zb")
CHECK_EVERY = 10


class SimulationUnstable(RuntimeError):
    """The flow left the stable low-Mach regime (or produced non-finite values)."""


@dataclass
class SimulationSpec:
    sim_id: str
    category: str
    regime: str
    labels: Tuple[int, int]
    seed: int
    params: Dict[str, Any]
    nx: int
    nz: int
    dx: float
    dt: float
    nu: float
    diffusivity: float
    smagorinsky_cs: float
    sponge_width: int
    sponge_rate: float
    solid: np.ndarray
    theta_ref: np.ndarray                 # (nz,) K, diagnosed consistently from thetal_ref, qt_ref
    qv_ref: np.ndarray                    # (nz,) kg/kg
    ql_ref: np.ndarray                    # (nz,) kg/kg (non-zero only for an ambient fog layer)
    initial: Dict[str, np.ndarray]        # ABSOLUTE theta, qv, ql, smoke, tracer (nz, nx) at t = 0
    wind: Callable[[float], np.ndarray]   # t -> (nz,) mean ambient u, m/s (gusts included)
    inflow: Callable[[float], Dict[str, Dict[str, np.ndarray]]]   # ABSOLUTE ghost values
    sources: Callable[[float, Mapping[str, np.ndarray]], Dict[str, np.ndarray]]
    tracer_settling: float
    spinup_s: float
    frame_times: np.ndarray               # (T,) s after spin-up
    flow_lag_s: float = 1.0
    precursor_s: float = 30.0
    tex_period_s: float = 30.0
    top_sponge_theta_rate: float = 0.5    # 1/s, buoyancy relaxation in the top sponge
    theta0: float = 288.0
    p_surface: float = 101325.0
    max_lattice_speed: float = 0.14
    max_view_density_deviation: float = 0.05
    view_margin: int = 6
    dt_halvings: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def units(self) -> LatticeUnits:
        return LatticeUnits(self.dx, self.dt)

    @property
    def z(self) -> np.ndarray:
        return (np.arange(self.nz) + 0.5) * self.dx

    @property
    def x(self) -> np.ndarray:
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def end_time(self) -> float:
        return self.spinup_s + float(self.frame_times[-1])


# ---------------------------------------------------------------------- physics helpers
def buoyancy(theta, qv, ql, spec: SimulationSpec, g: float = thermo.G) -> np.ndarray:
    """Boussinesq buoyancy (m/s^2, +up) relative to the ambient reference profiles."""
    return g * ((theta - spec.theta_ref[:, None]) / spec.theta0
                + 0.608 * (qv - spec.qv_ref[:, None])
                - (ql - spec.ql_ref[:, None]))


def diagnose(thetal: np.ndarray, qt: np.ndarray, p: np.ndarray, pi: np.ndarray):
    """(theta, q_v, q_l) in moist equilibrium for absolute theta_l and q_t."""
    theta = np.array(thetal, dtype=np.float64, copy=True)
    qv = np.maximum(np.array(qt, dtype=np.float64, copy=True), 0.0)
    ql = np.zeros_like(theta)
    thermo.saturation_adjustment(theta, qv, ql, p, pi)
    return theta, qv, ql


def consistent_reference(theta_ref, qv_ref, ql_ref, p, pi):
    """Reference profiles that are an exact fixed point of the diagnosis."""
    lat = thermo.L_V / thermo.CP / pi
    thetal_ref = theta_ref - lat * ql_ref
    qt_ref = qv_ref + ql_ref
    th, qv, ql = diagnose(thetal_ref[:, None], qt_ref[:, None], p, pi)
    return thetal_ref, qt_ref, th[:, 0], qv[:, 0], ql[:, 0]


def _absolute_to_conserved(theta, qv, ql, lat):
    return theta - lat * ql, qv + ql


# ---------------------------------------------------------------------- the run
def run_simulation(spec: SimulationSpec, cfg: Optional[Mapping[str, Any]] = None) -> SimulationResult:
    """Run one simulation; raises SimulationUnstable if the flow goes unstable."""
    cfg = cfg or {}
    units = spec.units
    nz, nx = spec.nz, spec.nx
    dt = spec.dt
    solid = np.asarray(spec.solid, dtype=bool)
    fluid = ~solid
    z, x = spec.z, spec.x
    zz, xx = np.meshgrid(z, x, indexing="ij")
    p = thermo.pressure_profile(z, spec.p_surface, spec.theta0)
    pi = thermo.exner(p)
    lat = thermo.L_V / thermo.CP / pi                     # (nz,)
    thetal_ref, qt_ref, theta_ref, qv_ref, ql_ref = consistent_reference(
        spec.theta_ref, spec.qv_ref, spec.ql_ref, p, pi)
    spec.theta_ref, spec.qv_ref, spec.ql_ref = theta_ref, qv_ref, ql_ref
    dthl_dz = np.gradient(thetal_ref, spec.dx)
    dqt_dz = np.gradient(qt_ref, spec.dx)
    rows, cols = view_slices(nz, nx, spec.view_margin)

    lbm = LBMSolver(
        LBMConfig(nx=nx, nz=nz, units=units, nu=spec.nu, smagorinsky_cs=spec.smagorinsky_cs,
                  sponge_width=spec.sponge_width, sponge_rate=spec.sponge_rate),
        solid=solid, u_ambient=spec.wind(-spec.precursor_s),
    )
    specs = [
        ScalarSpec("thetal", spec.diffusivity, nonnegative=False),
        ScalarSpec("qt", spec.diffusivity, nonnegative=False),
        ScalarSpec("smoke", spec.diffusivity),
        ScalarSpec("tracer", spec.diffusivity, settling_velocity=spec.tracer_settling),
    ] + [ScalarSpec(name, 0.0, nonnegative=False) for name in TEX_NAMES]
    transport = ScalarTransport(nx, nz, units, solid, specs,
                                scheme=str(cfg.get("scalars", {}).get("scheme", "muscl")))
    # Top sponge rows: buoyancy (theta_l') is relaxed there instead of momentum.
    top_relax = np.zeros((nz, nx))
    w = max(1, spec.sponge_width)
    for d in range(min(w, nz)):
        top_relax[nz - 1 - d, :] = spec.top_sponge_theta_rate * ((w - d) / w) ** 2
    top_relax[solid] = 0.0

    def body_force(t: float, b: Optional[np.ndarray]):
        u_now = spec.wind(t)
        u_prev = spec.wind(t - dt)
        a_x = (u_now - u_prev) / dt + ambient_maintaining_acceleration(u_now, spec.nu, spec.dx, spec.smagorinsky_cs)
        ax = np.repeat(a_x[:, None], nx, axis=1)
        az = np.zeros((nz, nx)) if b is None else b.copy()
        ax[solid] = 0.0
        az[solid] = 0.0
        return u_now, ax, az

    # ---- precursor: ambient flow only ------------------------------------------
    n_pre = int(round(spec.precursor_s / dt))
    for step in range(n_pre):
        t = -spec.precursor_s + (step + 1) * dt
        u_now, ax, az = body_force(t, None)
        lbm.set_ambient(u_now)
        lbm.step(ax, az)
        speed = lbm.max_lattice_speed()
        if not np.isfinite(speed) or speed > spec.max_lattice_speed:
            raise SimulationUnstable(f"precursor lattice speed {speed:.3f} at t = {t:.1f} s (dt = {dt})")

    # ---- initial scalars ---------------------------------------------------------
    init = spec.initial
    thl0, qt0 = _absolute_to_conserved(init["theta"], init["qv"], init["ql"], lat[:, None])
    fields = {
        "thetal": np.where(fluid, thl0 - thetal_ref[:, None], 0.0),
        "qt": np.where(fluid, qt0 - qt_ref[:, None], 0.0),
        "smoke": np.where(fluid, init["smoke"], 0.0).astype(np.float64),
        "tracer": np.where(fluid, init["tracer"], 0.0).astype(np.float64),
    }
    tex_reset = {"a": 0.0, "b": -0.5 * spec.tex_period_s}
    fields["tex_xa"], fields["tex_za"] = xx.copy(), zz.copy()
    fields["tex_xb"], fields["tex_zb"] = xx.copy(), zz.copy()
    transport.start_budget(fields)

    def tex_age(pair: str, t: float) -> float:
        return (t - tex_reset[pair]) % spec.tex_period_s

    def tex_weights(t: float) -> Tuple[float, float]:
        wa = math.sin(math.pi * tex_age("a", t) / spec.tex_period_s) ** 2
        return wa, 1.0 - wa

    def ghosts(t: float, u_amb: np.ndarray):
        g = spec.inflow(t)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        th, qv, ql = g["theta"], g["qv"], g["ql"]
        top_lat = lat[-1]
        out["thetal"] = {
            "left": th["left"] - lat * ql["left"] - thetal_ref,
            "right": th["right"] - lat * ql["right"] - thetal_ref,
            "top": th["top"] - top_lat * ql["top"] - thetal_ref[-1],
        }
        out["qt"] = {
            "left": qv["left"] + ql["left"] - qt_ref,
            "right": qv["right"] + ql["right"] - qt_ref,
            "top": qv["top"] + ql["top"] - qt_ref[-1],
        }
        out["smoke"] = g["smoke"]
        out["tracer"] = g["tracer"]
        width = nx * spec.dx
        height = nz * spec.dx
        for pair in ("a", "b"):
            age = tex_age(pair, t)
            out[f"tex_x{pair}"] = {"left": -u_amb * age, "right": width - u_amb * age, "top": x.copy()}
            out[f"tex_z{pair}"] = {"left": z.copy(), "right": z.copy(), "top": np.full(nx, height)}
        return out

    # ---- output bookkeeping ------------------------------------------------------------
    T = len(spec.frame_times)
    obs_t = spec.spinup_s + np.asarray(spec.frame_times, dtype=float)
    lag_t = obs_t - spec.flow_lag_s
    obs_steps = np.rint(obs_t / dt).astype(int)
    lag_steps = np.rint(lag_t / dt).astype(int)
    if lag_steps.min() < 0:
        raise ValueError("spin-up shorter than the flow lag")
    snap_names = ("smoke", "lwc", "tracer")
    obs = {name: np.zeros((T, nz, nx), dtype=np.float32) for name in snap_names}
    lag = {name: np.zeros((T, nz, nx), dtype=np.float32) for name in snap_names}
    obs_tex = np.zeros((T, 4, nz, nx), dtype=np.float32)
    lag_tex = np.zeros((T, 4, nz, nx), dtype=np.float32)
    obs_w = np.zeros((T, 2), dtype=np.float32)
    lag_w = np.zeros((T, 2), dtype=np.float32)
    diag = {name: np.zeros((T, nz, nx), dtype=np.float32) for name in ("theta_anom", "qv", "ux", "uz")}
    rho_air = float(cfg.get("physics", {}).get("rho_air", 1.2))
    by_step: Dict[int, list] = {}
    for k, s in enumerate(obs_steps):
        by_step.setdefault(int(s), []).append(("obs", k))
    for k, s in enumerate(lag_steps):
        by_step.setdefault(int(s), []).append(("lag", k))

    def store(t: float, theta, qv, ql):
        for kind, k in by_step.get(step_now[0], []):
            tgt, tex, wts = (obs, obs_tex, obs_w) if kind == "obs" else (lag, lag_tex, lag_w)
            tgt["smoke"][k] = fields["smoke"]
            tgt["tracer"][k] = fields["tracer"]
            tgt["lwc"][k] = thermo.lwc_from_ql(ql, rho_air)
            for j, name in enumerate(TEX_NAMES):
                tex[k, j] = fields[name]
            wts[k] = tex_weights(t)
            if kind == "obs":
                diag["theta_anom"][k] = theta - theta_ref[:, None]
                diag["qv"][k] = qv
                ux, uz = lbm.velocity()
                diag["ux"][k] = ux
                diag["uz"][k] = uz

    n_steps = int(round(spec.end_time / dt))
    max_speed = 0.0
    max_rho_dev = 0.0
    max_theta_anom = 0.0
    step_now = [0]
    th_abs = thetal_ref[:, None] + fields["thetal"]
    theta, qv, ql = diagnose(th_abs, qt_ref[:, None] + fields["qt"], p, pi)
    store(0.0, theta, qv, ql)

    for step in range(1, n_steps + 1):
        t = step * dt
        step_now[0] = step
        b = np.where(fluid, buoyancy(theta, qv, ql, spec), 0.0)
        u_now, ax, az = body_force(t, b)
        lbm.set_ambient(u_now)
        lbm.step(ax, az)
        speed = lbm.max_lattice_speed()
        if not np.isfinite(speed) or speed > spec.max_lattice_speed:
            raise SimulationUnstable(
                f"lattice speed {speed:.3f} > {spec.max_lattice_speed} at t = {t:.1f} s (dt = {dt})")
        ux, uz = lbm.velocity()
        # Texture coordinate pairs reset to the identity on their cycle.
        for pair in ("a", "b"):
            if tex_age(pair, t) < dt * 0.5 or tex_age(pair, t) > spec.tex_period_s - dt * 0.5:
                if t - tex_reset[pair] >= spec.tex_period_s - dt * 0.5:
                    tex_reset[pair] = t
                    fields[f"tex_x{pair}"][...] = xx
                    fields[f"tex_z{pair}"][...] = zz
        try:
            transport.step(fields, ux, uz, ghosts(t, u_now))
        except ValueError as exc:   # advective CFL limit
            raise SimulationUnstable(f"{exc} at t = {t:.1f} s (dt = {dt})") from exc
        rates = dict(spec.sources(t, {"theta": theta, "qv": qv, "ql": ql,
                                       "theta_anom": theta - theta_ref[:, None],
                                       "smoke": fields["smoke"], "tracer": fields["tracer"]}))
        # theta and q_v sources act on the conserved variables.
        rates["thetal"] = rates.pop("theta", 0.0) - uz * dthl_dz[:, None] - top_relax * fields["thetal"]
        rates["qt"] = rates.pop("qv", 0.0) - uz * dqt_dz[:, None]
        for name, rate in rates.items():
            if np.ndim(rate) == 0 and rate == 0.0:
                continue
            transport.add_source(name, fields[name], np.broadcast_to(rate, (nz, nx)))
        theta, qv, ql = diagnose(thetal_ref[:, None] + fields["thetal"], qt_ref[:, None] + fields["qt"], p, pi)
        if step % CHECK_EVERY == 0 or step == n_steps:
            speed = lbm.max_lattice_speed()
            rho_dev = lbm.max_density_deviation((rows, cols))
            max_speed = max(max_speed, speed)
            max_rho_dev = max(max_rho_dev, rho_dev)
            th_anom = float(np.abs(theta - theta_ref[:, None])[fluid].max())
            max_theta_anom = max(max_theta_anom, th_anom)
            if not np.isfinite(speed) or speed > spec.max_lattice_speed:
                raise SimulationUnstable(
                    f"lattice speed {speed:.3f} > {spec.max_lattice_speed} at t = {t:.1f} s (dt = {dt})")
            if rho_dev > spec.max_view_density_deviation:
                raise SimulationUnstable(
                    f"in-view density deviation {rho_dev:.3f} > {spec.max_view_density_deviation} at t = {t:.1f} s (dt = {dt})")
            if not lbm.is_finite() or not all(np.isfinite(f).all() for f in fields.values()):
                raise SimulationUnstable(f"non-finite state at t = {t:.1f} s")
        if step in by_step:
            store(t, theta, qv, ql)

    validity = summarize_validity(
        fields={"thetal": fields["thetal"], "qt": fields["qt"], "smoke": fields["smoke"],
                "tracer": fields["tracer"], "ql": ql, "qv": qv},
        transport=transport, solid=solid, max_lattice_speed=max_speed,
        max_density_deviation=max_rho_dev, max_theta_anomaly=max_theta_anom, spec=spec,
        smoke_frames=obs["smoke"], lwc_frames=obs["lwc"], tracer_frames=obs["tracer"],
        uz_frames=diag["uz"], cfg=cfg,
    )
    times = np.asarray(spec.frame_times, dtype=np.float64)
    return SimulationResult(
        sim_id=spec.sim_id, category=spec.category, regime=spec.regime, labels=tuple(spec.labels),
        seed=int(spec.seed), params=spec.params, dx=float(spec.dx), times=times,
        smoke=obs["smoke"], lwc=obs["lwc"], tracer=obs["tracer"], solid=solid.copy(),
        valid=bool(validity["valid"]), validity=validity,
        texcoords=obs_tex, tex_weights=obs_w, lag_times=times - spec.flow_lag_s,
        smoke_lag=lag["smoke"], lwc_lag=lag["lwc"], tracer_lag=lag["tracer"],
        texcoords_lag=lag_tex, tex_weights_lag=lag_w,
        theta_anom=diag["theta_anom"], qv=diag["qv"], ux=diag["ux"], uz=diag["uz"],
        extra=dict(spec.extra),
    )
