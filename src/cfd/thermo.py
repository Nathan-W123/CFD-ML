"""Moist thermodynamics for fog: saturation, saturation adjustment, optics.

The temperature scalar is potential temperature theta; absolute temperature
is T = theta * Exner(p(z)) with a hydrostatic reference pressure. Fog is
liquid water q_l produced by saturation adjustment: after transport, any
supersaturation (q_v > q_s(T, p)) condenses and any liquid in sub-saturated
air evaporates, until the cell is exactly saturated or the liquid is gone.
Latent heat L_v dq warms (condensation) or cools (evaporation) the air, and
the saturation point is solved for with that feedback (Newton iterations).
Total water q_v + q_l is conserved to round-off and neither goes negative.

Where this fits: `simulation.py` calls `saturation_adjustment` once per step
after scalar transport; `fog.py` builds initial and inflow moisture profiles
with `saturation_mixing_ratio`; the renderer converts q_l to optical
extinction with `lwc_from_ql` and `fog_extinction`.
Calls into: numpy and numba.
Start reading at: saturation_adjustment.
"""

from __future__ import annotations

import math

import numba
import numpy as np

G = 9.81
CP = 1004.0
R_D = 287.04
R_V = 461.5
L_V = 2.5e6
EPS = R_D / R_V          # 0.622
P0 = 100000.0
KAPPA = R_D / CP


def pressure_profile(z: np.ndarray, p_surface: float = 101325.0, theta0: float = 288.0) -> np.ndarray:
    """Hydrostatic pressure (Pa) at heights z (m) for a constant potential temperature theta0."""
    z = np.asarray(z, dtype=np.float64)
    pi_s = (p_surface / P0) ** KAPPA
    pi = pi_s - G * z / (CP * theta0)
    return P0 * pi ** (1.0 / KAPPA)


def exner(p: np.ndarray) -> np.ndarray:
    """Exner function (p / P0) ** (R_d / c_p)."""
    return (np.asarray(p, dtype=np.float64) / P0) ** KAPPA


def saturation_vapor_pressure(T: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over liquid water (Pa), Bolton (1980); T in K."""
    T = np.asarray(T, dtype=np.float64)
    return 611.2 * np.exp(17.67 * (T - 273.15) / (T - 29.65))


def saturation_mixing_ratio(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Saturation mixing ratio (kg/kg) at temperature T (K) and pressure p (Pa)."""
    es = saturation_vapor_pressure(T)
    p = np.asarray(p, dtype=np.float64)
    es = np.minimum(es, 0.5 * p)
    return EPS * es / (p - es)


def mixing_ratio_from_rh(rh: np.ndarray, T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Vapour mixing ratio (kg/kg) for relative humidity rh (0-1, as q_v / q_s)."""
    return np.asarray(rh, dtype=np.float64) * saturation_mixing_ratio(T, p)


def relative_humidity(qv: np.ndarray, T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """q_v / q_s (the mixing-ratio definition of relative humidity)."""
    return np.asarray(qv, dtype=np.float64) / saturation_mixing_ratio(T, p)


@numba.njit(cache=True)
def _qs_and_derivative(T, p):
    es = 611.2 * math.exp(17.67 * (T - 273.15) / (T - 29.65))
    if es > 0.5 * p:
        es = 0.5 * p
    qs = EPS * es / (p - es)
    dln_es = 17.67 * (273.15 - 29.65) / ((T - 29.65) * (T - 29.65))
    dqs = qs * p / (p - es) * dln_es
    return qs, dqs


@numba.njit(cache=True)
def _adjust(theta, qv, ql, p, pi, iterations, out_dq):
    nz, nx = theta.shape
    gamma = L_V / CP
    for k in range(nz):
        pk = p[k]
        pik = pi[k]
        for i in range(nx):
            v = qv[k, i]
            l = ql[k, i]
            T0 = theta[k, i] * pik
            qs0, _ = _qs_and_derivative(T0, pk)
            if v <= qs0 and l <= 0.0:
                out_dq[k, i] = 0.0
                continue
            # Solve v - dq = q_s(T0 + gamma dq) for dq in [-l, v] (Newton, clamped).
            dq = 0.0
            for _ in range(iterations):
                qs, dqs = _qs_and_derivative(T0 + gamma * dq, pk)
                fval = v - dq - qs
                fder = -1.0 - dqs * gamma
                dq = dq - fval / fder
                if dq < -l:
                    dq = -l
                if dq > v:
                    dq = v
            out_dq[k, i] = dq
            qv[k, i] = v - dq
            ql[k, i] = l + dq
            if ql[k, i] < 0.0:
                ql[k, i] = 0.0
            theta[k, i] += gamma / pik * dq


def saturation_adjustment(theta: np.ndarray, qv: np.ndarray, ql: np.ndarray,
                          p: np.ndarray, exner_p: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Bring every cell to saturation or dryness, in place; return condensed dq (kg/kg).

    theta, qv, ql: (nz, nx) arrays (modified in place). p, exner_p: (nz,) or
    (nz, nx)-broadcastable per-row pressure (Pa) and Exner values. dq > 0 is
    condensation (vapour -> liquid, warming), dq < 0 evaporation.
    """
    if theta.dtype != np.float64 or qv.dtype != np.float64 or ql.dtype != np.float64:
        raise TypeError("saturation_adjustment works in place on float64 arrays")
    nz = theta.shape[0]
    p_rows = np.ascontiguousarray(np.broadcast_to(np.asarray(p, dtype=np.float64).reshape(-1), (nz,)))
    pi_rows = np.ascontiguousarray(np.broadcast_to(np.asarray(exner_p, dtype=np.float64).reshape(-1), (nz,)))
    dq = np.zeros_like(theta)
    _adjust(theta, qv, ql, p_rows, pi_rows, int(iterations), dq)
    return dq


def lwc_from_ql(ql: np.ndarray, rho_air: float = 1.2) -> np.ndarray:
    """Liquid water content (g/m^3) from the liquid mixing ratio (kg/kg)."""
    return np.asarray(ql, dtype=np.float64) * rho_air * 1000.0


def fog_extinction(lwc: np.ndarray) -> np.ndarray:
    """Visible extinction coefficient (1/m) of fog with LWC in g/m^3, Kunkel (1984)."""
    return 0.1447 * np.maximum(np.asarray(lwc, dtype=np.float64), 0.0) ** 0.88
