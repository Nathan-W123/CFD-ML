"""Scalar transport (advection-diffusion, budgets) and moist thermodynamics."""

import numpy as np
import pytest

from src.cfd import thermo
from src.cfd.lbm import LatticeUnits, LBMConfig, LBMSolver
from src.cfd.scalars import MassBudget, ScalarSpec, ScalarTransport

UNITS = LatticeUnits(5.0, 0.1)


def _blob(nz, nx, k0, i0, r2=20.0):
    zz, xx = np.mgrid[0:nz, 0:nx]
    return np.exp(-((xx - i0) ** 2 + (zz - k0) ** 2) / r2)


def test_uniform_field_stays_uniform_in_lbm_flow_with_terrain():
    nz = nx = 48
    solid = np.zeros((nz, nx), dtype=bool)
    solid[:5, 20:30] = True
    st = ScalarTransport(nx, nz, UNITS, solid, [ScalarSpec("th", 5.0, nonnegative=False)])
    lbm = LBMSolver(LBMConfig(nx, nz, UNITS, 5.0), solid=solid, u_ambient=np.full(nz, 4.0))
    fields = {"th": np.full((nz, nx), 288.0)}
    ghost = {"th": {"left": np.full(nz, 288.0), "right": np.full(nz, 288.0), "top": np.full(nx, 288.0)}}
    for _ in range(400):
        lbm.step()
        st.step(fields, *lbm.velocity(), ghost)
    assert np.abs(fields["th"][~solid] - 288.0).max() < 1e-9


def test_blob_translates_with_uniform_flow_and_stays_nonnegative():
    nz, nx = 32, 96
    st = ScalarTransport(nx, nz, UNITS, None, [ScalarSpec("c", 0.0)], lateral="periodic", top="wall")
    c = _blob(nz, nx, 16, 20)
    fields = {"c": c}
    ux = np.full((nz, nx), 5.0)  # 0.1 cells / step
    uz = np.zeros((nz, nx))
    x0 = (c.sum(axis=0) * np.arange(nx)).sum() / c.sum()
    for _ in range(300):
        st.step(fields, ux, uz)
    x1 = (c.sum(axis=0) * np.arange(nx)).sum() / c.sum()
    assert abs((x1 - x0) - 30.0) < 0.5
    assert c.min() >= 0.0


def test_muscl_less_diffusive_than_upwind():
    nz, nx = 16, 96
    peaks = {}
    for scheme in ("upwind", "muscl"):
        st = ScalarTransport(nx, nz, UNITS, None, [ScalarSpec("c", 0.0)], scheme=scheme,
                             lateral="periodic", top="wall")
        c = _blob(nz, nx, 8, 20, r2=8.0)
        for _ in range(400):
            st.step({"c": c}, np.full((nz, nx), 5.0), np.zeros((nz, nx)))
        peaks[scheme] = c.max()
    assert peaks["muscl"] > 1.3 * peaks["upwind"]


def test_budget_closes_with_open_boundaries_sources_and_settling():
    nz = nx = 48
    solid = np.zeros((nz, nx), dtype=bool)
    solid[:4, 10:16] = True
    specs = [ScalarSpec("c", 5.0), ScalarSpec("d", 5.0, settling_velocity=0.4)]
    st = ScalarTransport(nx, nz, UNITS, solid, specs)
    lbm = LBMSolver(LBMConfig(nx, nz, UNITS, 5.0), solid=solid, u_ambient=np.full(nz, -3.0))
    fields = {"c": _blob(nz, nx, 20, 30), "d": _blob(nz, nx, 12, 30)}
    for f in fields.values():
        f[solid] = 0.0
    st.start_budget(fields)
    rate = np.zeros((nz, nx))
    rate[8, 30] = 2.0
    ghost = {"c": {"left": np.full(nz, 0.2), "right": np.full(nz, 0.3), "top": np.full(nx, 0.1)}}
    for _ in range(800):
        lbm.step()
        st.step(fields, *lbm.velocity(), ghost)
        st.add_source("c", fields["c"], rate)
    res = st.residuals(fields)
    assert abs(res["c"]) < 1e-10 and abs(res["d"]) < 1e-10
    b = st.budgets
    assert b["c"].inflow > 0 and b["c"].outflow > 0 and b["c"].sources > 0
    assert b["d"].deposited > 0
    assert b["c"].clipped == 0.0 and b["d"].clipped == 0.0
    # The weakly compressible correction stays a small part of the budget.
    assert abs(b["c"].compressibility) < 0.01 * b["c"].scale()
    assert min(f[~solid].min() for f in fields.values()) >= 0.0


def test_closed_box_conserves_exactly():
    nz = nx = 32
    st = ScalarTransport(nx, nz, UNITS, None, [ScalarSpec("c", 8.0)], lateral="wall", top="wall")
    lbm = LBMSolver(LBMConfig(nx, nz, UNITS, 5.0, lateral="wall", top="wall"))
    zz, xx = np.mgrid[0:nz, 0:nx]
    az = 0.3 * np.exp(-((xx - 16) ** 2 + (zz - 8) ** 2) / 10.0)
    fields = {"c": _blob(nz, nx, 8, 16)}
    st.start_budget(fields)
    for _ in range(500):
        lbm.step(None, az)
        st.step(fields, *lbm.velocity())
    b = st.budgets["c"]
    assert b.inflow == 0.0 and b.outflow == 0.0
    assert abs(st.residuals(fields)["c"]) < 1e-12


def test_diffusion_matches_analytic_variance_growth():
    nz, nx = 8, 128
    K = 10.0
    st = ScalarTransport(nx, nz, UNITS, None, [ScalarSpec("c", K)], lateral="periodic", top="wall")
    x = np.arange(nx) - 64.0
    sigma0 = 3.0
    c = np.tile(np.exp(-x ** 2 / (2 * sigma0 ** 2)), (nz, 1))
    steps = 2000
    zero = np.zeros((nz, nx))
    for _ in range(steps):
        st.step({"c": c}, zero, zero)
    prof = c[0] / c[0].sum()
    var = (prof * x ** 2).sum()
    expected = sigma0 ** 2 + 2 * K * steps * UNITS.dt / UNITS.dx ** 2
    assert abs(var - expected) / expected < 0.02


def test_signed_scalar_allows_negative_values():
    st = ScalarTransport(8, 8, UNITS, None, [ScalarSpec("a", 1.0, nonnegative=False)])
    f = {"a": np.full((8, 8), -1.0)}
    st.start_budget(f)
    st.step(f, np.zeros((8, 8)), np.zeros((8, 8)))
    assert np.allclose(f["a"], -1.0)
    assert st.budgets["a"].clipped == 0.0


def test_stability_limits_are_enforced():
    with pytest.raises(ValueError):
        ScalarTransport(8, 8, UNITS, None, [ScalarSpec("c", 100.0)])
    st = ScalarTransport(8, 8, UNITS, None, [ScalarSpec("c", 1.0)])
    with pytest.raises(ValueError):
        st.step({"c": np.zeros((8, 8))}, np.full((8, 8), 40.0), np.zeros((8, 8)))


def test_mass_budget_arithmetic():
    b = MassBudget(initial=1.0, sources=2.0, inflow=0.5, outflow=0.25, deposited=0.25,
                   conversion=-0.5, compressibility=0.1, clipped=0.0)
    assert np.isclose(b.expected(), 2.6)
    assert np.isclose(b.residual(2.6), 0.0)


# ---------------------------------------------------------------- thermodynamics
Z = (np.arange(16) + 0.5) * 5.0
P = thermo.pressure_profile(Z)
PI = thermo.exner(P)


def test_saturation_increases_with_temperature():
    T = np.linspace(260.0, 310.0, 11)
    qs = thermo.saturation_mixing_ratio(T, 101325.0)
    assert np.all(np.diff(qs) > 0)
    # ~10.0 g/kg at 287 K and sea-level pressure (standard tables).
    assert 9.0e-3 < thermo.saturation_mixing_ratio(287.0, 101325.0) < 11.0e-3


def _state(rh, ql0, theta0=283.0):
    theta = np.full((16, 8), theta0)
    T = theta * PI[:, None]
    qv = rh * thermo.saturation_mixing_ratio(T, P[:, None])
    ql = np.full_like(theta, ql0)
    return theta, qv, ql


def test_supersaturated_air_condenses_to_saturation_and_warms():
    theta, qv, ql = _state(1.03, 0.0)
    total = qv + ql
    theta0 = theta.copy()
    dq = thermo.saturation_adjustment(theta, qv, ql, P, PI)
    rh = thermo.relative_humidity(qv, theta * PI[:, None], P[:, None])
    assert np.all(dq > 0) and np.all(ql > 0)
    assert np.allclose(rh, 1.0, atol=1e-3)
    assert np.all(theta > theta0)
    assert np.allclose(theta - theta0, thermo.L_V / thermo.CP * dq / PI[:, None])
    assert np.abs(qv + ql - total).max() < 1e-15


def test_subsaturated_fog_evaporates_and_cools():
    theta, qv, ql = _state(0.9, 1e-4)
    total = qv + ql
    dq = thermo.saturation_adjustment(theta, qv, ql, P, PI)
    assert np.all(dq < 0)
    assert np.all(ql == 0.0)          # 0.1 g/kg cannot saturate 10 % RH deficit
    assert np.all(theta < 283.0)
    assert np.abs(qv + ql - total).max() < 1e-15
    assert qv.min() >= 0


def test_partial_evaporation_reaches_saturation():
    theta, qv, ql = _state(0.995, 5e-4)
    thermo.saturation_adjustment(theta, qv, ql, P, PI)
    rh = thermo.relative_humidity(qv, theta * PI[:, None], P[:, None])
    assert np.all(ql > 0)
    assert np.allclose(rh, 1.0, atol=1e-3)


def test_saturated_fog_is_a_fixed_point():
    theta, qv, ql = _state(1.03, 2e-4)
    thermo.saturation_adjustment(theta, qv, ql, P, PI)
    snapshot = (theta.copy(), qv.copy(), ql.copy())
    dq = thermo.saturation_adjustment(theta, qv, ql, P, PI)
    assert np.abs(dq).max() < 1e-12
    assert np.allclose(theta, snapshot[0]) and np.allclose(ql, snapshot[2])


def test_dry_air_is_untouched():
    theta, qv, ql = _state(0.5, 0.0)
    before = qv.copy()
    dq = thermo.saturation_adjustment(theta, qv, ql, P, PI)
    assert np.all(dq == 0.0) and np.array_equal(qv, before)


def test_fog_optics():
    beta = thermo.fog_extinction(np.array([0.0, 0.1, 0.4]))
    assert beta[0] == 0.0
    # LWC 0.1 g/m^3 -> visibility (3.912 / beta) of roughly 200 m.
    assert 150.0 < 3.912 / beta[1] < 260.0
    assert beta[2] > beta[1]
    assert np.isclose(thermo.lwc_from_ql(1e-4, 1.2), 0.12)
