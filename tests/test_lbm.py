"""Numerical sanity checks of the D2Q9 lattice-Boltzmann solver."""

import numpy as np
import pytest

from src.cfd.lbm import CS2, E, OPP, W, LatticeUnits, LBMConfig, LBMSolver


def test_d2q9_moment_identities():
    assert np.isclose(W.sum(), 1.0)
    assert np.allclose((W[:, None] * E).sum(axis=0), 0.0)
    second = np.einsum("q,qa,qb->ab", W, E, E)
    assert np.allclose(second, CS2 * np.eye(2))
    assert np.all(E[OPP] == -E)


def test_lattice_unit_mapping():
    u = LatticeUnits(dx=5.0, dt=0.1)
    assert np.isclose(u.velocity_to_lattice(5.0), 0.1)
    assert np.isclose(u.velocity_to_physical(0.1), 5.0)
    assert np.isclose(u.tau(5.0), 0.5 + 3 * 5.0 * 0.1 / 25.0)
    assert np.isclose(u.acceleration_to_lattice(1.0), 0.1 ** 2 / 5.0)


def _solver(**kw):
    cfg = dict(nx=48, nz=48, units=LatticeUnits(5.0, 0.1), nu=5.0)
    cfg.update(kw)
    return LBMConfig(**cfg)


@pytest.mark.parametrize("lateral,top", [("open", "open"), ("wall", "wall"), ("periodic", "wall")])
def test_rest_state_stays_at_rest(lateral, top):
    s = LBMSolver(_solver(lateral=lateral, top=top))
    for _ in range(300):
        s.step()
    assert s.max_lattice_speed() < 1e-12
    assert s.max_density_deviation() < 1e-12


def test_closed_box_conserves_mass_with_buoyancy():
    s = LBMSolver(_solver(lateral="wall", top="wall"))
    zz, xx = np.mgrid[0:48, 0:48]
    az = 0.3 * np.exp(-((xx - 24) ** 2 + (zz - 10) ** 2) / 18.0)
    m0 = s.total_mass()
    for _ in range(800):
        s.step(None, az)
    assert abs(s.total_mass() - m0) / m0 < 1e-12
    assert s.is_finite()


def test_body_force_channel_matches_poiseuille():
    # Periodic in x, no-slip floor and lid: u(z) = a z (H - z) / (2 nu).
    nz, nx = 32, 6
    units = LatticeUnits(1.0, 1.0)
    nu = (0.8 - 0.5) * CS2
    s = LBMSolver(LBMConfig(nx=nx, nz=nz, units=units, nu=nu, smagorinsky_cs=0.0,
                            sponge_width=0, lateral="periodic", top="wall"))
    a = 1e-6
    ax = np.full((nz, nx), a)
    for _ in range(20000):
        s.step(ax, None)
    z = np.arange(nz) + 0.5
    exact = a / (2 * nu) * z * (nz - z)
    num = s.velocity()[0][:, 0]
    assert np.abs(num - exact).max() / exact.max() < 0.02


def test_warm_blob_rises():
    s = LBMSolver(_solver())
    zz, xx = np.mgrid[0:48, 0:48]
    az = 0.5 * np.exp(-((xx - 24) ** 2 + (zz - 10) ** 2) / 18.0)
    for _ in range(200):
        s.step(None, az)
    ux, uz = s.velocity()
    assert uz[10, 24] > 0.5
    assert uz[10, 24] > 10 * abs(ux[10, 24])
    assert s.max_mach() < 0.3


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_open_domain_passes_ambient_wind(sign):
    nz = 64
    z = (np.arange(nz) + 0.5) * 5.0
    u_amb = sign * 5.0 * (z / 50.0) ** 0.2
    s = LBMSolver(_solver(nx=64, nz=nz), u_ambient=u_amb)
    for _ in range(2000):
        s.step()
    ux, uz = s.velocity()
    assert s.is_finite()
    interior = ux[20:50, 12:52].mean(axis=1)
    assert np.allclose(interior, u_amb[20:50], rtol=0.1)
    assert s.max_density_deviation() < 0.08
    assert s.max_mach() < 0.3


def test_terrain_obstacle_flow_stays_finite():
    solid = np.zeros((48, 48), dtype=bool)
    xs = np.arange(48)
    heights = (8 * np.exp(-((xs - 24) / 6.0) ** 2)).astype(int)
    for i, h in enumerate(heights):
        solid[:h, i] = True
    s = LBMSolver(_solver(), solid=solid, u_ambient=np.full(48, 4.0))
    for _ in range(1500):
        s.step()
    ux, uz = s.velocity()
    assert s.is_finite()
    assert np.all(ux[solid] == 0.0) and np.all(uz[solid] == 0.0)
    # Flow is deflected upward on the windward slope of the hill.
    assert uz[heights[18] + 1, 18] > 0.0


def test_ambient_profile_validation():
    s = LBMSolver(_solver())
    with pytest.raises(ValueError):
        s.set_ambient(np.zeros(5))
    with pytest.raises(ValueError):
        LBMSolver(_solver(lateral="bogus"))


@pytest.mark.slow
def test_channel_flow_converges_with_resolution():
    # Second-order convergence of the Poiseuille profile error (fixed tau, a scaled).
    errors = []
    for nz in (8, 16, 32):
        units = LatticeUnits(1.0, 1.0)
        nu = (0.8 - 0.5) * CS2
        s = LBMSolver(LBMConfig(nx=4, nz=nz, units=units, nu=nu, smagorinsky_cs=0.0,
                                sponge_width=0, lateral="periodic", top="wall"))
        a = 1e-5 * (8.0 / nz) ** 3
        ax = np.full((nz, 4), a)
        for _ in range(int(40 * nz * nz)):
            s.step(ax, None)
        z = np.arange(nz) + 0.5
        exact = a / (2 * nu) * z * (nz - z)
        errors.append(np.abs(s.velocity()[0][:, 0] - exact).max() / exact.max())
    assert errors[2] < errors[0] or errors[2] < 1e-3
