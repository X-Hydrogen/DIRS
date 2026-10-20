from __future__ import annotations

import numpy as np

from dirs.two_oscillator import (
    analytic_eigenfrequencies,
    analytic_susceptibility,
    eta_from_static_chi,
    matrix_susceptibility,
    spectral_density,
    static_susceptibility,
)

M1 = 1.0
M2 = 1.7
K1 = 1.5
K2 = 0.8
KC = 0.25


def _matrices(mass_scale: float = 1.0, coupling: float = KC) -> tuple[np.ndarray, np.ndarray]:
    mass = np.diag([M1, M2]) * mass_scale
    force = np.array([[K1, coupling], [coupling, K2]], dtype=float)
    return mass, force


def test_analytic_and_matrix_susceptibility_agree() -> None:
    mass, force = _matrices()
    for omega in np.linspace(0.05, 2.0, 41):
        analytic = analytic_susceptibility(omega, M1, M2, K1, K2, KC, 1.0e-5)
        matrix = matrix_susceptibility(omega, mass, force, 1.0e-5)
        assert np.allclose(analytic, matrix, rtol=1.0e-12, atol=1.0e-12)


def test_reciprocity() -> None:
    chi = analytic_susceptibility(0.7, M1, M2, K1, K2, KC)
    assert np.allclose(chi[0, 1], chi[1, 0], rtol=0.0, atol=1.0e-14)


def test_static_eta_matches_closed_form() -> None:
    _, force = _matrices()
    chi0 = static_susceptibility(force)
    eta = eta_from_static_chi(chi0)
    expected = -KC / np.sqrt(K1 * K2)
    assert np.isclose(eta, expected, rtol=1.0e-13, atol=1.0e-13)


def test_uniform_mass_scaling() -> None:
    original = analytic_eigenfrequencies(M1, M2, K1, K2, KC)
    scaled = analytic_eigenfrequencies(4.0 * M1, 4.0 * M2, K1, K2, KC)
    assert np.allclose(scaled, original / 2.0, rtol=1.0e-13, atol=1.0e-13)

    _, force = _matrices()
    chi0_before = static_susceptibility(force)
    chi0_after = static_susceptibility(force)
    assert np.allclose(chi0_before, chi0_after, rtol=0.0, atol=1.0e-14)
    assert np.isclose(
        eta_from_static_chi(chi0_before),
        eta_from_static_chi(chi0_after),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_zero_coupling_gives_zero_cross_response() -> None:
    mass, force = _matrices(coupling=0.0)
    for omega in np.linspace(0.05, 2.0, 31):
        chi = matrix_susceptibility(omega, mass, force, 1.0e-5)
        assert abs(chi[0, 1]) < 1.0e-14
        assert abs(chi[1, 0]) < 1.0e-14


def test_coordinate_reversal_changes_only_cross_sign() -> None:
    mass, force = _matrices()
    transform = np.diag([-1.0, 1.0])
    reversed_force = transform @ force @ transform
    reversed_mass = transform @ mass @ transform

    chi = matrix_susceptibility(0.6, mass, force, 1.0e-5)
    reversed_chi = matrix_susceptibility(0.6, reversed_mass, reversed_force, 1.0e-5)

    assert np.allclose(np.diag(reversed_chi), np.diag(chi), rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(reversed_chi[0, 1], -chi[0, 1], rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(reversed_chi[1, 0], -chi[1, 0], rtol=1.0e-12, atol=1.0e-12)


def test_self_spectral_density_is_nonnegative() -> None:
    mass, force = _matrices()
    for omega in np.linspace(0.01, 2.0, 200):
        chi = matrix_susceptibility(omega, mass, force, 1.0e-3)
        density = spectral_density(omega, chi)
        assert density[0, 0] >= -1.0e-12
        assert density[1, 1] >= -1.0e-12


def test_soft_mode_limit() -> None:
    near_boundary = 0.999999 * np.sqrt(K1 * K2)
    frequencies = analytic_eigenfrequencies(M1, M2, K1, K2, near_boundary)
    assert frequencies[0] < 1.0e-3

    _, force_regular = _matrices()
    _, force_soft = _matrices(coupling=near_boundary)
    assert np.linalg.norm(static_susceptibility(force_soft)) > np.linalg.norm(
        static_susceptibility(force_regular)
    )
