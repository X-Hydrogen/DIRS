from __future__ import annotations

import numpy as np

from dirs.two_oscillator import numerical_sum_rule

M1 = 1.0
M2 = 1.7
K1 = 1.5
K2 = 0.8
KC = 0.25


def _matrices() -> tuple[np.ndarray, np.ndarray]:
    mass = np.diag([M1, M2])
    force = np.array([[K1, KC], [KC, K2]], dtype=float)
    return mass, force


def test_numerical_sum_rule_closes_all_components() -> None:
    mass, force = _matrices()
    result = numerical_sum_rule(mass, force, epsilon=1.0e-4, range_multiplier=100.0)

    assert np.max(result.relative_error_static) < 1.0e-6
    assert np.max(result.relative_error_regularized) < 1.0e-8
    assert np.all(result.quadrature_error < 1.0e-8)
    assert np.allclose(result.integral, result.integral.T, rtol=0.0, atol=1.0e-11)


def test_finite_epsilon_bias_matches_regularized_static_limit() -> None:
    mass, force = _matrices()
    coarse = numerical_sum_rule(mass, force, epsilon=1.0e-3, range_multiplier=100.0)
    fine = numerical_sum_rule(mass, force, epsilon=1.0e-4, range_multiplier=100.0)

    assert np.max(coarse.relative_error_regularized) < 1.0e-8
    assert np.max(fine.relative_error_regularized) < 1.0e-8
    assert np.max(fine.relative_error_static) < np.max(coarse.relative_error_static) / 50.0


def test_sum_rule_is_converged_with_frequency_range() -> None:
    mass, force = _matrices()
    medium = numerical_sum_rule(mass, force, epsilon=1.0e-4, range_multiplier=50.0)
    wide = numerical_sum_rule(mass, force, epsilon=1.0e-4, range_multiplier=100.0)

    assert np.allclose(medium.integral, wide.integral, rtol=1.0e-8, atol=1.0e-10)


def test_sum_rule_never_evaluates_at_zero() -> None:
    mass, force = _matrices()
    result = numerical_sum_rule(mass, force)
    assert result.omega_min > 0.0
    assert result.breakpoints[0] == result.omega_min
    assert 0.0 not in result.breakpoints
