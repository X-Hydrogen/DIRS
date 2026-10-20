"""Analytic utilities for the Stage 0A two-coupled-oscillator model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.linalg import eigh

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SumRuleResult:
    """Numerical closure report for the Stage 0A spectral sum rule."""

    integral: FloatArray
    static_target: FloatArray
    regularized_target: FloatArray
    quadrature_error: FloatArray
    relative_error_static: FloatArray
    relative_error_regularized: FloatArray
    omega_min: float
    omega_max: float
    epsilon: float
    epsabs: float
    epsrel: float
    resonances: FloatArray
    breakpoints: tuple[float, ...]


def _validate_parameters(m1: float, m2: float, k1: float, k2: float, kc: float) -> None:
    """Validate positivity of masses and harmonic stability of the force matrix."""
    if m1 <= 0.0 or m2 <= 0.0:
        raise ValueError("Masses must be strictly positive.")
    if k1 <= 0.0 or k2 <= 0.0:
        raise ValueError("Diagonal force constants must be strictly positive.")
    if k1 * k2 - kc**2 <= 0.0:
        raise ValueError("The harmonic force matrix is not positive definite.")


def _validate_matrices(
    mass_matrix: FloatArray,
    force_matrix: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Validate and return symmetric positive-definite Stage 0A matrices."""
    mass_matrix = np.asarray(mass_matrix, dtype=np.float64)
    force_matrix = np.asarray(force_matrix, dtype=np.float64)

    if mass_matrix.shape != (2, 2) or force_matrix.shape != (2, 2):
        raise ValueError("Stage 0A matrices must both have shape (2, 2).")
    if not np.allclose(mass_matrix, mass_matrix.T):
        raise ValueError("The mass matrix must be symmetric.")
    if not np.allclose(force_matrix, force_matrix.T):
        raise ValueError("The force matrix must be symmetric.")
    if np.any(np.linalg.eigvalsh(mass_matrix) <= 0.0):
        raise ValueError("The mass matrix must be positive definite.")
    if np.any(np.linalg.eigvalsh(force_matrix) <= 0.0):
        raise ValueError("The force matrix must be positive definite.")
    return mass_matrix, force_matrix


def analytic_eigenfrequencies(
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    kc: float,
) -> FloatArray:
    """Return the two positive normal-mode angular frequencies in ascending order."""
    _validate_parameters(m1, m2, k1, k2, kc)

    trace_term = k1 * m2 + k2 * m1
    discriminant = (k1 * m2 - k2 * m1) ** 2 + 4.0 * m1 * m2 * kc**2
    denominator = 2.0 * m1 * m2

    omega_sq_low = (trace_term - np.sqrt(discriminant)) / denominator
    omega_sq_high = (trace_term + np.sqrt(discriminant)) / denominator
    return np.sqrt(np.array([omega_sq_low, omega_sq_high], dtype=np.float64))


def analytic_susceptibility(
    omega: float,
    m1: float,
    m2: float,
    k1: float,
    k2: float,
    kc: float,
    epsilon: float = 1.0e-6,
) -> ComplexArray:
    """Evaluate the analytic retarded susceptibility at one positive frequency."""
    _validate_parameters(m1, m2, k1, k2, kc)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")

    z = complex(omega, epsilon)
    d1 = k1 - m1 * z**2
    d2 = k2 - m2 * z**2
    determinant = d1 * d2 - kc**2

    return np.array(
        [[d2, -kc], [-kc, d1]],
        dtype=np.complex128,
    ) / determinant


def matrix_susceptibility(
    omega: float,
    mass_matrix: FloatArray,
    force_matrix: FloatArray,
    epsilon: float = 1.0e-6,
) -> ComplexArray:
    """Evaluate the retarded susceptibility by direct matrix inversion."""
    mass_matrix, force_matrix = _validate_matrices(mass_matrix, force_matrix)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")

    z = complex(omega, epsilon)
    dynamic_matrix = force_matrix.astype(np.complex128) - z**2 * mass_matrix
    return np.linalg.inv(dynamic_matrix)


def static_susceptibility(force_matrix: FloatArray) -> FloatArray:
    """Return the static susceptibility, equal to the inverse force matrix."""
    force_matrix = np.asarray(force_matrix, dtype=np.float64)
    if force_matrix.shape != (2, 2):
        raise ValueError("Stage 0A force matrix must have shape (2, 2).")
    if not np.allclose(force_matrix, force_matrix.T):
        raise ValueError("The force matrix must be symmetric.")
    if np.any(np.linalg.eigvalsh(force_matrix) <= 0.0):
        raise ValueError("The force matrix must be positive definite.")
    return np.linalg.inv(force_matrix)


def eta_from_static_chi(chi0: FloatArray) -> float:
    """Return the normalized static cross-response eta_12."""
    chi0 = np.asarray(chi0, dtype=np.float64)
    if chi0.shape != (2, 2):
        raise ValueError("Stage 0A susceptibility must have shape (2, 2).")
    denominator = np.sqrt(chi0[0, 0] * chi0[1, 1])
    if denominator <= 0.0:
        raise ValueError("Self-susceptibilities must be strictly positive.")
    return float(chi0[0, 1] / denominator)


def spectral_density(omega: float, chi: ComplexArray) -> FloatArray:
    """Return D(omega) = (2/pi) Im[chi(omega)] / omega for omega > 0."""
    if omega <= 0.0:
        raise ValueError("The Stage 0A spectral density is defined for omega > 0.")
    chi = np.asarray(chi, dtype=np.complex128)
    if chi.shape != (2, 2):
        raise ValueError("Stage 0A susceptibility must have shape (2, 2).")
    return (2.0 / np.pi) * np.imag(chi) / omega


def _relative_error(value: FloatArray, target: FloatArray) -> FloatArray:
    """Return elementwise relative error, using absolute error for zero targets."""
    absolute = np.abs(value - target)
    denominator = np.abs(target)
    return np.where(denominator > 0.0, absolute / denominator, absolute)


def _resonance_breakpoints(
    resonances: FloatArray,
    epsilon: float,
    omega_min: float,
    omega_max: float,
) -> tuple[float, ...]:
    """Construct resonance-aware integration intervals without touching zero."""
    points = {float(omega_min), float(omega_max)}
    width_multipliers = (-100.0, -20.0, -5.0, 0.0, 5.0, 20.0, 100.0)
    for resonance in resonances:
        for multiplier in width_multipliers:
            point = float(resonance + multiplier * epsilon)
            if omega_min < point < omega_max:
                points.add(point)
    return tuple(sorted(points))


def numerical_sum_rule(
    mass_matrix: FloatArray,
    force_matrix: FloatArray,
    *,
    epsilon: float = 1.0e-4,
    omega_min: float | None = None,
    omega_max: float | None = None,
    range_multiplier: float = 100.0,
    epsabs: float = 1.0e-11,
    epsrel: float = 1.0e-10,
    quad_limit: int = 400,
) -> SumRuleResult:
    """Numerically integrate the Stage 0A spectral sum rule.

    The finite-``epsilon`` prescription uses ``z = omega + i epsilon``. Its exact
    Kramers--Kronig target is ``[K + epsilon**2 M]**-1``; the desired static
    susceptibility ``K**-1`` is recovered as ``epsilon -> 0``. Both errors are
    reported so the regularization bias is never hidden by normalization.
    """
    mass_matrix, force_matrix = _validate_matrices(mass_matrix, force_matrix)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")
    if range_multiplier <= 1.0:
        raise ValueError("range_multiplier must be greater than one.")
    if epsabs <= 0.0 or epsrel <= 0.0:
        raise ValueError("Quadrature tolerances must be strictly positive.")
    if quad_limit < 50:
        raise ValueError("quad_limit must be at least 50.")

    eigenvalues = eigh(force_matrix, mass_matrix, eigvals_only=True, check_finite=True)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("Generalized eigenvalues must be strictly positive.")
    resonances = np.sqrt(eigenvalues).astype(np.float64)

    if omega_min is None:
        omega_min = max(np.finfo(float).eps * resonances[0], 1.0e-12 * resonances[0])
    if omega_max is None:
        omega_max = range_multiplier * resonances[-1]
    if omega_min <= 0.0:
        raise ValueError("omega_min must be strictly positive.")
    if omega_max <= resonances[-1]:
        raise ValueError("omega_max must exceed the highest resonance.")
    if omega_max <= omega_min:
        raise ValueError("omega_max must exceed omega_min.")

    breakpoints = _resonance_breakpoints(resonances, epsilon, omega_min, omega_max)
    integral = np.zeros((2, 2), dtype=np.float64)
    quadrature_error = np.zeros((2, 2), dtype=np.float64)

    for row in range(2):
        for column in range(2):

            def integrand(omega: float) -> float:
                chi = matrix_susceptibility(
                    omega,
                    mass_matrix,
                    force_matrix,
                    epsilon=epsilon,
                )
                return float(spectral_density(omega, chi)[row, column])

            component_integral = 0.0
            component_error = 0.0
            for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
                value, error = quad(
                    integrand,
                    lower,
                    upper,
                    epsabs=epsabs,
                    epsrel=epsrel,
                    limit=quad_limit,
                )
                component_integral += value
                component_error += error
            integral[row, column] = component_integral
            quadrature_error[row, column] = component_error

    static_target = static_susceptibility(force_matrix)
    regularized_target = np.linalg.inv(force_matrix + epsilon**2 * mass_matrix)

    return SumRuleResult(
        integral=integral,
        static_target=static_target,
        regularized_target=regularized_target,
        quadrature_error=quadrature_error,
        relative_error_static=_relative_error(integral, static_target),
        relative_error_regularized=_relative_error(integral, regularized_target),
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        epsilon=float(epsilon),
        epsabs=float(epsabs),
        epsrel=float(epsrel),
        resonances=resonances,
        breakpoints=breakpoints,
    )
