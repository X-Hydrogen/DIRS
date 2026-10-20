"""General harmonic molecular response in the vibrational subspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .coordinates import mass_weight_wilson
from .rigid import RigidMotionBasis, expanded_masses, mass_weighted_rigid_basis

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
ResponseMethod = Literal["modal", "direct"]


@dataclass(frozen=True)
class MolecularVibrations:
    """Projected harmonic analysis for one molecular Cartesian Hessian."""

    positions: FloatArray
    masses: FloatArray
    hessian: FloatArray
    rigid: RigidMotionBasis
    mass_weighted_hessian: FloatArray
    projected_hessian: FloatArray
    reduced_hessian: FloatArray
    eigenvalues: FloatArray
    angular_frequencies: FloatArray
    modes: FloatArray
    rigid_block_norm: float
    rigid_vibrational_coupling_norm: float


def _validate_hessian(hessian: FloatArray, n_cart: int) -> FloatArray:
    hessian = np.asarray(hessian, dtype=np.float64)
    if hessian.shape != (n_cart, n_cart):
        raise ValueError("hessian must have shape (3*n_atoms, 3*n_atoms).")
    if not np.all(np.isfinite(hessian)):
        raise ValueError("hessian must contain only finite values.")
    asymmetry = float(np.max(np.abs(hessian - hessian.T)))
    scale = max(float(np.max(np.abs(hessian))), 1.0)
    if asymmetry > 1.0e-10 * scale:
        raise ValueError("hessian must be symmetric within numerical tolerance.")
    return 0.5 * (hessian + hessian.T)


def analyze_molecular_hessian(
    positions: FloatArray,
    masses: FloatArray,
    hessian: FloatArray,
    *,
    linearity_tolerance: float = 1.0e-10,
    imaginary_tolerance: float = 1.0e-10,
) -> MolecularVibrations:
    """Project rigid motions and diagonalize the physical vibrational Hessian."""
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    rigid = mass_weighted_rigid_basis(
        positions,
        masses,
        linearity_tolerance=linearity_tolerance,
    )
    n_cart = positions.size
    hessian = _validate_hessian(hessian, n_cart)
    if imaginary_tolerance <= 0.0:
        raise ValueError("imaginary_tolerance must be strictly positive.")

    mass_vector = expanded_masses(masses)
    inverse_sqrt_mass = 1.0 / np.sqrt(mass_vector)
    mass_weighted_hessian = (
        inverse_sqrt_mass[:, None] * hessian * inverse_sqrt_mass[None, :]
    )
    mass_weighted_hessian = 0.5 * (
        mass_weighted_hessian + mass_weighted_hessian.T
    )

    projector = rigid.projector
    projected_hessian = projector @ mass_weighted_hessian @ projector
    projected_hessian = 0.5 * (projected_hessian + projected_hessian.T)
    z = rigid.vibrational_basis
    reduced_hessian = z.T @ projected_hessian @ z
    reduced_hessian = 0.5 * (reduced_hessian + reduced_hessian.T)

    eigenvalues, reduced_modes = np.linalg.eigh(reduced_hessian)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    negative_cutoff = imaginary_tolerance * scale
    if np.any(eigenvalues < -negative_cutoff):
        minimum = float(np.min(eigenvalues))
        raise ValueError(f"Projected Hessian contains an imaginary mode: {minimum:.6e}.")
    eigenvalues = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
    angular_frequencies = np.sqrt(eigenvalues)
    modes = z @ reduced_modes

    q = rigid.rigid_basis
    rigid_block_norm = float(np.linalg.norm(q.T @ mass_weighted_hessian @ q))
    rigid_vibrational_coupling_norm = float(
        np.linalg.norm(q.T @ mass_weighted_hessian @ z)
    )

    return MolecularVibrations(
        positions=positions.copy(),
        masses=masses.copy(),
        hessian=hessian,
        rigid=rigid,
        mass_weighted_hessian=mass_weighted_hessian,
        projected_hessian=projected_hessian,
        reduced_hessian=reduced_hessian,
        eigenvalues=eigenvalues,
        angular_frequencies=angular_frequencies,
        modes=modes,
        rigid_block_norm=rigid_block_norm,
        rigid_vibrational_coupling_norm=rigid_vibrational_coupling_norm,
    )


def coordinate_mode_amplitudes(
    analysis: MolecularVibrations,
    wilson: FloatArray,
) -> FloatArray:
    """Project one Cartesian Wilson vector onto mass-weighted normal modes."""
    weighted = mass_weight_wilson(wilson, analysis.masses)
    return analysis.modes.T @ weighted


def modal_response_contributions(
    analysis: MolecularVibrations,
    wilson_a: FloatArray,
    wilson_b: FloatArray,
    *,
    omega: float = 0.0,
    epsilon: float = 0.0,
) -> ComplexArray:
    """Return per-mode contributions to the local-coordinate susceptibility."""
    if omega < 0.0:
        raise ValueError("omega must be non-negative.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if omega > 0.0 and epsilon == 0.0:
        raise ValueError("A positive-frequency retarded response requires epsilon > 0.")
    if np.any(analysis.eigenvalues <= 0.0):
        raise ValueError("Response is undefined because a vibrational eigenvalue is non-positive.")

    a = coordinate_mode_amplitudes(analysis, wilson_a)
    b = coordinate_mode_amplitudes(analysis, wilson_b)
    z = complex(omega, epsilon)
    return (b * a / (analysis.eigenvalues - z**2)).astype(np.complex128)


def local_coordinate_response(
    analysis: MolecularVibrations,
    wilson_a: FloatArray,
    wilson_b: FloatArray,
    *,
    omega: float = 0.0,
    epsilon: float = 0.0,
    method: ResponseMethod = "modal",
) -> complex:
    """Evaluate one signed self- or cross-response matrix element."""
    if method == "modal":
        return complex(
            np.sum(
                modal_response_contributions(
                    analysis,
                    wilson_a,
                    wilson_b,
                    omega=omega,
                    epsilon=epsilon,
                )
            )
        )
    if method != "direct":
        raise ValueError("method must be either 'modal' or 'direct'.")
    if omega < 0.0:
        raise ValueError("omega must be non-negative.")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative.")
    if omega > 0.0 and epsilon == 0.0:
        raise ValueError("A positive-frequency retarded response requires epsilon > 0.")
    if np.any(analysis.eigenvalues <= 0.0):
        raise ValueError("Response is undefined because a vibrational eigenvalue is non-positive.")

    z_basis = analysis.rigid.vibrational_basis
    weighted_a = mass_weight_wilson(wilson_a, analysis.masses)
    weighted_b = mass_weight_wilson(wilson_b, analysis.masses)
    reduced_a = z_basis.T @ weighted_a
    reduced_b = z_basis.T @ weighted_b
    z_frequency = complex(omega, epsilon)
    dynamic = analysis.reduced_hessian.astype(np.complex128) - z_frequency**2 * np.eye(
        analysis.reduced_hessian.shape[0]
    )
    solution = np.linalg.solve(dynamic, reduced_a.astype(np.complex128))
    return complex(reduced_b @ solution)


def local_response_matrix(
    analysis: MolecularVibrations,
    wilson_vectors: list[FloatArray],
    *,
    omega: float = 0.0,
    epsilon: float = 0.0,
    method: ResponseMethod = "modal",
) -> ComplexArray:
    """Return the susceptibility matrix for an ordered list of local coordinates."""
    n_coordinates = len(wilson_vectors)
    if n_coordinates == 0:
        raise ValueError("At least one Wilson vector is required.")
    response = np.empty((n_coordinates, n_coordinates), dtype=np.complex128)
    for row, wilson_b in enumerate(wilson_vectors):
        for column, wilson_a in enumerate(wilson_vectors):
            response[row, column] = local_coordinate_response(
                analysis,
                wilson_a,
                wilson_b,
                omega=omega,
                epsilon=epsilon,
                method=method,
            )
    return response


def vibrational_pseudoinverse(analysis: MolecularVibrations) -> FloatArray:
    """Return the mass-weighted Green's function with only rigid modes nulled."""
    if np.any(analysis.eigenvalues <= 0.0):
        raise ValueError("Pseudoinverse is undefined for non-positive vibrational modes.")
    return (analysis.modes / analysis.eigenvalues[None, :]) @ analysis.modes.T
