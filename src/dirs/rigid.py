"""Rigid-motion projection for molecular Cartesian Hessians."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import null_space

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RigidMotionBasis:
    """Mass-weighted rigid-motion basis and complementary vibrational subspace."""

    center_of_mass: FloatArray
    translation_basis: FloatArray
    rotation_basis: FloatArray
    rigid_basis: FloatArray
    vibrational_basis: FloatArray
    projector: FloatArray
    rotational_singular_values: FloatArray
    n_translations: int
    n_rotations: int
    is_linear: bool


def _validate_geometry(positions: FloatArray, masses: FloatArray) -> tuple[FloatArray, FloatArray]:
    positions = np.asarray(positions, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3).")
    if masses.shape != (positions.shape[0],):
        raise ValueError("masses must have shape (n_atoms,).")
    if positions.shape[0] < 2:
        raise ValueError("Stage 0B requires at least two atoms.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(masses)):
        raise ValueError("positions and masses must be finite.")
    if np.any(masses <= 0.0):
        raise ValueError("All masses must be strictly positive.")
    return positions, masses


def expanded_masses(masses: FloatArray) -> FloatArray:
    """Repeat atomic masses over x, y, z Cartesian components."""
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim != 1 or np.any(masses <= 0.0):
        raise ValueError("masses must be a one-dimensional positive array.")
    return np.repeat(masses, 3)


def mass_weighted_rigid_basis(
    positions: FloatArray,
    masses: FloatArray,
    *,
    linearity_tolerance: float = 1.0e-10,
) -> RigidMotionBasis:
    """Construct orthonormal mass-weighted translation and rotation vectors.

    Positions may use any consistent length unit. Masses may use any consistent
    mass unit. Only the subspaces are used, so a uniform change of either unit
    leaves the resulting projector unchanged up to roundoff.
    """
    positions, masses = _validate_geometry(positions, masses)
    if linearity_tolerance <= 0.0:
        raise ValueError("linearity_tolerance must be strictly positive.")

    n_atoms = positions.shape[0]
    n_cart = 3 * n_atoms
    sqrt_masses = np.sqrt(masses)
    total_mass = float(np.sum(masses))
    center_of_mass = np.sum(positions * masses[:, None], axis=0) / total_mass
    centered = positions - center_of_mass

    translation_basis = np.zeros((n_cart, 3), dtype=np.float64)
    for axis in range(3):
        translation_basis[axis::3, axis] = sqrt_masses / np.sqrt(total_mass)

    rotation_candidates = np.zeros((n_cart, 3), dtype=np.float64)
    axes = np.eye(3)
    for axis in range(3):
        displacements = np.cross(axes[axis], centered)
        rotation_candidates[:, axis] = (sqrt_masses[:, None] * displacements).reshape(-1)

    rotation_candidates -= translation_basis @ (
        translation_basis.T @ rotation_candidates
    )
    u_rot, singular_values, _ = np.linalg.svd(rotation_candidates, full_matrices=False)
    max_singular = float(singular_values[0]) if singular_values.size else 0.0
    if max_singular == 0.0:
        n_rotations = 0
    else:
        n_rotations = int(
            np.count_nonzero(singular_values > linearity_tolerance * max_singular)
        )
    if n_rotations not in (2, 3):
        raise ValueError(
            "Molecular geometry does not define the expected two or three rotations."
        )
    rotation_basis = u_rot[:, :n_rotations]

    rigid_basis = np.column_stack((translation_basis, rotation_basis))
    u_rigid, s_rigid, _ = np.linalg.svd(rigid_basis, full_matrices=False)
    expected_rank = 3 + n_rotations
    if np.count_nonzero(s_rigid > linearity_tolerance * s_rigid[0]) != expected_rank:
        raise ValueError("Rigid-motion candidates are numerically rank deficient.")
    rigid_basis = u_rigid[:, :expected_rank]

    vibrational_basis = null_space(rigid_basis.T, rcond=linearity_tolerance)
    expected_vibrations = n_cart - expected_rank
    if vibrational_basis.shape != (n_cart, expected_vibrations):
        raise ValueError("Failed to construct the expected vibrational complement.")

    projector = np.eye(n_cart) - rigid_basis @ rigid_basis.T
    projector = 0.5 * (projector + projector.T)

    return RigidMotionBasis(
        center_of_mass=center_of_mass,
        translation_basis=translation_basis,
        rotation_basis=rotation_basis,
        rigid_basis=rigid_basis,
        vibrational_basis=vibrational_basis,
        projector=projector,
        rotational_singular_values=singular_values,
        n_translations=3,
        n_rotations=n_rotations,
        is_linear=n_rotations == 2,
    )
