"""Local Cartesian coordinate derivatives used by DIRS."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .rigid import expanded_masses

FloatArray = NDArray[np.float64]


def distance_wilson_vector(
    positions: FloatArray,
    atom_i: int,
    atom_j: int,
    *,
    orientation: float = 1.0,
    distance_tolerance: float = 1.0e-14,
) -> FloatArray:
    """Return the Cartesian Wilson vector for an interatomic distance.

    ``orientation=-1`` represents the covariant coordinate convention
    ``q -> -q`` without changing the underlying unsigned distance.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (n_atoms, 3).")
    n_atoms = positions.shape[0]
    if atom_i == atom_j:
        raise ValueError("A distance requires two distinct atoms.")
    if atom_i < 0 or atom_i >= n_atoms or atom_j < 0 or atom_j >= n_atoms:
        raise IndexError("Distance atom index is out of range.")
    if orientation not in (-1.0, 1.0):
        raise ValueError("orientation must be either +1 or -1.")
    if distance_tolerance <= 0.0:
        raise ValueError("distance_tolerance must be strictly positive.")

    displacement = positions[atom_i] - positions[atom_j]
    distance = float(np.linalg.norm(displacement))
    if distance <= distance_tolerance:
        raise ValueError("Distance Wilson vector is undefined for coincident atoms.")
    unit = orientation * displacement / distance

    wilson = np.zeros(3 * n_atoms, dtype=np.float64)
    wilson[3 * atom_i : 3 * atom_i + 3] = unit
    wilson[3 * atom_j : 3 * atom_j + 3] = -unit
    return wilson


def mass_weight_wilson(wilson: FloatArray, masses: FloatArray) -> FloatArray:
    """Convert a Cartesian Wilson vector to mass-weighted coordinates."""
    wilson = np.asarray(wilson, dtype=np.float64)
    mass_vector = expanded_masses(masses)
    if wilson.shape != mass_vector.shape:
        raise ValueError("Wilson vector length must equal 3 * n_atoms.")
    return wilson / np.sqrt(mass_vector)
