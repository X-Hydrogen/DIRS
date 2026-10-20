"""Explicit atomic-unit conversions used at DIRS backend boundaries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

BOHR_PER_ANGSTROM = 1.8897261254578281
ELECTRON_MASSES_PER_AMU = 1822.888486209
HARTREE_WAVENUMBER_CM = 219474.6313705


def angstrom_to_bohr(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64) * BOHR_PER_ANGSTROM


def bohr_to_angstrom(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64) / BOHR_PER_ANGSTROM


def amu_to_electron_masses(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64) * ELECTRON_MASSES_PER_AMU


def electron_masses_to_amu(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64) / ELECTRON_MASSES_PER_AMU


def angular_frequency_au_to_wavenumber(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64) * HARTREE_WAVENUMBER_CM


def wavenumber_to_angular_frequency_au(values: FloatArray) -> FloatArray:
    return np.asarray(values, dtype=np.float64) / HARTREE_WAVENUMBER_CM


def hessian_hartree_per_angstrom2_to_atomic(values: FloatArray) -> FloatArray:
    """Convert a Cartesian Hessian from Eh/angstrom^2 to Eh/bohr^2."""
    return np.asarray(values, dtype=np.float64) / BOHR_PER_ANGSTROM**2


def hessian_atomic_to_hartree_per_angstrom2(values: FloatArray) -> FloatArray:
    """Convert a Cartesian Hessian from Eh/bohr^2 to Eh/angstrom^2."""
    return np.asarray(values, dtype=np.float64) * BOHR_PER_ANGSTROM**2


def compliance_bohr2_per_hartree_to_angstrom2_per_hartree(
    values: FloatArray,
) -> FloatArray:
    """Convert a distance-distance compliance from bohr^2/Eh to angstrom^2/Eh."""
    return np.asarray(values, dtype=np.float64) / BOHR_PER_ANGSTROM**2


def compliance_angstrom2_per_hartree_to_bohr2_per_hartree(
    values: FloatArray,
) -> FloatArray:
    """Convert a distance-distance compliance from angstrom^2/Eh to bohr^2/Eh."""
    return np.asarray(values, dtype=np.float64) * BOHR_PER_ANGSTROM**2
