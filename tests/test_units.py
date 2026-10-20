import numpy as np

from dirs.units import (
    amu_to_electron_masses,
    angstrom_to_bohr,
    angular_frequency_au_to_wavenumber,
    bohr_to_angstrom,
    compliance_angstrom2_per_hartree_to_bohr2_per_hartree,
    compliance_bohr2_per_hartree_to_angstrom2_per_hartree,
    electron_masses_to_amu,
    hessian_atomic_to_hartree_per_angstrom2,
    hessian_hartree_per_angstrom2_to_atomic,
    wavenumber_to_angular_frequency_au,
)


def test_unit_conversions_round_trip() -> None:
    values = np.array([0.1, 1.0, 7.3])
    assert np.allclose(bohr_to_angstrom(angstrom_to_bohr(values)), values)
    assert np.allclose(electron_masses_to_amu(amu_to_electron_masses(values)), values)
    assert np.allclose(
        wavenumber_to_angular_frequency_au(
            angular_frequency_au_to_wavenumber(values)
        ),
        values,
    )

    matrix = np.diag(values)
    assert np.allclose(
        hessian_atomic_to_hartree_per_angstrom2(
            hessian_hartree_per_angstrom2_to_atomic(matrix)
        ),
        matrix,
    )
    assert np.allclose(
        compliance_angstrom2_per_hartree_to_bohr2_per_hartree(
            compliance_bohr2_per_hartree_to_angstrom2_per_hartree(matrix)
        ),
        matrix,
    )
