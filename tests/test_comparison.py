from __future__ import annotations

import numpy as np
import pytest

from dirs.backends import hessian_data_from_pyscf_arrays
from dirs.comparison import compare_hessian_data, compare_mode_subspaces
from dirs.rigid import expanded_masses, mass_weighted_rigid_basis


def _synthetic_record(source: str = "pyscf"):
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [-0.3, 1.1, 0.2]],
        dtype=float,
    )
    masses = np.array([16.0, 1.0, 1.0])
    rigid = mass_weighted_rigid_basis(positions, masses)
    z = rigid.vibrational_basis
    reduced = np.diag([0.2, 0.9, 1.6])
    mass_weighted = z @ reduced @ z.T
    sqrt_mass = np.sqrt(expanded_masses(masses))
    hessian = sqrt_mass[:, None] * mass_weighted * sqrt_mass[None, :]
    blocks = hessian.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)
    record = hessian_data_from_pyscf_arrays(
        ["O", "H", "H"],
        positions,
        masses,
        blocks,
    )
    return record.__class__(
        symbols=record.symbols,
        positions_bohr=record.positions_bohr,
        masses_amu=record.masses_amu,
        hessian_hartree_per_bohr2=record.hessian_hartree_per_bohr2,
        source_format=source,
        energy_hartree=record.energy_hartree,
        multiplicity=record.multiplicity,
        max_hessian_asymmetry=record.max_hessian_asymmetry,
    )


def test_identical_backend_records_compare_exactly() -> None:
    reference = _synthetic_record("orca-hess")
    candidate = _synthetic_record("pyscf")
    result = compare_hessian_data(
        reference,
        candidate,
        coordinate_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    assert result.geometry_max_abs_bohr == 0.0
    assert result.mass_max_abs_amu == 0.0
    assert result.hessian_max_abs_hartree_per_bohr2 == 0.0
    assert result.hessian_rms_hartree_per_bohr2 == 0.0
    assert result.hessian_relative_frobenius == 0.0
    assert result.frequency_max_abs == 0.0
    assert result.frequency_max_relative == 0.0
    assert result.mode_subspaces.global_minimum_singular_value == pytest.approx(1.0)
    assert result.response_max_abs == 0.0
    assert result.response_max_relative == 0.0


def test_mode_subspace_comparison_is_invariant_to_sign_and_degenerate_rotation() -> None:
    reference = np.eye(4)
    candidate = reference.copy()
    candidate[:, 0] *= -1.0
    angle = 0.63
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    candidate[:, 1:3] = reference[:, 1:3] @ rotation
    frequencies = np.array([0.5, 1.0, 1.0, 1.8])
    result = compare_mode_subspaces(reference, candidate, frequencies)
    assert result.clusters == ((0,), (1, 2), (3,))
    assert np.allclose(result.minimum_singular_values, 1.0, atol=1.0e-14)
    assert result.global_minimum_singular_value == pytest.approx(1.0)


def test_backend_comparison_rejects_geometry_mismatch() -> None:
    reference = _synthetic_record("orca-hess")
    candidate = _synthetic_record("pyscf")
    shifted_positions = candidate.positions_bohr.copy()
    shifted_positions[0, 0] += 1.0e-5
    shifted = candidate.__class__(
        symbols=candidate.symbols,
        positions_bohr=shifted_positions,
        masses_amu=candidate.masses_amu,
        hessian_hartree_per_bohr2=candidate.hessian_hartree_per_bohr2,
        source_format=candidate.source_format,
    )
    with pytest.raises(ValueError, match="not identical"):
        compare_hessian_data(reference, shifted)


def test_native_mass_difference_is_reported_but_common_mass_analysis_is_used() -> None:
    reference = _synthetic_record("orca-hess")
    candidate = _synthetic_record("pyscf")
    changed_masses = candidate.masses_amu.copy()
    changed_masses[0] += 0.01
    changed = candidate.__class__(
        symbols=candidate.symbols,
        positions_bohr=candidate.positions_bohr,
        masses_amu=changed_masses,
        hessian_hartree_per_bohr2=candidate.hessian_hartree_per_bohr2,
        source_format=candidate.source_format,
    )
    result = compare_hessian_data(reference, changed, coordinate_pairs=[(0, 1)])
    assert result.mass_max_abs_amu == pytest.approx(0.01)
    assert result.frequency_max_abs == 0.0
    assert result.response_max_abs == 0.0


def test_translated_but_otherwise_identical_geometries_align() -> None:
    """Translated geometries must be accepted after centroid alignment."""
    reference = _synthetic_record("orca-hess")
    candidate = _synthetic_record("pyscf")
    # Apply a translation to the candidate geometry
    translation = np.array([1.5, -2.3, 0.7])
    translated_positions = candidate.positions_bohr + translation
    translated = candidate.__class__(
        symbols=candidate.symbols,
        positions_bohr=translated_positions,
        masses_amu=candidate.masses_amu,
        hessian_hartree_per_bohr2=candidate.hessian_hartree_per_bohr2,
        source_format=candidate.source_format,
    )
    result = compare_hessian_data(
        reference,
        translated,
        coordinate_pairs=[(0, 1), (0, 2), (1, 2)],
    )
    # Unaligned difference should be large
    assert result.geometry_max_abs_bohr > 1.0
    # Translation should be reported (ref_centroid - cand_centroid = -translation)
    assert result.translation_removed_bohr is not None
    assert np.allclose(result.translation_removed_bohr, -translation)
    # Aligned residual should be zero (same geometry, just translated)
    assert result.geometry_max_abs_aligned_bohr == pytest.approx(0.0, abs=1e-14)
    # All physical comparisons should be exact (same Hessian, same masses)
    assert result.hessian_max_abs_hartree_per_bohr2 == 0.0
    assert result.frequency_max_abs == 0.0
    assert result.response_max_abs == 0.0
