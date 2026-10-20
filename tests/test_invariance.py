from __future__ import annotations

import numpy as np

from dirs.coordinates import distance_wilson_vector
from dirs.invariance import enforce_cartesian_rigid_invariance
from dirs.molecular import analyze_molecular_hessian, local_response_matrix
from dirs.rigid import mass_weighted_rigid_basis


def _near_linear_triatomic() -> np.ndarray:
    return np.asarray(
        [
            [-1.14, -1.0e-10, -4.0e-9],
            [0.0, 2.0e-10, 8.0e-9],
            [1.14, -1.0e-10, -4.0e-9],
        ],
        dtype=np.float64,
    )


def _positive_geometric_hessian(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    geometric = mass_weighted_rigid_basis(
        positions,
        np.ones(positions.shape[0]),
        linearity_tolerance=1.0e-8,
    )
    assert geometric.is_linear
    assert geometric.rigid_basis.shape[1] == 5
    z = geometric.vibrational_basis
    eigenvalues = np.asarray([0.08, 0.12, 0.18, 0.25], dtype=np.float64)
    hessian = (z * eigenvalues[None, :]) @ z.T
    return hessian, geometric.rigid_basis, z


def test_cartesian_rigid_projection_removes_contamination() -> None:
    positions = _near_linear_triatomic()
    physical, q, z = _positive_geometric_hessian(positions)

    rigid_block = np.diag(np.linspace(1.0, 2.0, q.shape[1])) * 2.0e-5
    coupling = np.arange(q.shape[1] * z.shape[1], dtype=float).reshape(q.shape[1], z.shape[1])
    coupling = (coupling + 1.0) * 1.0e-6
    contaminated = physical + q @ rigid_block @ q.T + q @ coupling @ z.T + z @ coupling.T @ q.T
    contaminated = 0.5 * (contaminated + contaminated.T)

    result = enforce_cartesian_rigid_invariance(
        positions,
        contaminated,
        linearity_tolerance=1.0e-8,
    )

    assert result.is_linear
    assert result.rigid_rank == 5
    assert result.raw_rigid_block_norm > 0.0
    assert result.raw_rigid_vibrational_coupling_norm > 0.0
    np.testing.assert_allclose(result.rigid_basis.T @ result.hessian, 0.0, atol=2.0e-16)
    np.testing.assert_allclose(result.hessian @ result.rigid_basis, 0.0, atol=2.0e-16)
    np.testing.assert_allclose(result.hessian, physical, atol=3.0e-16)


def test_exactly_invariant_hessian_is_unchanged() -> None:
    positions = _near_linear_triatomic()
    physical, _, _ = _positive_geometric_hessian(positions)
    result = enforce_cartesian_rigid_invariance(
        positions,
        physical,
        linearity_tolerance=1.0e-8,
    )
    np.testing.assert_allclose(result.hessian, physical, atol=3.0e-16)
    assert result.correction_relative_frobenius < 1.0e-14


def test_mass_only_static_response_is_invariant_after_projection() -> None:
    positions = _near_linear_triatomic()
    physical, q, z = _positive_geometric_hessian(positions)
    coupling = np.full((q.shape[1], z.shape[1]), 2.0e-6)
    contaminated = physical + q @ coupling @ z.T + z @ coupling.T @ q.T
    corrected = enforce_cartesian_rigid_invariance(
        positions,
        contaminated,
        linearity_tolerance=1.0e-8,
    ).hessian

    wilsons = [
        distance_wilson_vector(positions, 0, 1),
        distance_wilson_vector(positions, 1, 2),
        distance_wilson_vector(positions, 0, 2),
    ]
    hydrogen_masses = np.asarray([19.0, 1.008, 19.0])
    deuterium_masses = np.asarray([19.0, 2.01410177812, 19.0])

    analysis_h = analyze_molecular_hessian(
        positions,
        hydrogen_masses,
        corrected,
        linearity_tolerance=1.0e-8,
    )
    analysis_d = analyze_molecular_hessian(
        positions,
        deuterium_masses,
        corrected,
        linearity_tolerance=1.0e-8,
    )
    chi_h = local_response_matrix(analysis_h, wilsons, method="direct").real
    chi_d = local_response_matrix(analysis_d, wilsons, method="direct").real

    np.testing.assert_allclose(chi_h, chi_d, atol=2.0e-12, rtol=2.0e-12)
    assert not np.allclose(
        analysis_h.angular_frequencies,
        analysis_d.angular_frequencies,
        atol=1.0e-12,
        rtol=1.0e-8,
    )
