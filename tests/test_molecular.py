from __future__ import annotations

import numpy as np

from dirs.coordinates import distance_wilson_vector, mass_weight_wilson
from dirs.molecular import (
    analyze_molecular_hessian,
    local_coordinate_response,
    local_response_matrix,
    modal_response_contributions,
    vibrational_pseudoinverse,
)
from dirs.rigid import expanded_masses, mass_weighted_rigid_basis


def _synthetic_hessian(
    positions: np.ndarray,
    masses: np.ndarray,
    eigenvalues: np.ndarray | None = None,
    seed: int = 7,
) -> np.ndarray:
    rigid = mass_weighted_rigid_basis(positions, masses)
    z = rigid.vibrational_basis
    n_vib = z.shape[1]
    if eigenvalues is None:
        eigenvalues = np.linspace(0.2, 1.7, n_vib)
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    assert eigenvalues.shape == (n_vib,)
    rng = np.random.default_rng(seed)
    random = rng.normal(size=(n_vib, n_vib))
    u, _ = np.linalg.qr(random)
    d_reduced = u @ np.diag(eigenvalues) @ u.T
    d_full = z @ d_reduced @ z.T
    sqrt_mass = np.sqrt(expanded_masses(masses))
    return sqrt_mass[:, None] * d_full * sqrt_mass[None, :]


def _nonlinear_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [-0.3, 1.1, 0.2]],
        dtype=float,
    )
    masses = np.array([16.0, 1.0, 1.0])
    hessian = _synthetic_hessian(positions, masses)
    return positions, masses, hessian


def _rotation_matrix() -> np.ndarray:
    axis = np.array([1.0, -2.0, 0.5])
    axis /= np.linalg.norm(axis)
    angle = 0.73
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (
        cross @ cross
    )


def _block_rotation(rotation: np.ndarray, n_atoms: int) -> np.ndarray:
    return np.kron(np.eye(n_atoms), rotation)


def _permutation_matrix(permutation: np.ndarray) -> np.ndarray:
    n_atoms = len(permutation)
    matrix = np.zeros((3 * n_atoms, 3 * n_atoms))
    for new_index, old_index in enumerate(permutation):
        matrix[
            3 * new_index : 3 * new_index + 3,
            3 * old_index : 3 * old_index + 3,
        ] = np.eye(3)
    return matrix


def test_linear_and_nonlinear_rigid_mode_counts() -> None:
    linear_positions = np.array([[-0.7, 0.0, 0.0], [0.7, 0.0, 0.0]])
    linear_masses = np.array([1.0, 19.0])
    linear = mass_weighted_rigid_basis(linear_positions, linear_masses)
    assert linear.is_linear
    assert linear.n_rotations == 2
    assert linear.rigid_basis.shape == (6, 5)
    assert linear.vibrational_basis.shape == (6, 1)

    positions, masses, _ = _nonlinear_model()
    nonlinear = mass_weighted_rigid_basis(positions, masses)
    assert not nonlinear.is_linear
    assert nonlinear.n_rotations == 3
    assert nonlinear.rigid_basis.shape == (9, 6)
    assert nonlinear.vibrational_basis.shape == (9, 3)


def test_distance_wilson_is_rigid_motion_free() -> None:
    positions, masses, _ = _nonlinear_model()
    rigid = mass_weighted_rigid_basis(positions, masses)
    wilson = distance_wilson_vector(positions, 0, 1)
    weighted = mass_weight_wilson(wilson, masses)
    assert np.linalg.norm(rigid.rigid_basis.T @ weighted) < 1.0e-12


def test_direct_and_modal_responses_agree() -> None:
    positions, masses, hessian = _nonlinear_model()
    analysis = analyze_molecular_hessian(positions, masses, hessian)
    a = distance_wilson_vector(positions, 0, 1)
    b = distance_wilson_vector(positions, 0, 2)

    for omega, epsilon in [(0.0, 0.0), (0.37, 1.0e-4)]:
        modal = local_coordinate_response(
            analysis,
            a,
            b,
            omega=omega,
            epsilon=epsilon,
            method="modal",
        )
        direct = local_coordinate_response(
            analysis,
            a,
            b,
            omega=omega,
            epsilon=epsilon,
            method="direct",
        )
        assert np.allclose(modal, direct, rtol=1.0e-11, atol=1.0e-12)

    contributions = modal_response_contributions(analysis, a, b)
    assert np.allclose(
        np.sum(contributions),
        local_coordinate_response(analysis, a, b),
    )


def test_vibrational_pseudoinverse_matches_static_response() -> None:
    positions, masses, hessian = _nonlinear_model()
    analysis = analyze_molecular_hessian(positions, masses, hessian)
    a = distance_wilson_vector(positions, 0, 1)
    b = distance_wilson_vector(positions, 1, 2)
    weighted_a = mass_weight_wilson(a, masses)
    weighted_b = mass_weight_wilson(b, masses)
    pseudoinverse = vibrational_pseudoinverse(analysis)
    direct = weighted_b @ pseudoinverse @ weighted_a
    modal = local_coordinate_response(analysis, a, b)
    numerical = np.linalg.pinv(
        analysis.projected_hessian,
        rcond=1.0e-12,
        hermitian=True,
    )
    assert np.allclose(pseudoinverse, numerical, rtol=1.0e-11, atol=1.0e-11)
    assert np.allclose(direct, modal.real, rtol=1.0e-12, atol=1.0e-12)


def test_rigid_translation_invariance() -> None:
    positions, masses, hessian = _nonlinear_model()
    shifted = positions + np.array([3.7, -2.1, 0.9])
    original = analyze_molecular_hessian(positions, masses, hessian)
    translated = analyze_molecular_hessian(shifted, masses, hessian)
    original_a = distance_wilson_vector(positions, 0, 1)
    original_b = distance_wilson_vector(positions, 1, 2)
    shifted_a = distance_wilson_vector(shifted, 0, 1)
    shifted_b = distance_wilson_vector(shifted, 1, 2)
    assert np.allclose(
        local_coordinate_response(original, original_a, original_b),
        local_coordinate_response(translated, shifted_a, shifted_b),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_rigid_rotation_invariance() -> None:
    positions, masses, hessian = _nonlinear_model()
    rotation = _rotation_matrix()
    block = _block_rotation(rotation, len(masses))
    rotated_positions = positions @ rotation.T
    rotated_hessian = block @ hessian @ block.T
    original = analyze_molecular_hessian(positions, masses, hessian)
    rotated = analyze_molecular_hessian(
        rotated_positions,
        masses,
        rotated_hessian,
    )
    a = distance_wilson_vector(positions, 0, 1)
    b = distance_wilson_vector(positions, 0, 2)
    rotated_a = distance_wilson_vector(rotated_positions, 0, 1)
    rotated_b = distance_wilson_vector(rotated_positions, 0, 2)
    assert np.allclose(
        local_coordinate_response(
            original,
            a,
            b,
            omega=0.41,
            epsilon=1.0e-4,
        ),
        local_coordinate_response(
            rotated,
            rotated_a,
            rotated_b,
            omega=0.41,
            epsilon=1.0e-4,
        ),
        rtol=1.0e-10,
        atol=1.0e-11,
    )


def test_atom_permutation_invariance() -> None:
    positions, masses, hessian = _nonlinear_model()
    permutation = np.array([2, 0, 1])
    matrix = _permutation_matrix(permutation)
    permuted_positions = positions[permutation]
    permuted_masses = masses[permutation]
    permuted_hessian = matrix @ hessian @ matrix.T
    original = analyze_molecular_hessian(positions, masses, hessian)
    permuted = analyze_molecular_hessian(
        permuted_positions,
        permuted_masses,
        permuted_hessian,
    )

    original_coordinates = [
        distance_wilson_vector(positions, 0, 1),
        distance_wilson_vector(positions, 0, 2),
    ]
    permuted_coordinates = [
        distance_wilson_vector(permuted_positions, 1, 2),
        distance_wilson_vector(permuted_positions, 1, 0),
    ]
    assert np.allclose(
        local_response_matrix(original, original_coordinates),
        local_response_matrix(permuted, permuted_coordinates),
        rtol=1.0e-11,
        atol=1.0e-12,
    )


def test_coordinate_orientation_covariance() -> None:
    positions, masses, hessian = _nonlinear_model()
    analysis = analyze_molecular_hessian(positions, masses, hessian)
    a = distance_wilson_vector(positions, 0, 1)
    reversed_a = distance_wilson_vector(
        positions,
        0,
        1,
        orientation=-1.0,
    )
    b = distance_wilson_vector(positions, 0, 2)
    cross = local_coordinate_response(analysis, a, b)
    reversed_cross = local_coordinate_response(analysis, reversed_a, b)
    self_response = local_coordinate_response(analysis, a, a)
    reversed_self = local_coordinate_response(analysis, reversed_a, reversed_a)
    assert np.allclose(reversed_cross, -cross, rtol=0.0, atol=1.0e-12)
    assert np.allclose(reversed_self, self_response, rtol=0.0, atol=1.0e-12)


def test_weak_low_frequency_mode_is_not_dropped() -> None:
    positions, masses, _ = _nonlinear_model()
    low_eigenvalue = 1.0e-10
    hessian = _synthetic_hessian(
        positions,
        masses,
        eigenvalues=np.array([low_eigenvalue, 0.7, 1.4]),
        seed=11,
    )
    analysis = analyze_molecular_hessian(positions, masses, hessian)
    assert np.isclose(
        analysis.eigenvalues[0],
        low_eigenvalue,
        rtol=5.0e-6,
        atol=1.0e-15,
    )
    a = distance_wilson_vector(positions, 0, 1)
    response = local_coordinate_response(analysis, a, a)
    assert np.isfinite(response.real)
    assert response.real > 0.0


def test_distance_atom_order_does_not_change_unsigned_coordinate() -> None:
    positions, _, _ = _nonlinear_model()
    forward = distance_wilson_vector(positions, 0, 1)
    reverse = distance_wilson_vector(positions, 1, 0)
    assert np.allclose(forward, reverse, rtol=0.0, atol=0.0)
