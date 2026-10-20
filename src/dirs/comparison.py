"""Cross-backend Hessian and DIRS comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from .backends import HessianData
from .coordinates import distance_wilson_vector
from .molecular import analyze_molecular_hessian, local_response_matrix

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
DistancePair = tuple[int, int]


@dataclass(frozen=True)
class ModeSubspaceResult:
    """Sign- and degeneracy-robust comparison of vibrational eigenspaces."""

    clusters: tuple[tuple[int, ...], ...]
    minimum_singular_values: FloatArray
    mean_singular_values: FloatArray
    global_minimum_singular_value: float


@dataclass(frozen=True)
class BackendComparison:
    """Traceable comparison between two backend-neutral Hessian records."""

    reference_format: str
    candidate_format: str
    geometry_max_abs_bohr: float
    translation_removed_bohr: FloatArray | None
    geometry_max_abs_aligned_bohr: float
    mass_max_abs_amu: float
    hessian_max_abs_hartree_per_bohr2: float
    hessian_rms_hartree_per_bohr2: float
    hessian_relative_frobenius: float
    reference_angular_frequencies: FloatArray
    candidate_angular_frequencies: FloatArray
    frequency_max_abs: float
    frequency_max_relative: float
    mode_subspaces: ModeSubspaceResult
    coordinate_pairs: tuple[DistancePair, ...]
    reference_response: ComplexArray | None
    candidate_response: ComplexArray | None
    response_max_abs: float | None
    response_max_relative: float | None


def _relative_scalar_error(value: float, target: float) -> float:
    denominator = abs(target)
    return abs(value - target) / denominator if denominator > 0.0 else abs(value - target)


def _cluster_sorted_values(
    values: FloatArray,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a nonempty one-dimensional array.")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError("Clustering tolerances must be non-negative.")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("values must be sorted in ascending order.")

    clusters: list[list[int]] = [[0]]
    for index in range(1, values.size):
        previous = values[index - 1]
        current = values[index]
        threshold = absolute_tolerance + relative_tolerance * max(
            abs(previous), abs(current)
        )
        if abs(current - previous) <= threshold:
            clusters[-1].append(index)
        else:
            clusters.append([index])
    return tuple(tuple(cluster) for cluster in clusters)


def compare_mode_subspaces(
    reference_modes: FloatArray,
    candidate_modes: FloatArray,
    reference_frequencies: FloatArray,
    *,
    absolute_tolerance: float = 1.0e-10,
    relative_tolerance: float = 1.0e-7,
) -> ModeSubspaceResult:
    """Compare corresponding mode clusters through principal-angle singular values."""
    reference_modes = np.asarray(reference_modes, dtype=np.float64)
    candidate_modes = np.asarray(candidate_modes, dtype=np.float64)
    frequencies = np.asarray(reference_frequencies, dtype=np.float64)
    if reference_modes.shape != candidate_modes.shape:
        raise ValueError("Reference and candidate mode matrices must have equal shape.")
    if reference_modes.ndim != 2 or reference_modes.shape[1] != frequencies.size:
        raise ValueError("Mode columns must correspond to reference frequencies.")

    clusters = _cluster_sorted_values(
        frequencies,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    minima: list[float] = []
    means: list[float] = []
    for cluster in clusters:
        indices = np.asarray(cluster, dtype=int)
        overlap = reference_modes[:, indices].T @ candidate_modes[:, indices]
        singular_values = np.linalg.svd(overlap, compute_uv=False)
        minima.append(float(np.min(singular_values)))
        means.append(float(np.mean(singular_values)))
    minimum_array = np.asarray(minima, dtype=np.float64)
    mean_array = np.asarray(means, dtype=np.float64)
    return ModeSubspaceResult(
        clusters=clusters,
        minimum_singular_values=minimum_array,
        mean_singular_values=mean_array,
        global_minimum_singular_value=float(np.min(minimum_array)),
    )


def compare_hessian_data(
    reference: HessianData,
    candidate: HessianData,
    *,
    coordinate_pairs: Iterable[DistancePair] = (),
    omega: float = 0.0,
    epsilon: float = 0.0,
    geometry_tolerance_bohr: float = 1.0e-10,
    mode_absolute_tolerance: float = 1.0e-10,
    mode_relative_tolerance: float = 1.0e-7,
    imaginary_tolerance: float = 1.0e-8,
) -> BackendComparison:
    """Compare two records at identical atom order, geometry, and reference masses.

    Native mass differences are reported, but both vibrational analyses use the
    reference masses. This isolates electronic Hessian differences from backend
    choices of default isotope or standard atomic weights.
    """
    if reference.symbols != candidate.symbols:
        raise ValueError("Backend records must use exactly the same atom order and symbols.")
    if reference.positions_bohr.shape != candidate.positions_bohr.shape:
        raise ValueError("Backend records have incompatible geometry shapes.")
    if reference.hessian_hartree_per_bohr2.shape != (
        candidate.hessian_hartree_per_bohr2.shape
    ):
        raise ValueError("Backend records have incompatible Hessian shapes.")
    if geometry_tolerance_bohr <= 0.0:
        raise ValueError("geometry_tolerance_bohr must be strictly positive.")

    geometry_max_abs = float(
        np.max(np.abs(reference.positions_bohr - candidate.positions_bohr))
    )
    translation = None
    geometry_max_abs_aligned = geometry_max_abs
    if geometry_max_abs > geometry_tolerance_bohr:
        # ORCA may translate the molecule; align arithmetic centroids
        ref_com = np.mean(reference.positions_bohr, axis=0)
        cand_com = np.mean(candidate.positions_bohr, axis=0)
        translation = ref_com - cand_com
        ref_centered = reference.positions_bohr - ref_com
        cand_centered = candidate.positions_bohr - cand_com
        geometry_max_abs_aligned = float(np.max(np.abs(ref_centered - cand_centered)))
        if geometry_max_abs_aligned > geometry_tolerance_bohr:
            raise ValueError(
                "Backend geometries are not identical within the declared tolerance: "
                f"unaligned max {geometry_max_abs:.6e} bohr, "
                f"translation {np.linalg.norm(translation):.6e} bohr, "
                f"aligned residual {geometry_max_abs_aligned:.6e} bohr."
            )
    mass_max_abs = float(np.max(np.abs(reference.masses_amu - candidate.masses_amu)))

    hessian_difference = (
        candidate.hessian_hartree_per_bohr2
        - reference.hessian_hartree_per_bohr2
    )
    hessian_max_abs = float(np.max(np.abs(hessian_difference)))
    hessian_rms = float(np.sqrt(np.mean(hessian_difference**2)))
    reference_norm = float(np.linalg.norm(reference.hessian_hartree_per_bohr2))
    hessian_relative = (
        float(np.linalg.norm(hessian_difference)) / reference_norm
        if reference_norm > 0.0
        else float(np.linalg.norm(hessian_difference))
    )

    common_positions = reference.positions_bohr
    common_masses = reference.masses_electron
    reference_analysis = analyze_molecular_hessian(
        common_positions,
        common_masses,
        reference.hessian_hartree_per_bohr2,
        imaginary_tolerance=imaginary_tolerance,
    )
    candidate_analysis = analyze_molecular_hessian(
        common_positions,
        common_masses,
        candidate.hessian_hartree_per_bohr2,
        imaginary_tolerance=imaginary_tolerance,
    )
    if reference_analysis.angular_frequencies.shape != (
        candidate_analysis.angular_frequencies.shape
    ):
        raise ValueError("Backend analyses produced different vibrational dimensions.")

    frequency_difference = np.abs(
        candidate_analysis.angular_frequencies
        - reference_analysis.angular_frequencies
    )
    frequency_max_abs = float(np.max(frequency_difference))
    frequency_relative = np.divide(
        frequency_difference,
        np.abs(reference_analysis.angular_frequencies),
        out=frequency_difference.copy(),
        where=np.abs(reference_analysis.angular_frequencies) > 0.0,
    )
    frequency_max_relative = float(np.max(frequency_relative))

    mode_subspaces = compare_mode_subspaces(
        reference_analysis.modes,
        candidate_analysis.modes,
        reference_analysis.angular_frequencies,
        absolute_tolerance=mode_absolute_tolerance,
        relative_tolerance=mode_relative_tolerance,
    )

    pairs = tuple((int(i), int(j)) for i, j in coordinate_pairs)
    if pairs:
        wilson_vectors = [
            distance_wilson_vector(common_positions, atom_i, atom_j)
            for atom_i, atom_j in pairs
        ]
        reference_response = local_response_matrix(
            reference_analysis,
            wilson_vectors,
            omega=omega,
            epsilon=epsilon,
        )
        candidate_response = local_response_matrix(
            candidate_analysis,
            wilson_vectors,
            omega=omega,
            epsilon=epsilon,
        )
        response_difference = np.abs(candidate_response - reference_response)
        response_max_abs = float(np.max(response_difference))
        reference_response_scale = float(np.max(np.abs(reference_response)))
        response_max_relative = (
            response_max_abs / reference_response_scale
            if reference_response_scale > 0.0
            else response_max_abs
        )
    else:
        reference_response = None
        candidate_response = None
        response_max_abs = None
        response_max_relative = None

    return BackendComparison(
        reference_format=reference.source_format,
        candidate_format=candidate.source_format,
        geometry_max_abs_bohr=geometry_max_abs,
        translation_removed_bohr=(
            translation.copy() if translation is not None else None
        ),
        geometry_max_abs_aligned_bohr=geometry_max_abs_aligned,
        mass_max_abs_amu=mass_max_abs,
        hessian_max_abs_hartree_per_bohr2=hessian_max_abs,
        hessian_rms_hartree_per_bohr2=hessian_rms,
        hessian_relative_frobenius=hessian_relative,
        reference_angular_frequencies=reference_analysis.angular_frequencies,
        candidate_angular_frequencies=candidate_analysis.angular_frequencies,
        frequency_max_abs=frequency_max_abs,
        frequency_max_relative=frequency_max_relative,
        mode_subspaces=mode_subspaces,
        coordinate_pairs=pairs,
        reference_response=reference_response,
        candidate_response=candidate_response,
        response_max_abs=response_max_abs,
        response_max_relative=response_max_relative,
    )
