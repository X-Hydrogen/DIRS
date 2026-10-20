from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dirs.archive import load_hessian_npz
from dirs.coordinates import distance_wilson_vector, mass_weight_wilson
from dirs.invariance import enforce_cartesian_rigid_invariance
from dirs.rigid import expanded_masses, mass_weighted_rigid_basis
from dirs.units import HARTREE_WAVENUMBER_CM, amu_to_electron_masses

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "stage3a_outputs"
OUT.mkdir(parents=True, exist_ok=True)

STAGE2 = ROOT / "calculations" / "prototypes_stage2"
STAGE3 = ROOT / "calculations" / "benchmark_stage3"
AROMATIC_R2 = ROOT / "calculations" / "benchmark_stage3_aromatic_r2"

GAMMAS_CM = (5.0, 20.0, 50.0)
GRID_STEPS_CM = (1.0, 0.5, 0.25)
DEUTERIUM_MASS_AMU = 2.01410177812
LINEARITY_TOLERANCE = 1.0e-8

# Numerical gates are declared before this analysis is executed.
DIRECT_MODAL_REL_TOL = 1.0e-8
RECIPROCITY_ABS_TOL = 1.0e-10
ETA_BOUND_TOL = 1.0e-10
SPECTRAL_SUM_ABS_TOL = 2.0e-4
SPECTRAL_REFINEMENT_ABS_TOL = 5.0e-5
ISOTOPE_STATIC_REL_TOL = 1.0e-10

ACCEPTED_SOURCES = {
    "water_dimer": STAGE2 / "results" / "compact" / "water_dimer.npz",
    "methane_dimer": STAGE2 / "results" / "compact" / "methane_dimer.npz",
    "fhf_minus": STAGE2 / "results" / "compact" / "fhf_minus.npz",
    "hf_dimer": STAGE3 / "results" / "compact" / "hf_dimer.npz",
    "hcn_dimer": STAGE3 / "results" / "compact" / "hcn_dimer.npz",
    "nh3_hf": STAGE3 / "results" / "compact" / "nh3_hf.npz",
    "chloride_water": STAGE3 / "results" / "compact" / "chloride_water.npz",
    "fluorobenzene_fluoroform": AROMATIC_R2 / "results" / "compact" / "fluorobenzene_fluoroform__n3_pmp.npz",
    "h5o2_plus": STAGE3 / "results" / "compact" / "h5o2_plus.npz",
    "ammonia_borane": STAGE3 / "results" / "compact" / "ammonia_borane.npz",
}

STAGE2_NAMES = {"water_dimer", "methane_dimer", "fhf_minus"}


@dataclass
class HarmonicSystem:
    name: str
    symbols: tuple[str, ...]
    positions_bohr: np.ndarray
    masses_amu: np.ndarray
    masses_electron: np.ndarray
    raw_hessian: np.ndarray
    hessian: np.ndarray
    rigid_rank: int
    is_linear: bool
    z: np.ndarray
    reduced_hessian: np.ndarray
    eigenvalues: np.ndarray
    frequencies_cm_signed: np.ndarray
    modes: np.ndarray
    invariance: dict[str, float | int | bool]


def relative_error(value: float, target: float, floor: float = 1.0e-30) -> float:
    return abs(value - target) / max(abs(target), floor)


def load_coordinate_maps() -> dict[str, dict[str, Any]]:
    s2 = json.loads((STAGE2 / "local_coordinates.json").read_text(encoding="utf-8"))["systems"]
    s3 = json.loads((STAGE3 / "local_coordinates.json").read_text(encoding="utf-8"))["systems"]
    merged = {**s3, **s2}
    expected = set(ACCEPTED_SOURCES)
    if set(merged).issuperset(expected) is False:
        raise RuntimeError("local-coordinate maps do not cover the accepted ten-system set")
    return merged


def analyse_hessian(name: str, masses_override_amu: np.ndarray | None = None) -> HarmonicSystem:
    source = ACCEPTED_SOURCES[name]
    if not source.is_file():
        raise FileNotFoundError(source)
    rec = load_hessian_npz(source)
    positions = np.asarray(rec.positions_bohr, dtype=float)
    masses_amu = rec.masses_amu.copy() if masses_override_amu is None else np.asarray(masses_override_amu, dtype=float)
    masses_e = amu_to_electron_masses(masses_amu)
    raw = 0.5 * (rec.hessian_hartree_per_bohr2 + rec.hessian_hartree_per_bohr2.T)

    invariant = enforce_cartesian_rigid_invariance(
        positions,
        raw,
        linearity_tolerance=LINEARITY_TOLERANCE,
    )
    hessian = invariant.hessian

    rigid = mass_weighted_rigid_basis(
        positions,
        masses_e,
        linearity_tolerance=LINEARITY_TOLERANCE,
    )
    mass_vector = expanded_masses(masses_e)
    inv_sqrt_mass = 1.0 / np.sqrt(mass_vector)
    dyn = inv_sqrt_mass[:, None] * hessian * inv_sqrt_mass[None, :]
    dyn = 0.5 * (dyn + dyn.T)
    z = rigid.vibrational_basis
    reduced = z.T @ dyn @ z
    reduced = 0.5 * (reduced + reduced.T)
    eig, reduced_modes = np.linalg.eigh(reduced)
    frequencies_cm = np.sign(eig) * np.sqrt(np.abs(eig)) * HARTREE_WAVENUMBER_CM
    modes = z @ reduced_modes

    return HarmonicSystem(
        name=name,
        symbols=rec.symbols,
        positions_bohr=positions,
        masses_amu=masses_amu,
        masses_electron=masses_e,
        raw_hessian=raw,
        hessian=hessian,
        rigid_rank=rigid.rigid_basis.shape[1],
        is_linear=rigid.is_linear,
        z=z,
        reduced_hessian=reduced,
        eigenvalues=eig,
        frequencies_cm_signed=frequencies_cm,
        modes=modes,
        invariance={
            "geometric_rigid_rank": invariant.rigid_rank,
            "geometric_is_linear": invariant.is_linear,
            "raw_rigid_block_norm": invariant.raw_rigid_block_norm,
            "raw_rigid_vibrational_coupling_norm": invariant.raw_rigid_vibrational_coupling_norm,
            "correction_max_abs_hartree_per_bohr2": invariant.correction_max_abs,
            "correction_frobenius": invariant.correction_frobenius,
            "correction_relative_frobenius": invariant.correction_relative_frobenius,
        },
    )


def coordinate_data(system: HarmonicSystem, definitions: dict[str, list[int]]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = list(definitions)
    wilsons = [
        distance_wilson_vector(
            system.positions_bohr,
            int(definitions[name][0]),
            int(definitions[name][1]),
        )
        for name in names
    ]
    mw = np.column_stack(
        [mass_weight_wilson(w, system.masses_electron) for w in wilsons]
    )
    reduced_vectors = system.z.T @ mw
    amplitudes = system.modes.T @ mw
    lengths_bohr = np.asarray(
        [
            np.linalg.norm(
                system.positions_bohr[int(definitions[name][0])]
                - system.positions_bohr[int(definitions[name][1])]
            )
            for name in names
        ],
        dtype=float,
    )
    return names, reduced_vectors, amplitudes, lengths_bohr


def static_matrices(system: HarmonicSystem, reduced_vectors: np.ndarray, amplitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if np.any(system.eigenvalues <= 0.0):
        raise ValueError(f"{system.name}: non-positive projected vibrational eigenvalue")
    direct = reduced_vectors.T @ np.linalg.solve(system.reduced_hessian, reduced_vectors)
    modal = amplitudes.T @ ((1.0 / system.eigenvalues)[:, None] * amplitudes)
    denom = np.sqrt(np.diag(modal)[:, None] * np.diag(modal)[None, :])
    eta = modal / denom
    return direct, modal, eta


def raw_projected_static(system: HarmonicSystem, definitions: dict[str, list[int]]) -> tuple[list[str], np.ndarray] | None:
    masses = system.masses_electron
    rigid = mass_weighted_rigid_basis(
        system.positions_bohr,
        masses,
        linearity_tolerance=LINEARITY_TOLERANCE,
    )
    inv_sqrt = 1.0 / np.sqrt(expanded_masses(masses))
    dyn = inv_sqrt[:, None] * system.raw_hessian * inv_sqrt[None, :]
    dyn = 0.5 * (dyn + dyn.T)
    z = rigid.vibrational_basis
    red = z.T @ dyn @ z
    red = 0.5 * (red + red.T)
    eig = np.linalg.eigvalsh(red)
    if np.any(eig <= 0.0):
        return None
    names = list(definitions)
    mw = np.column_stack(
        [
            mass_weight_wilson(
                distance_wilson_vector(
                    system.positions_bohr,
                    int(definitions[name][0]),
                    int(definitions[name][1]),
                ),
                masses,
            )
            for name in names
        ]
    )
    a = z.T @ mw
    chi = a.T @ np.linalg.solve(red, a)
    denom = np.sqrt(np.diag(chi)[:, None] * np.diag(chi)[None, :])
    return names, chi / denom


def pair_metrics(system: HarmonicSystem, names: list[str], amplitudes: np.ndarray, chi: np.ndarray, eta: np.ndarray, left: str, right: str, pair_role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    i = names.index(left)
    j = names.index(right)
    weights = amplitudes[:, i] * amplitudes[:, j] / system.eigenvalues
    norm = math.sqrt(float(chi[i, i] * chi[j, j]))
    normalized_weights = weights / norm
    activity = float(np.sum(np.abs(normalized_weights)))
    kappa = abs(float(eta[i, j])) / activity if activity > 0.0 else 0.0
    abs_activity = np.abs(normalized_weights)
    total = float(abs_activity.sum())
    freq = system.frequencies_cm_signed

    def fraction(mask: np.ndarray) -> float:
        return float(abs_activity[mask].sum() / total) if total > 0.0 else 0.0

    row = {
        "system": system.name,
        "pair_role": pair_role,
        "coordinate_a": left,
        "coordinate_b": right,
        "chi_ab_bohr2_per_hartree": float(chi[i, j]),
        "eta": float(eta[i, j]),
        "absolute_modal_activity_A": activity,
        "coherence_kappa": kappa,
        "abs_activity_below_200_cm_fraction": fraction(freq < 200.0),
        "abs_activity_200_1000_cm_fraction": fraction((freq >= 200.0) & (freq < 1000.0)),
        "abs_activity_above_1000_cm_fraction": fraction(freq >= 1000.0),
    }
    order = np.argsort(np.abs(normalized_weights))[::-1]
    modes: list[dict[str, Any]] = []
    for rank, index in enumerate(order[: min(10, len(order))], start=1):
        modes.append({
            "system": system.name,
            "pair_role": pair_role,
            "coordinate_a": left,
            "coordinate_b": right,
            "rank": rank,
            "mode_index_zero_based": int(index),
            "frequency_cm_minus1": float(freq[index]),
            "normalized_static_modal_weight": float(normalized_weights[index]),
            "fraction_of_absolute_activity": float(abs_activity[index] / total) if total > 0.0 else 0.0,
        })
    return row, modes


def normalized_spectrum(system: HarmonicSystem, amplitudes: np.ndarray, chi: np.ndarray, i: int, j: int, grid_cm: np.ndarray, gamma_cm: float) -> np.ndarray:
    omega = grid_cm / HARTREE_WAVENUMBER_CM
    gamma = gamma_cm / HARTREE_WAVENUMBER_CM
    norm = math.sqrt(float(chi[i, i] * chi[j, j]))
    coefficients = amplitudes[:, i] * amplitudes[:, j] / norm
    eig = system.eigenvalues[:, None]
    w = omega[None, :]
    denominator = (eig - w**2) ** 2 + (2.0 * gamma * w) ** 2
    d_au = np.sum((4.0 * gamma / math.pi) * coefficients[:, None] / denominator, axis=0)
    return d_au / HARTREE_WAVENUMBER_CM


def spectral_audit(system: HarmonicSystem, names: list[str], amplitudes: np.ndarray, chi: np.ndarray, eta: np.ndarray, pairs: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    upper = max(10000.0, 3.0 * float(np.max(system.frequencies_cm_signed)))
    for left, right in pairs:
        i = names.index(left)
        j = names.index(right)
        target = float(eta[i, j])
        for gamma_cm in GAMMAS_CM:
            integrals: dict[str, float] = {}
            for step in GRID_STEPS_CM:
                grid = np.arange(0.0, upper + 0.5 * step, step)
                spectrum = normalized_spectrum(system, amplitudes, chi, i, j, grid, gamma_cm)
                integrals[f"step_{step:g}_cm"] = float(np.trapezoid(spectrum, grid))
            finest = integrals["step_0.25_cm"]
            rows.append({
                "system": system.name,
                "coordinate_a": left,
                "coordinate_b": right,
                "gamma_cm_minus1": gamma_cm,
                "upper_limit_cm_minus1": upper,
                "eta_target": target,
                "integrals": integrals,
                "finest_absolute_error": abs(finest - target),
                "refinement_difference_1_to_0p5": abs(integrals["step_1_cm"] - integrals["step_0.5_cm"]),
                "refinement_difference_0p5_to_0p25": abs(integrals["step_0.5_cm"] - finest),
            })
    return rows


def equivalence_audit(system: HarmonicSystem, names: list[str], lengths: np.ndarray, chi: np.ndarray, groups: list[list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        indices = [names.index(name) for name in group]
        group_lengths = lengths[indices] / 1.8897261254578281
        self_chi = np.diag(chi)[indices]
        rows.append({
            "system": system.name,
            "group_index": group_index,
            "members": group,
            "lengths_angstrom": [float(x) for x in group_lengths],
            "length_span_angstrom": float(np.max(group_lengths) - np.min(group_lengths)),
            "self_chi_bohr2_per_hartree": [float(x) for x in self_chi],
            "self_chi_relative_span": float((np.max(self_chi) - np.min(self_chi)) / max(float(np.mean(np.abs(self_chi))), 1e-30)),
            "averaging_performed": False,
        })
    return rows


def isotope_audit(name: str, atom_index: int, definitions: dict[str, list[int]], pairs: list[tuple[str, str]]) -> dict[str, Any]:
    reference = analyse_hessian(name)
    masses_d = reference.masses_amu.copy()
    masses_d[atom_index] = DEUTERIUM_MASS_AMU
    isotope = analyse_hessian(name, masses_override_amu=masses_d)

    names_h, a_h, amp_h, _ = coordinate_data(reference, definitions)
    names_d, a_d, amp_d, _ = coordinate_data(isotope, definitions)
    if names_h != names_d:
        raise RuntimeError(f"{name}: coordinate order changed in isotope audit")
    _, chi_h, eta_h = static_matrices(reference, a_h, amp_h)
    _, chi_d, eta_d = static_matrices(isotope, a_d, amp_d)

    pair_rows = []
    max_chi_rel = 0.0
    max_eta_abs = 0.0
    for left, right in pairs:
        i = names_h.index(left)
        j = names_h.index(right)
        chi_rel = relative_error(float(chi_d[i, j]), float(chi_h[i, j]), floor=1e-14)
        eta_abs = abs(float(eta_d[i, j] - eta_h[i, j]))
        max_chi_rel = max(max_chi_rel, chi_rel)
        max_eta_abs = max(max_eta_abs, eta_abs)
        pair_rows.append({
            "coordinate_a": left,
            "coordinate_b": right,
            "chi_h": float(chi_h[i, j]),
            "chi_d": float(chi_d[i, j]),
            "chi_relative_difference": chi_rel,
            "eta_h": float(eta_h[i, j]),
            "eta_d": float(eta_d[i, j]),
            "eta_absolute_difference": eta_abs,
        })
    freq_h = reference.frequencies_cm_signed
    freq_d = isotope.frequencies_cm_signed
    return {
        "system": name,
        "atom_index_replaced_by_D": atom_index,
        "electronic_hessian_and_geometry_reused": True,
        "maximum_primary_chi_relative_difference": max_chi_rel,
        "maximum_primary_eta_absolute_difference": max_eta_abs,
        "minimum_frequency_H_cm_minus1": float(np.min(freq_h)),
        "minimum_frequency_D_cm_minus1": float(np.min(freq_d)),
        "maximum_absolute_frequency_shift_cm_minus1": float(np.max(np.abs(freq_d - freq_h))),
        "pair_results": pair_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    coordinate_maps = load_coordinate_maps()
    system_summary: dict[str, Any] = {}
    pair_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    gate_accumulator = {
        "all_stationary": True,
        "direct_modal_max_relative": 0.0,
        "reciprocity_max_abs": 0.0,
        "eta_max_excess": 0.0,
        "spectral_sum_max_abs": 0.0,
        "spectral_refinement_max_abs": 0.0,
        "raw_regularized_primary_sign_reversal_count": 0,
    }

    for name in ACCEPTED_SOURCES:
        system = analyse_hessian(name)
        definitions = coordinate_maps[name]["coordinates"]
        primary_pairs = [tuple(pair) for pair in coordinate_maps[name].get("primary_cross_pairs", [])]
        internal_pairs = [tuple(pair) for pair in coordinate_maps[name].get("internal_controls", [])]
        all_pairs = primary_pairs + internal_pairs
        names, reduced_vectors, amplitudes, lengths = coordinate_data(system, definitions)

        stationary = bool(np.all(system.eigenvalues > 0.0))
        gate_accumulator["all_stationary"] = gate_accumulator["all_stationary"] and stationary
        summary = {
            "source_npz": str(ACCEPTED_SOURCES[name].relative_to(ROOT)),
            "atom_count": len(system.symbols),
            "rigid_rank": system.rigid_rank,
            "is_linear": system.is_linear,
            "vibrational_dimension": int(len(system.eigenvalues)),
            "minimum_frequency_cm_minus1": float(np.min(system.frequencies_cm_signed)),
            "maximum_frequency_cm_minus1": float(np.max(system.frequencies_cm_signed)),
            "stationary_minimum_pass": stationary,
            "invariance": system.invariance,
        }
        if not stationary:
            summary["response_status"] = "BLOCKED_NONPOSITIVE_MODE"
            system_summary[name] = summary
            continue

        direct, chi, eta = static_matrices(system, reduced_vectors, amplitudes)
        direct_abs = float(np.max(np.abs(direct - chi)))
        direct_rel = direct_abs / max(float(np.max(np.abs(chi))), 1e-30)
        reciprocity = float(np.max(np.abs(chi - chi.T)))
        eta_excess = max(0.0, float(np.max(np.abs(eta))) - 1.0)
        gate_accumulator["direct_modal_max_relative"] = max(gate_accumulator["direct_modal_max_relative"], direct_rel)
        gate_accumulator["reciprocity_max_abs"] = max(gate_accumulator["reciprocity_max_abs"], reciprocity)
        gate_accumulator["eta_max_excess"] = max(gate_accumulator["eta_max_excess"], eta_excess)
        summary.update({
            "response_status": "COMPUTED",
            "direct_modal_max_abs": direct_abs,
            "direct_modal_max_relative": direct_rel,
            "reciprocity_max_abs": reciprocity,
            "eta_max_abs": float(np.max(np.abs(eta))),
        })

        for coord_name, length in zip(names, lengths, strict=True):
            geometry_rows.append({
                "system": name,
                "coordinate": coord_name,
                "length_bohr": float(length),
                "length_angstrom": float(length / 1.8897261254578281),
                "self_chi_bohr2_per_hartree": float(chi[names.index(coord_name), names.index(coord_name)]),
            })

        for pair_role, pairs in (("primary", primary_pairs), ("internal_control", internal_pairs)):
            for left, right in pairs:
                row, modes = pair_metrics(system, names, amplitudes, chi, eta, left, right, pair_role)
                pair_rows.append(row)
                mode_rows.extend(modes)

        raw_result = raw_projected_static(system, definitions)
        if raw_result is None:
            summary["raw_response_sensitivity_status"] = "BLOCKED_NONPOSITIVE_RAW_PROJECTED_MODE"
        else:
            raw_names, raw_eta = raw_result
            summary["raw_response_sensitivity_status"] = "COMPUTED"
            for left, right in primary_pairs:
                i = names.index(left)
                j = names.index(right)
                ri = raw_names.index(left)
                rj = raw_names.index(right)
                corrected_value = float(eta[i, j])
                raw_value = float(raw_eta[ri, rj])
                reversal = bool(np.signbit(raw_value) != np.signbit(corrected_value) and raw_value != 0.0 and corrected_value != 0.0)
                if reversal:
                    gate_accumulator["raw_regularized_primary_sign_reversal_count"] += 1
                sensitivity_rows.append({
                    "system": name,
                    "coordinate_a": left,
                    "coordinate_b": right,
                    "eta_raw_projected": raw_value,
                    "eta_regularized": corrected_value,
                    "eta_absolute_change": abs(corrected_value - raw_value),
                    "sign_reversal": reversal,
                })

        eq_groups = coordinate_maps[name].get("equivalent_coordinate_groups", [])
        equivalence_rows.extend(equivalence_audit(system, names, lengths, chi, eq_groups))

        spec = spectral_audit(system, names, amplitudes, chi, eta, primary_pairs)
        spectral_rows.extend(spec)
        for entry in spec:
            gate_accumulator["spectral_sum_max_abs"] = max(
                gate_accumulator["spectral_sum_max_abs"],
                float(entry["finest_absolute_error"]),
            )
            gate_accumulator["spectral_refinement_max_abs"] = max(
                gate_accumulator["spectral_refinement_max_abs"],
                float(entry["refinement_difference_1_to_0p5"]),
                float(entry["refinement_difference_0p5_to_0p25"]),
            )

        system_summary[name] = summary

    isotope_results = {
        "fhf_minus_shared_H_to_D": isotope_audit(
            "fhf_minus",
            1,
            coordinate_maps["fhf_minus"]["coordinates"],
            [tuple(x) for x in coordinate_maps["fhf_minus"]["primary_cross_pairs"]],
        ),
        "h5o2_plus_shared_H_to_D": isotope_audit(
            "h5o2_plus",
            1,
            coordinate_maps["h5o2_plus"]["coordinates"],
            [tuple(x) for x in coordinate_maps["h5o2_plus"]["primary_cross_pairs"]],
        ),
    }
    isotope_max_static = max(
        result["maximum_primary_chi_relative_difference"]
        for result in isotope_results.values()
    )

    gates = {
        "declared_tolerances": {
            "direct_modal_relative": DIRECT_MODAL_REL_TOL,
            "reciprocity_absolute": RECIPROCITY_ABS_TOL,
            "eta_bound_excess": ETA_BOUND_TOL,
            "spectral_sum_absolute": SPECTRAL_SUM_ABS_TOL,
            "spectral_refinement_absolute": SPECTRAL_REFINEMENT_ABS_TOL,
            "isotope_static_relative": ISOTOPE_STATIC_REL_TOL,
        },
        "observed": {
            **gate_accumulator,
            "isotope_static_max_relative": isotope_max_static,
        },
    }
    gates["pass"] = {
        "all_stationary": bool(gate_accumulator["all_stationary"]),
        "direct_modal": gate_accumulator["direct_modal_max_relative"] <= DIRECT_MODAL_REL_TOL,
        "reciprocity": gate_accumulator["reciprocity_max_abs"] <= RECIPROCITY_ABS_TOL,
        "eta_bound": gate_accumulator["eta_max_excess"] <= ETA_BOUND_TOL,
        "spectral_sum": gate_accumulator["spectral_sum_max_abs"] <= SPECTRAL_SUM_ABS_TOL,
        "spectral_refinement": gate_accumulator["spectral_refinement_max_abs"] <= SPECTRAL_REFINEMENT_ABS_TOL,
        "isotope_static_invariance": isotope_max_static <= ISOTOPE_STATIC_REL_TOL,
    }
    gates["overall_numerical_pass"] = all(gates["pass"].values())
    gates["scientific_review_flags"] = {
        "raw_regularized_primary_sign_reversal_count": gate_accumulator["raw_regularized_primary_sign_reversal_count"],
        "note": "Any raw-versus-regularized sign reversal is reported for curator review and is not silently thresholded away.",
    }

    summary_document = {
        "analysis_role": "harmonic-response postprocessing",
        "accepted_set_commit": "7f1865175a9fc30394af68a2b3dcf6405a8096a3",
        "accepted_system_count": len(ACCEPTED_SOURCES),
        "excluded_from_static_inversion": ["benzene_methane"],
        "systems": system_summary,
    }
    (OUT / "stage3a_system_summary.json").write_text(json.dumps(summary_document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "stage3a_numerical_gates.json").write_text(json.dumps(gates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "stage3a_spectral_grid_audit.json").write_text(json.dumps(spectral_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "stage3a_equivalence_audit.json").write_text(json.dumps(equivalence_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "stage3a_isotope_audit.json").write_text(json.dumps(isotope_results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "stage3a_invariance_sensitivity.json").write_text(json.dumps(sensitivity_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(OUT / "stage3a_pair_metrics.csv", pair_rows)
    write_csv(OUT / "stage3a_dominant_modes.csv", mode_rows)
    write_csv(OUT / "stage3a_geometry.csv", geometry_rows)

    lines = [
        "# Stage 3A Unified Ten-System DIRS Transferability Audit",
        "",
        "This analysis uses exactly the ten-system accepted set frozen in `theory/stage3a_acceptance_decision.md`. Benzene···CH4 is retained as failed stationary-point evidence and is not inverted.",
        "",
        "## Numerical gates",
        "",
        f"- overall numerical pass: {gates['overall_numerical_pass']}",
        f"- all accepted Hessians stationary after rigid-invariance correction: {gates['pass']['all_stationary']}",
        f"- direct/modal max relative error: {gate_accumulator['direct_modal_max_relative']:.6e}",
        f"- reciprocity max absolute error: {gate_accumulator['reciprocity_max_abs']:.6e}",
        f"- eta bound maximum excess: {gate_accumulator['eta_max_excess']:.6e}",
        f"- spectral signed-area max absolute error: {gate_accumulator['spectral_sum_max_abs']:.6e}",
        f"- spectral grid-refinement max absolute difference: {gate_accumulator['spectral_refinement_max_abs']:.6e}",
        f"- isotope static-response max relative difference: {isotope_max_static:.6e}",
        f"- raw-versus-regularized primary eta sign reversals: {gate_accumulator['raw_regularized_primary_sign_reversal_count']}",
        "",
        "## System stationary-point status",
        "",
    ]
    for name, item in system_summary.items():
        lines.append(
            f"- `{name}`: min projected frequency {item['minimum_frequency_cm_minus1']:.6f} cm^-1; "
            f"stationary={item['stationary_minimum_pass']}; response={item['response_status']}."
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This file reports preregistered response observables and numerical audits only. Chemical interpretation of class-level sign/frequency/cancellation patterns is performed separately after these outputs are accepted. No new scalar descriptor or electronic-structure comparator is introduced here.",
        "",
    ])
    (OUT / "STAGE3A_TRANSFERABILITY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
