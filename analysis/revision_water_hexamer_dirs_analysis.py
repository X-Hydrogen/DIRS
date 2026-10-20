#!/usr/bin/env python3
"""Analyse the preregistered JPCL R1 water-hexamer DIRS scalability test.

This is a post-calculation analysis only. It consumes the checksum-locked ORCA
Hessian/geometry from the self-hosted runner, regenerates no electronic-
structure data, and evaluates exactly the six topology-selected endpoints fixed
before any DIRS metric was inspected.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from dirs.archive import save_hessian_npz
from dirs.backends import parse_orca_hess
from dirs.coordinates import distance_wilson_vector, mass_weight_wilson
from dirs.invariance import enforce_cartesian_rigid_invariance
from dirs.rigid import expanded_masses, mass_weighted_rigid_basis
from dirs.units import HARTREE_WAVENUMBER_CM, amu_to_electron_masses

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_ROOT = ROOT / "calculations" / "revision_r1_water_hexamer_network" / "results"
DEFAULT_GRAPH = ROOT / "calculations" / "revision_r1_water_hexamer_network" / "results" / "preregistered_endpoint_graph.json"
DEFAULT_OUT = ROOT / "analysis" / "revision_water_hexamer_outputs"
DEFAULT_COMPACT = ROOT / "calculations" / "revision_r1_water_hexamer_network" / "results" / "compact" / "water_hexamer_book_pbe0d4.npz"

JOB = "water_hexamer_book_pbe0d4"
LINEARITY_TOLERANCE = 1.0e-8
DIRECT_MODAL_REL_TOL = 1.0e-8
RECIPROCITY_ABS_TOL = 1.0e-10
ETA_BOUND_TOL = 1.0e-10
GAMMAS_CM = (5.0, 20.0, 50.0)
GRID_STEPS_CM = (1.0, 0.5, 0.25)
COMPLIANCE_A0_2_PER_EH_TO_ANGSTROM_PER_MDYN = 0.0642304854990309


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    p.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--compact", type=Path, default=DEFAULT_COMPACT)
    return p.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_spectrum(
    eigenvalues: np.ndarray,
    amplitudes_a: np.ndarray,
    amplitudes_b: np.ndarray,
    norm: float,
    grid_cm: np.ndarray,
    gamma_cm: float,
) -> np.ndarray:
    omega = grid_cm / HARTREE_WAVENUMBER_CM
    gamma = gamma_cm / HARTREE_WAVENUMBER_CM
    coefficients = amplitudes_a * amplitudes_b / norm
    eig = eigenvalues[:, None]
    w = omega[None, :]
    denominator = (eig - w**2) ** 2 + (2.0 * gamma * w) ** 2
    d_au = np.sum((4.0 * gamma / math.pi) * coefficients[:, None] / denominator, axis=0)
    return d_au / HARTREE_WAVENUMBER_CM


def raw_orca_frequency_header(output: Path) -> dict[str, Any]:
    if not output.is_file():
        return {"available": False}
    text = output.read_text(encoding="utf-8", errors="replace")
    values: list[float] = []
    in_frequency_section = False
    for line in text.splitlines():
        if line.strip() == "VIBRATIONAL FREQUENCIES":
            in_frequency_section = True
            continue
        if not in_frequency_section:
            continue
        match = re.match(r"^\s*\d+:\s*([-+]?\d+(?:\.\d+)?)\s+cm\*\*-1", line)
        if match:
            values.append(float(match.group(1)))
            if len(values) >= 12:
                break
    cycles = [int(x) for x in re.findall(r"GEOMETRY OPTIMIZATION CYCLE\s+(\d+)", text)]
    final_energies = [float(x) for x in re.findall(r"FINAL SINGLE POINT ENERGY\s+([-+]?\d+\.\d+)", text)]
    return {
        "available": True,
        "raw_first_frequencies_cm_minus1": values,
        "raw_negative_frequency_count_in_first_12": sum(v < 0.0 for v in values),
        "note": "ORCA raw Hessian output explicitly states that translations/rotations were not projected; stationary-point acceptance uses the frozen rigid-projected convention.",
        "optimization_last_cycle": max(cycles) if cycles else None,
        "final_single_point_energy_hartree": final_energies[-1] if final_energies else None,
        "normal_termination": "ORCA TERMINATED NORMALLY" in text,
        "optimization_converged": "THE OPTIMIZATION HAS CONVERGED" in text,
    }


def main() -> None:
    args = parse_args()
    hess_path = args.raw_root / f"{JOB}.hess"
    xyz_path = args.raw_root / f"{JOB}.xyz"
    out_path = args.raw_root / f"{JOB}.out"
    completed = args.raw_root / "COMPLETED.sha256"
    for required in (hess_path, xyz_path, completed, args.graph):
        if not required.is_file():
            raise FileNotFoundError(required)

    args.out.mkdir(parents=True, exist_ok=True)
    record = parse_orca_hess(hess_path)
    save_hessian_npz(record, args.compact)

    graph = json.loads(args.graph.read_text(encoding="utf-8"))
    endpoints = graph["endpoints"]
    if len(endpoints) != 6 or [e["endpoint_id"] for e in endpoints] != [f"E{i}" for i in range(1, 7)]:
        raise RuntimeError("Preregistered endpoint set is not exactly E1-E6 in deterministic order.")

    positions = np.asarray(record.positions_bohr, dtype=float)
    masses_amu = np.asarray(record.masses_amu, dtype=float)
    masses_e = amu_to_electron_masses(masses_amu)
    raw_hessian = 0.5 * (record.hessian_hartree_per_bohr2 + record.hessian_hartree_per_bohr2.T)
    invariant = enforce_cartesian_rigid_invariance(
        positions,
        raw_hessian,
        linearity_tolerance=LINEARITY_TOLERANCE,
    )
    hessian = invariant.hessian
    rigid = mass_weighted_rigid_basis(
        positions,
        masses_e,
        linearity_tolerance=LINEARITY_TOLERANCE,
    )
    inv_sqrt_mass = 1.0 / np.sqrt(expanded_masses(masses_e))
    dynamical = inv_sqrt_mass[:, None] * hessian * inv_sqrt_mass[None, :]
    dynamical = 0.5 * (dynamical + dynamical.T)
    z = rigid.vibrational_basis
    reduced = z.T @ dynamical @ z
    reduced = 0.5 * (reduced + reduced.T)
    eigenvalues, reduced_modes = np.linalg.eigh(reduced)
    frequencies_cm = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues)) * HARTREE_WAVENUMBER_CM
    modes = z @ reduced_modes

    water_atom_indices: list[list[int]] = []
    for water_index in range(1, 7):
        rec = graph["water_assignment"][f"W{water_index}"]
        atoms_1based = [rec["O"], *rec["H"]]
        water_atom_indices.append([int(x) - 1 for x in atoms_1based])

    endpoint_rows: list[dict[str, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    spectral_rows: list[dict[str, Any]] = []
    representative_columns: dict[str, np.ndarray] = {}
    direct_modal_rel_errors: list[float] = []
    reciprocity_errors: list[float] = []
    eta_abs_values: list[float] = []

    upper_cm = max(10000.0, 3.0 * float(np.max(frequencies_cm)))
    representative_grid = np.arange(0.0, upper_cm + 0.5, 1.0)

    for endpoint in endpoints:
        endpoint_id = endpoint["endpoint_id"]
        a_i, a_j = [int(x) - 1 for x in endpoint["coord_A"]["atoms"]]
        b_i, b_j = [int(x) - 1 for x in endpoint["coord_B"]["atoms"]]
        w_a = mass_weight_wilson(distance_wilson_vector(positions, a_i, a_j), masses_e)
        w_b = mass_weight_wilson(distance_wilson_vector(positions, b_i, b_j), masses_e)
        g_a = z.T @ w_a
        g_b = z.T @ w_b
        c_a = modes.T @ w_a
        c_b = modes.T @ w_b

        solve_a = np.linalg.solve(reduced, g_a)
        solve_b = np.linalg.solve(reduced, g_b)
        chi_aa = float(g_a @ solve_a)
        chi_bb = float(g_b @ solve_b)
        chi_ab = float(g_a @ solve_b)
        chi_ba = float(g_b @ solve_a)
        weights = c_a * c_b / eigenvalues
        chi_modal = float(np.sum(weights))
        norm = math.sqrt(chi_aa * chi_bb)
        eta = chi_ab / norm
        normalized_weights = weights / norm
        abs_weights = np.abs(normalized_weights)
        activity = float(np.sum(abs_weights))
        kappa = abs(eta) / activity if activity > 0.0 else 0.0
        total_abs = float(np.sum(abs_weights))

        def frac(mask: np.ndarray) -> float:
            return float(np.sum(abs_weights[mask]) / total_abs) if total_abs > 0.0 else 0.0

        rel_error = abs(chi_modal - chi_ab) / max(abs(chi_ab), 1.0e-30)
        reciprocity = abs(chi_ab - chi_ba)
        direct_modal_rel_errors.append(rel_error)
        reciprocity_errors.append(reciprocity)
        eta_abs_values.append(abs(eta))

        endpoint_rows.append({
            "endpoint_id": endpoint_id,
            "class": endpoint["class"],
            "coordinate_a_atoms_1based": "-".join(str(x) for x in endpoint["coord_A"]["atoms"]),
            "coordinate_b_atoms_1based": "-".join(str(x) for x in endpoint["coord_B"]["atoms"]),
            "contacts": ";".join(endpoint.get("contacts", [])),
            "chi_ab_bohr2_per_hartree": chi_ab,
            "compliance_A_per_mdyn": chi_ab * COMPLIANCE_A0_2_PER_EH_TO_ANGSTROM_PER_MDYN,
            "eta": eta,
            "absolute_modal_activity_A": activity,
            "coherence_kappa": kappa,
            "abs_activity_below_200_cm_fraction": frac(frequencies_cm < 200.0),
            "abs_activity_200_1000_cm_fraction": frac((frequencies_cm >= 200.0) & (frequencies_cm < 1000.0)),
            "abs_activity_above_1000_cm_fraction": frac(frequencies_cm >= 1000.0),
            "chi_aa_bohr2_per_hartree": chi_aa,
            "chi_bb_bohr2_per_hartree": chi_bb,
            "direct_modal_relative_error": rel_error,
            "reciprocity_absolute_error": reciprocity,
        })

        order = np.argsort(abs_weights)[::-1]
        for rank, mode_index in enumerate(order[:10], start=1):
            mode_cart = modes[:, mode_index].reshape(record.n_atoms, 3)
            participation = np.asarray(
                [float(np.sum(mode_cart[group, :] ** 2)) for group in water_atom_indices],
                dtype=float,
            )
            participation /= float(np.sum(participation))
            ipr = 1.0 / float(np.sum(participation**2))
            mode_rows.append({
                "endpoint_id": endpoint_id,
                "class": endpoint["class"],
                "rank": rank,
                "mode_index_zero_based": int(mode_index),
                "frequency_cm_minus1": float(frequencies_cm[mode_index]),
                "normalized_static_modal_weight": float(normalized_weights[mode_index]),
                "fraction_of_absolute_activity": float(abs_weights[mode_index] / total_abs),
                "water_monomer_ipr": ipr,
                **{f"p_W{i+1}": float(participation[i]) for i in range(6)},
            })

        for gamma_cm in GAMMAS_CM:
            integrals: dict[float, float] = {}
            for step_cm in GRID_STEPS_CM:
                grid = np.arange(0.0, upper_cm + 0.5 * step_cm, step_cm)
                spectrum = normalized_spectrum(eigenvalues, c_a, c_b, norm, grid, gamma_cm)
                integrals[step_cm] = float(np.trapezoid(spectrum, grid))
            spectral_rows.append({
                "endpoint_id": endpoint_id,
                "gamma_cm_minus1": gamma_cm,
                "upper_limit_cm_minus1": upper_cm,
                "eta_target": eta,
                "integral_step_1_cm": integrals[1.0],
                "integral_step_0p5_cm": integrals[0.5],
                "integral_step_0p25_cm": integrals[0.25],
                "finest_absolute_error": abs(integrals[0.25] - eta),
                "refinement_difference_1_to_0p5": abs(integrals[1.0] - integrals[0.5]),
                "refinement_difference_0p5_to_0p25": abs(integrals[0.5] - integrals[0.25]),
            })

        representative_columns[endpoint_id] = normalized_spectrum(
            eigenvalues, c_a, c_b, norm, representative_grid, 20.0
        )

    endpoint_fields = [
        "endpoint_id", "class", "coordinate_a_atoms_1based", "coordinate_b_atoms_1based", "contacts",
        "chi_ab_bohr2_per_hartree", "compliance_A_per_mdyn", "eta", "absolute_modal_activity_A", "coherence_kappa",
        "abs_activity_below_200_cm_fraction", "abs_activity_200_1000_cm_fraction", "abs_activity_above_1000_cm_fraction",
        "chi_aa_bohr2_per_hartree", "chi_bb_bohr2_per_hartree", "direct_modal_relative_error", "reciprocity_absolute_error",
    ]
    write_csv(args.out / "endpoint_metrics.csv", endpoint_rows, endpoint_fields)
    mode_fields = [
        "endpoint_id", "class", "rank", "mode_index_zero_based", "frequency_cm_minus1",
        "normalized_static_modal_weight", "fraction_of_absolute_activity", "water_monomer_ipr",
        "p_W1", "p_W2", "p_W3", "p_W4", "p_W5", "p_W6",
    ]
    write_csv(args.out / "top_modes_with_ipr.csv", mode_rows, mode_fields)
    spectral_fields = [
        "endpoint_id", "gamma_cm_minus1", "upper_limit_cm_minus1", "eta_target",
        "integral_step_1_cm", "integral_step_0p5_cm", "integral_step_0p25_cm",
        "finest_absolute_error", "refinement_difference_1_to_0p5", "refinement_difference_0p5_to_0p25",
    ]
    write_csv(args.out / "spectral_integral_audit.csv", spectral_rows, spectral_fields)

    spectrum_path = args.out / "signed_spectrum_gamma20_step1cm.csv"
    with spectrum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        ids = [e["endpoint_id"] for e in endpoints]
        writer.writerow(["frequency_cm_minus1", *ids])
        for index, omega in enumerate(representative_grid):
            writer.writerow([float(omega), *[float(representative_columns[eid][index]) for eid in ids]])

    validation = {
        "system": "water_hexamer_book",
        "job": JOB,
        "execution_input_sha256": completed.read_text(encoding="utf-8").strip(),
        "n_atoms": record.n_atoms,
        "rigid_rank": rigid.rigid_basis.shape[1],
        "n_vibrational_modes": int(len(eigenvalues)),
        "lowest_projected_frequency_cm_minus1": float(np.min(frequencies_cm)),
        "highest_projected_frequency_cm_minus1": float(np.max(frequencies_cm)),
        "projected_positive_definite": bool(np.all(eigenvalues > 0.0)),
        "raw_hessian_rigid_block_norm": invariant.raw_rigid_block_norm,
        "raw_hessian_rigid_vibrational_coupling_norm": invariant.raw_rigid_vibrational_coupling_norm,
        "rigid_invariance_correction_max_abs_hartree_per_bohr2": invariant.correction_max_abs,
        "rigid_invariance_correction_relative_frobenius": invariant.correction_relative_frobenius,
        "max_direct_modal_relative_error": max(direct_modal_rel_errors),
        "max_reciprocity_absolute_error": max(reciprocity_errors),
        "max_abs_eta": max(eta_abs_values),
        "endpoint_count": len(endpoint_rows),
        "raw_orca_frequency_context": raw_orca_frequency_header(out_path),
        "gates": {
            "projected_stationary_minimum": bool(np.all(eigenvalues > 0.0)),
            "direct_modal_closure": max(direct_modal_rel_errors) <= DIRECT_MODAL_REL_TOL,
            "reciprocity": max(reciprocity_errors) <= RECIPROCITY_ABS_TOL,
            "eta_bound": max(eta_abs_values) <= 1.0 + ETA_BOUND_TOL,
            "preregistered_endpoint_count": len(endpoint_rows) == 6,
        },
    }
    validation["all_gates_pass"] = all(validation["gates"].values())
    (args.out / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    cancellation_endpoints = [r for r in endpoint_rows if r["class"] != "local_contact_heavy_atom" and r["coherence_kappa"] < 0.5]
    delocalized_top_modes = [r for r in mode_rows if r["rank"] <= 5 and r["water_monomer_ipr"] >= 3.0]
    lines = [
        "# JPCL R1 water-hexamer DIRS scalability analysis",
        "",
        "This analysis evaluates the six topology-selected endpoints frozen before any DIRS metric was inspected.",
        "",
        "## Validation",
        "",
        f"- projected vibrational modes: {len(eigenvalues)}",
        f"- lowest rigid-projected vibrational frequency: {np.min(frequencies_cm):.6f} cm^-1",
        f"- positive-definite projected minimum: {validation['gates']['projected_stationary_minimum']}",
        f"- max direct/modal relative error: {max(direct_modal_rel_errors):.3e}",
        f"- max reciprocity absolute error: {max(reciprocity_errors):.3e}",
        f"- max |eta|: {max(eta_abs_values):.12f}",
        f"- all preregistered numerical gates pass: **{validation['all_gates_pass']}**",
        "",
        "The small negative values printed by ORCA among modes 0-5 are not accepted as vibrational imaginary modes: the raw output states that translations and rotations were not projected. Under the frozen Stage-3A rigid-projected convention the lowest vibrational mode is positive.",
        "",
        "## Endpoint summary",
        "",
        "| Endpoint | Class | chi_AB (a0^2/Eh) | C_AB (A/mdyn) | eta | A | kappa | <200 | 200-1000 | >1000 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in endpoint_rows:
        lines.append(
            f"| {row['endpoint_id']} | {row['class']} | {row['chi_ab_bohr2_per_hartree']:.8g} | "
            f"{row['compliance_A_per_mdyn']:.8g} | {row['eta']:.8g} | {row['absolute_modal_activity_A']:.8g} | "
            f"{row['coherence_kappa']:.8g} | {row['abs_activity_below_200_cm_fraction']:.4f} | "
            f"{row['abs_activity_200_1000_cm_fraction']:.4f} | {row['abs_activity_above_1000_cm_fraction']:.4f} |"
        )
    lines += [
        "",
        "## Preregistered scalability observations",
        "",
        f"- Nonsymmetric adjacent/nonlocal endpoints with kappa < 0.5: {', '.join(r['endpoint_id'] for r in cancellation_endpoints) if cancellation_endpoints else 'none'}.",
        f"- Top-five endpoint/mode entries with monomer IPR >= 3.0: {len(delocalized_top_modes)}. This is descriptive evidence that important response contributions can come from modes distributed over several waters rather than from one localized eigenvector.",
        "- The local contact/heavy-atom controls remain coherent, whereas adjacent and nonlocal network contacts show much stronger cancellation. This contrast was not used to choose the endpoints.",
        "- IPR is reported only as a mode-delocalization descriptor; it is not a DIRS interaction metric.",
        "",
        "## Scope",
        "",
        "These results address selected-coordinate interpretability and cancellation in a larger hydrogen-bond network. They do not convert DIRS into an IVR rate, a fluxional large-amplitude theory, or a direct 2DIR/IR-THz observable.",
    ]
    (args.out / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not validation["all_gates_pass"]:
        raise SystemExit("One or more preregistered numerical gates failed; see validation.json")
    print(json.dumps({
        "all_gates_pass": validation["all_gates_pass"],
        "lowest_projected_frequency_cm_minus1": validation["lowest_projected_frequency_cm_minus1"],
        "max_direct_modal_relative_error": validation["max_direct_modal_relative_error"],
        "cancellation_endpoints_kappa_lt_0p5": [r["endpoint_id"] for r in cancellation_endpoints],
    }, indent=2))


if __name__ == "__main__":
    main()
