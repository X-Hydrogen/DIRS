#!/usr/bin/env python3
"""Recompute the JPCL DIRS tables from archived geometries/Hessians.

No ORCA execution, pretrained results, external network, or saved response
matrices are used. Reference CSVs serve ONLY as targets for post-computation
comparisons, not as sources for evaluated numerical observables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # permit source-tree analysis import without packaging it

import numpy as np

from dirs.archive import load_hessian_npz
from dirs.backends import parse_orca_hess
from dirs.units import HARTREE_WAVENUMBER_CM
import analysis.stage3a_unified_response_analysis as s3

REFERENCE = ROOT / "reference"
CONVERSION = 0.0642304854990309  # (a0^2/Eh) -> (Angstrom/mdyn), distance coordinates
PRIMARY_FIELDS = {
    "chi_ab_bohr2_per_hartree": "chi_ab_bohr2_per_hartree",
    "eta": "eta",
    "A": "absolute_modal_activity_A",
    "kappa": "coherence_kappa",
    "low_lt200_fraction": "abs_activity_below_200_cm_fraction",
    "mid_200_1000_fraction": "abs_activity_200_1000_cm_fraction",
    "high_gt1000_fraction": "abs_activity_above_1000_cm_fraction",
}
ALTERNATIVE_FIELDS = {
    "chi_ab_bohr2_per_hartree": "chi_ab_bohr2_per_hartree",
    "eta": "eta",
    "absolute_modal_activity_A": "absolute_modal_activity_A",
    "coherence_kappa": "coherence_kappa",
    "abs_activity_below_200_cm_fraction": "abs_activity_below_200_cm_fraction",
    "abs_activity_200_1000_cm_fraction": "abs_activity_200_1000_cm_fraction",
    "abs_activity_above_1000_cm_fraction": "abs_activity_above_1000_cm_fraction",
}
HEX_FIELDS = {
    "chi_ab_bohr2_per_hartree": "chi_ab_bohr2_per_hartree",
    "eta": "eta",
    "absolute_modal_activity_A": "absolute_modal_activity_A",
    "coherence_kappa": "coherence_kappa",
    "abs_activity_below_200_cm_fraction": "abs_activity_below_200_cm_fraction",
    "abs_activity_200_1000_cm_fraction": "abs_activity_200_1000_cm_fraction",
    "abs_activity_above_1000_cm_fraction": "abs_activity_above_1000_cm_fraction",
}
ALTERNATIVE_SOURCES = {
    name: ROOT / "calculations" / "stage3b_method_robustness" / "results" /
          "compact" / f"{name}_wb97x_d4.npz"
    for name in ("water_dimer", "methane_dimer", "chloride_water",
                 "fhf_minus", "ammonia_borane")
}


def read_csv(filename: str) -> list[dict[str, str]]:
    with (REFERENCE / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty reproduction table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_close(expected: str | float, observed: float, label: str,
                 *, rtol: float = 2e-6, atol: float = 2e-8) -> float:
    e = float(expected)
    o = float(observed)
    if not np.isfinite(e) or not np.isfinite(o):
        raise AssertionError(f"{label}: nonfinite value ({e}, {o})")
    error = abs(e - o)
    if not np.isclose(e, o, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}: expected {e:.16g}, computed {o:.16g}, "
            f"abs error {error:.3g} (rtol={rtol}, atol={atol})"
        )
    return error


def check_closure(hs: s3.HarmonicSystem, direct: np.ndarray,
                  modal: np.ndarray, label: str) -> float:
    if np.any(hs.eigenvalues <= 0):
        raise AssertionError(f"{label}: nonpositive projected vibrational eigenvalue")
    diff = float(np.max(np.abs(direct - modal)))
    scale = max(float(np.max(np.abs(modal))), 1e-30)
    relative = diff / scale
    if relative > 1e-8:
        raise AssertionError(f"{label}: direct/modal closure {relative:.3e} > 1e-8")
    if float(np.max(np.abs(modal - modal.T))) > 1e-10:
        raise AssertionError(f"{label}: reciprocity check failed")
    return relative


def run_primary(out: Path) -> dict[str, object]:
    expected = read_csv("compliance_37_endpoints.csv")
    targets = {
        (r["system"], r["coordinate_a"], r["coordinate_b"]): r
        for r in expected
    }
    if len(expected) != 37 or len(targets) != 37:
        raise AssertionError("Expected exactly 37 unique primary endpoints")
    accepted = sorted({r["system"] for r in expected})
    if len(accepted) != 9:
        raise AssertionError("Expected exactly nine accepted primary systems")

    maps = s3.load_coordinate_maps()
    observed: list[dict[str, object]] = []
    closure: dict[str, float] = {}
    max_difference = 0.0

    for system in accepted:
        hs = s3.analyse_hessian(system)
        definitions = maps[system]["coordinates"]
        names, reduced_vectors, amplitudes, _ = s3.coordinate_data(hs, definitions)
        direct, chi, eta = s3.static_matrices(hs, reduced_vectors, amplitudes)
        closure[system] = check_closure(hs, direct, chi, system)
        pairs = (
            [tuple(x) for x in maps[system].get("primary_cross_pairs", [])]
            + [tuple(x) for x in maps[system].get("internal_controls", [])]
        )
        for left, right in pairs:
            key = (system, left, right)
            if key not in targets:
                continue
            calculated, _ = s3.pair_metrics(
                hs, names, amplitudes, chi, eta, left, right, "published"
            )
            for pub_name, src_name in PRIMARY_FIELDS.items():
                max_difference = max(
                    max_difference,
                    assert_close(targets[key][pub_name], calculated[src_name],
                                 f"37:{system}/{left}/{right}/{pub_name}"),
                )
            calculated_compliance = calculated["chi_ab_bohr2_per_hartree"] * CONVERSION
            max_difference = max(
                max_difference,
                assert_close(targets[key]["C_AB_angstrom_per_mdyn"],
                             calculated_compliance,
                             f"37:{system}/{left}/{right}/compliance",
                             rtol=1e-8, atol=1e-8),
            )
            observed.append({
                "system": system, "coordinate_a": left, "coordinate_b": right,
                **{k: float(calculated[v]) for k, v in PRIMARY_FIELDS.items()},
                "C_AB_angstrom_per_mdyn": float(calculated_compliance),
            })

    found = {(r["system"], r["coordinate_a"], r["coordinate_b"]) for r in observed}
    if found != set(targets) or len(observed) != 37:
        raise AssertionError(f"Primary endpoint mismatch: missing {sorted(set(targets)-found)}")
    write_csv(out / "primary_37_recomputed.csv", observed)
    return {
        "accepted_systems": accepted, "endpoint_count": len(observed),
        "max_absolute_target_difference": max_difference,
        "max_direct_modal_relative": max(closure.values()),
    }


def run_hexamer(out: Path) -> dict[str, object]:
    path = (
        ROOT / "calculations" / "revision_r1_water_hexamer_network" /
        "results" / "compact" / "water_hexamer_book_pbe0d4.npz"
    )
    graph_path = (
        ROOT / "calculations" / "revision_r1_water_hexamer_network" /
        "results" / "preregistered_endpoint_graph.json"
    )
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    endpoints = graph["endpoints"]
    if [x["endpoint_id"] for x in endpoints] != [f"E{i}" for i in range(1, 7)]:
        raise AssertionError("Frozen water-hexamer E1–E6 graph missing or reordered")
    record = load_hessian_npz(path)
    registry = dict(s3.ACCEPTED_SOURCES)
    s3.ACCEPTED_SOURCES["water_hexamer_book"] = path
    try:
        hs = s3.analyse_hessian("water_hexamer_book")
    finally:
        s3.ACCEPTED_SOURCES.clear()
        s3.ACCEPTED_SOURCES.update(registry)

    definitions: dict[str, list[int]] = {}
    for e in endpoints:
        for letter in ("A", "B"):
            definitions[f'{e["endpoint_id"]}_{letter}'] = [
                int(n)-1 for n in e[f"coord_{letter}"]["atoms"]
            ]
    names, red, amps, _ = s3.coordinate_data(hs, definitions)
    direct, chi, eta = s3.static_matrices(hs, red, amps)
    closure = check_closure(hs, direct, chi, "water_hexamer_book")
    expected = {r["endpoint_id"]: r for r in read_csv("water_hexamer_endpoint_metrics.csv")}
    expected_modes = {
        (r["endpoint_id"], int(r["rank"])): r
        for r in read_csv("water_hexamer_top_modes_with_ipr.csv")
    }
    waters = []
    for j in range(1, 7):
        w = graph["water_assignment"][f"W{j}"]
        waters.append([int(w["O"])-1] + [int(h)-1 for h in w["H"]])
    observed: list[dict[str, object]] = []
    max_difference = 0.0
    top5_delocalized = 0

    for e in endpoints:
        eid = e["endpoint_id"]
        a, b = f"{eid}_A", f"{eid}_B"
        calculated, _ = s3.pair_metrics(hs, names, amps, chi, eta, a, b, "hexamer")
        for pub, comp in HEX_FIELDS.items():
            max_difference = max(max_difference, assert_close(
                expected[eid][pub], calculated[comp], f"hex:{eid}/{pub}"
            ))
        compliance = float(calculated["chi_ab_bohr2_per_hartree"]) * CONVERSION
        max_difference = max(max_difference, assert_close(
            expected[eid]["compliance_A_per_mdyn"], compliance,
            f"hex:{eid}/compliance", rtol=1e-8, atol=1e-8,
        ))
        observed.append({
            "endpoint_id": eid, "class": e["class"],
            **{k: float(calculated[v]) for k, v in HEX_FIELDS.items()},
            "compliance_A_per_mdyn": compliance,
        })

        index_a, index_b = names.index(a), names.index(b)
        weights = amps[:, index_a] * amps[:, index_b] / hs.eigenvalues
        order = np.argsort(np.abs(weights))[::-1]
        for rank, mode_index in enumerate(order[:5], start=1):
            mode = hs.modes[:, mode_index].reshape(record.n_atoms, 3)
            participation = np.array(
                [float(np.sum(mode[indices, :]**2)) for indices in waters]
            )
            participation /= float(participation.sum())
            ipr = 1.0 / float(np.sum(participation**2))
            target = expected_modes[(eid, rank)]
            max_difference = max(max_difference, assert_close(
                target["water_monomer_ipr"], ipr, f"hex:{eid}/IPR{rank}"
            ))
            if ipr >= 3:
                top5_delocalized += 1

    if len(observed) != 6 or len(expected) != 6:
        raise AssertionError("Water-hexamer endpoint count is not six")
    if top5_delocalized != 27:
        raise AssertionError(f"Expected 27 delocalized top-five entries; got {top5_delocalized}")
    lowest_frequency = float(np.sqrt(np.min(hs.eigenvalues)) * HARTREE_WAVENUMBER_CM)
    assert_close(29.785779267230357, lowest_frequency, "hex:lowest_frequency",
                 rtol=1e-6, atol=1e-5)
    write_csv(out / "water_hexamer_6_recomputed.csv", observed)
    return {
        "endpoint_count": len(observed),
        "lowest_projected_frequency_cm_minus1": lowest_frequency,
        "delocalized_top_five_entries_ipr_ge_3": top5_delocalized,
        "max_absolute_target_difference": max_difference,
        "direct_modal_relative": closure,
    }


def run_functional(out: Path) -> dict[str, object]:
    expected_rows = read_csv("stage3b_endpoint_comparison.csv")
    expected = {
        (r["system"], r["coordinate_a"], r["coordinate_b"]): r
        for r in expected_rows
    }
    if len(expected) != len(expected_rows):
        raise AssertionError("Duplicate functional-control endpoints")
    maps = s3.load_coordinate_maps()
    observed = []
    max_difference = 0.0
    max_closure = 0.0

    registry = dict(s3.ACCEPTED_SOURCES)
    try:
        observed, max_difference, max_closure = _functional_endpoints(
            expected, maps, observed, max_difference, max_closure
        )
    finally:
        # The controls reuse the primary system names; restore the PBE0-D4
        # registry so later callers never silently analyse omegaB97X-D4 data.
        s3.ACCEPTED_SOURCES.clear()
        s3.ACCEPTED_SOURCES.update(registry)
    observed_keys = {(x["system"], x["coordinate_a"], x["coordinate_b"]) for x in observed}
    if observed_keys != set(expected):
        raise AssertionError(
            f"Functional control set mismatch, missing={set(expected)-observed_keys}"
        )
    write_csv(out / "functional_controls_recomputed.csv", observed)
    return {
        "endpoint_count": len(observed),
        "systems": list(ALTERNATIVE_SOURCES),
        "max_absolute_target_difference": max_difference,
        "max_direct_modal_relative": max_closure,
    }


def _functional_endpoints(expected, maps, observed, max_difference, max_closure):
    for name, compact_path in ALTERNATIVE_SOURCES.items():
        s3.ACCEPTED_SOURCES[name] = compact_path
        hs = s3.analyse_hessian(name)
        names, reduced, amplitudes, _ = s3.coordinate_data(
            hs, maps[name]["coordinates"]
        )
        direct, chi, eta = s3.static_matrices(hs, reduced, amplitudes)
        max_closure = max(max_closure, check_closure(hs, direct, chi, name + ":alternative"))
        for left, right in [tuple(p) for p in maps[name].get("primary_cross_pairs", [])]:
            key = (name, left, right)
            if key not in expected:
                raise AssertionError(f"Missing functional-control reference: {key}")
            calculated, _ = s3.pair_metrics(
                hs, names, amplitudes, chi, eta, left, right, "functional"
            )
            for pub, src in ALTERNATIVE_FIELDS.items():
                max_difference = max(max_difference, assert_close(
                    expected[key][pub], calculated[src],
                    f"alternative:{name}/{left}/{right}/{pub}"
                ))
            observed.append({
                "system": name, "coordinate_a": left, "coordinate_b": right,
                **{k: float(calculated[v]) for k, v in ALTERNATIVE_FIELDS.items()},
            })
    return observed, max_difference, max_closure


def verify_raw_orca() -> dict[str, object]:
    """Independently parse deposited text .hess files vs archived numeric NPZs."""
    entries: list[tuple[str, Path, Path]] = []
    for name in ("water_dimer", "methane_dimer", "fhf_minus"):
        base = ROOT / "calculations" / "prototypes_stage2" / "results"
        entries.append((name, base / "hessians" / f"{name}.hess",
                       base / "compact" / f"{name}.npz"))
    for name in ("hf_dimer", "hcn_dimer", "nh3_hf", "chloride_water",
                 "h5o2_plus", "ammonia_borane"):
        base = ROOT / "calculations" / "benchmark_stage3" / "results"
        entries.append((name, base / "hessians" / f"{name}.hess",
                       base / "compact" / f"{name}.npz"))
    for name, path in ALTERNATIVE_SOURCES.items():
        stem = f"{name}_wb97x_d4"
        base = ROOT / "calculations" / "stage3b_method_robustness" / "results"
        entries.append((stem, base / "hessians" / f"{stem}.hess", path))
    base = ROOT / "calculations" / "revision_r1_water_hexamer_network" / "results"
    entries.append(("water_hexamer_book_pbe0d4",
                    base / "water_hexamer_book_pbe0d4.hess",
                    base / "compact" / "water_hexamer_book_pbe0d4.npz"))

    max_hessian_difference = 0.0
    for name, hess_file, compact_file in entries:
        raw = parse_orca_hess(hess_file)
        compact = load_hessian_npz(compact_file)
        if raw.symbols != compact.symbols:
            raise AssertionError(f"{name}: ORCA text vs NPZ symbols mismatch")
        for quantity in ("positions_bohr", "masses_amu", "hessian_hartree_per_bohr2"):
            x = getattr(raw, quantity)
            y = getattr(compact, quantity)
            if x.shape != y.shape:
                raise AssertionError(f"{name}: {quantity} shape mismatch")
            diff = float(np.max(np.abs(x - y)))
            if not np.allclose(x, y, rtol=5e-9, atol=5e-9):
                raise AssertionError(
                    f"{name}: ORCA text vs NPZ {quantity} differs by {diff:.6e}"
                )
            if quantity == "hessian_hartree_per_bohr2":
                max_hessian_difference = max(max_hessian_difference, diff)
    return {
        "orca_hessians_independently_parsed": len(entries),
        "max_hessian_absolute_difference_au": max_hessian_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--verify-raw", action="store_true",
                        help="Also cross-check every deposited ORCA .hess against its NPZ")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "scope": "postprocess existing ORCA Hessians; no DFT job",
        "primary": run_primary(out),
        "water_hexamer": run_hexamer(out),
        "functional_control": run_functional(out),
    }
    if args.verify_raw:
        result["raw_orca_crosscheck"] = verify_raw_orca()
    result["all_comparisons_passed"] = True
    (out / "reproduction_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
