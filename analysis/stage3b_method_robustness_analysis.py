from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "calculations" / "stage3b_method_robustness"
OUT = ROOT / "analysis" / "stage3b_method_robustness_outputs"
OUT.mkdir(parents=True, exist_ok=True)

SYSTEMS = {
    "water_dimer": "water_dimer_wb97x_d4",
    "methane_dimer": "methane_dimer_wb97x_d4",
    "chloride_water": "chloride_water_wb97x_d4",
    "fhf_minus": "fhf_minus_wb97x_d4",
    "ammonia_borane": "ammonia_borane_wb97x_d4",
}

# Frozen before alternative-method inspection: theory/stage3b_method_robustness_design_r1.md
MAX_DELTA_ETA = 0.15
MAX_DELTA_KAPPA = 0.20
MIN_SPEARMAN = 0.80
NONCANCELLATION_KAPPA = 0.20


def collect_compact() -> dict[str, Path]:
    compact = STAGE / "results" / "compact"
    metadata = STAGE / "results" / "compact_metadata"
    final_xyz = STAGE / "results" / "compact_structures"
    paths: dict[str, Path] = {}
    collector = ROOT / "calculations" / "benchmark_stage3" / "collect_orca_result.py"
    for system, stem in SYSTEMS.items():
        hess = STAGE / "results" / "hessians" / f"{stem}.hess"
        xyz = STAGE / "results" / "structures" / f"{stem}.xyz"
        npz = compact / f"{system}_wb97x_d4.npz"
        subprocess.run([
            sys.executable, str(collector), "--system", f"{system}_wb97x_d4",
            "--hess", str(hess), "--starting-xyz", str(xyz),
            "--output-npz", str(npz), "--output-xyz", str(final_xyz / f"{stem}.xyz"),
            "--output-json", str(metadata / f"{stem}.json")
        ], check=True)
        paths[system] = npz
    return paths


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    rx, ry = rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    compact = collect_compact()
    import analysis.stage3a_unified_response_analysis as s3
    s3.ACCEPTED_SOURCES = compact
    maps = s3.load_coordinate_maps()

    baseline_path = ROOT / "analysis" / "stage3a_nine_system_outputs" / "stage3a_primary_manuscript_metrics.csv"
    baseline_rows = list(csv.DictReader(baseline_path.open(encoding="utf-8")))
    baseline = {(r["system"], r["coordinate_a"], r["coordinate_b"]): r for r in baseline_rows if r["system"] in SYSTEMS}

    rows = []
    system_audit = {}
    raw_sign_reversals = []
    for system in SYSTEMS:
        hs = s3.analyse_hessian(system)
        defs = maps[system]["coordinates"]
        pairs = [tuple(p) for p in maps[system].get("primary_cross_pairs", [])]
        names, reduced, amps, lengths = s3.coordinate_data(hs, defs)
        stationary = bool(np.all(hs.eigenvalues > 0.0))
        entry = {
            "minimum_frequency_cm_minus1": float(np.min(hs.frequencies_cm_signed)),
            "stationary_minimum_pass": stationary,
            "invariance": hs.invariance,
            "coordinate_lengths_angstrom": {n: float(v / 1.8897261254578281) for n, v in zip(names, lengths, strict=True)},
        }
        if not stationary:
            entry["status"] = "FAIL_NONPOSITIVE_MODE"
            system_audit[system] = entry
            continue
        direct, chi, eta = s3.static_matrices(hs, reduced, amps)
        direct_rel = float(np.max(np.abs(direct - chi)) / max(float(np.max(np.abs(chi))), 1e-30))
        reciprocity = float(np.max(np.abs(chi - chi.T)))
        eta_excess = max(0.0, float(np.max(np.abs(eta))) - 1.0)
        entry.update({"status": "COMPUTED", "direct_modal_max_relative": direct_rel, "reciprocity_max_abs": reciprocity, "eta_max_excess": eta_excess})
        raw = s3.raw_projected_static(hs, defs)
        raw_names, raw_eta = raw if raw is not None else (None, None)
        for left, right in pairs:
            r, _ = s3.pair_metrics(hs, names, amps, chi, eta, left, right, "primary")
            key = (system, left, right)
            if key not in baseline:
                raise KeyError(f"missing baseline endpoint {key}")
            b = baseline[key]
            b_eta, b_A, b_k = float(b["eta"]), float(b["A"]), float(b["kappa"])
            r.update({
                "baseline_eta": b_eta, "baseline_A": b_A, "baseline_kappa": b_k,
                "delta_eta": float(r["eta"] - b_eta), "abs_delta_eta": abs(float(r["eta"] - b_eta)),
                "delta_kappa": float(r["coherence_kappa"] - b_k), "abs_delta_kappa": abs(float(r["coherence_kappa"] - b_k)),
                "baseline_noncancellation": b_k >= NONCANCELLATION_KAPPA,
            })
            if raw_names is not None and b_k >= NONCANCELLATION_KAPPA:
                ri, rj = raw_names.index(left), raw_names.index(right)
                raw_value = float(raw_eta[ri, rj])
                corrected = float(r["eta"])
                reversal = raw_value != 0.0 and corrected != 0.0 and np.signbit(raw_value) != np.signbit(corrected)
                r["raw_eta"] = raw_value
                r["raw_corrected_sign_reversal"] = bool(reversal)
                if reversal: raw_sign_reversals.append(key)
            rows.append(r)
        system_audit[system] = entry

    keys = [r for r in rows if r["baseline_noncancellation"]]
    eta0 = [float(r["baseline_eta"]) for r in keys]
    eta1 = [float(r["eta"]) for r in keys]
    max_de = max((float(r["abs_delta_eta"]) for r in keys), default=float("nan"))
    max_dk = max((float(r["abs_delta_kappa"]) for r in keys), default=float("nan"))
    rho = spearman(eta0, eta1)

    def row(system, a, b):
        return next(r for r in rows if r["system"] == system and r["coordinate_a"] == a and r["coordinate_b"] == b)
    qualitative = {}
    qualitative["water_dimer"] = (row("water_dimer","donor_OH","hydrogen_bond_HO")["eta"] < 0 and row("water_dimer","hydrogen_bond_HO","donor_acceptor_OO")["eta"] > 0 and row("water_dimer","hydrogen_bond_HO","donor_acceptor_OO")["coherence_kappa"] > 0.8)
    mcan = row("methane_dimer","CH_A_in_1","CH_B_in_1")
    mcoh = row("methane_dimer","CC","HH_A1_B1")
    qualitative["methane_dimer"] = (mcan["absolute_modal_activity_A"] > 0.5 and mcan["coherence_kappa"] < 0.1 and mcoh["eta"] > 0.1 and mcoh["coherence_kappa"] > 0.8)
    qualitative["chloride_water"] = (row("chloride_water","donor_OH","hydrogen_bond_HCl")["eta"] < 0 and row("chloride_water","hydrogen_bond_HCl","donor_acceptor_OCl")["eta"] > 0 and row("chloride_water","hydrogen_bond_HCl","donor_acceptor_OCl")["coherence_kappa"] > 0.8)
    f1, f2, f3 = row("fhf_minus","FH_left","HF_right"), row("fhf_minus","FH_left","FF"), row("fhf_minus","HF_right","FF")
    qualitative["fhf_minus"] = (f1["eta"] < 0 and f2["eta"] > 0 and f2["coherence_kappa"] > 0.8 and f3["eta"] > 0 and f3["coherence_kappa"] > 0.8)
    ab = [r for r in rows if r["system"] == "ammonia_borane"]
    qualitative["ammonia_borane"] = (all(r["eta"] < 0 for r in ab if r["coordinate_a"] == "NB") and all(r["coherence_kappa"] < 0.2 for r in ab if r["coordinate_a"].startswith("NH") and r["coordinate_b"].startswith("BH")))

    numerical = {
        s: bool(v.get("stationary_minimum_pass")) and v.get("direct_modal_max_relative", 1) <= 1e-8 and v.get("reciprocity_max_abs", 1) <= 1e-10 and v.get("eta_max_excess", 1) <= 1e-10
        for s, v in system_audit.items()
    }
    summary = {
        "frozen_design": "theory/stage3b_method_robustness_design_r1.md",
        "systems": system_audit,
        "numerical_admissibility": numerical,
        "qualitative_gates": qualitative,
        "raw_corrected_noncancellation_sign_reversals": [list(k) for k in raw_sign_reversals],
        "noncancellation_endpoint_count": len(keys),
        "max_abs_delta_eta": max_de,
        "max_abs_delta_kappa": max_dk,
        "spearman_signed_eta": rho,
        "quantitative_gates": {
            "max_abs_delta_eta_le_0p15": bool(max_de <= MAX_DELTA_ETA),
            "max_abs_delta_kappa_le_0p20": bool(max_dk <= MAX_DELTA_KAPPA),
            "spearman_ge_0p80": bool(rho >= MIN_SPEARMAN),
        },
    }
    summary["all_numerical_pass"] = all(numerical.values()) and not raw_sign_reversals
    summary["all_qualitative_pass"] = all(qualitative.values())
    summary["all_quantitative_pass"] = all(summary["quantitative_gates"].values())
    summary["classification"] = ("QUANTITATIVELY_ROBUST" if summary["all_numerical_pass"] and summary["all_qualitative_pass"] and summary["all_quantitative_pass"] else "QUALITATIVELY_ROBUST_ONLY" if summary["all_numerical_pass"] and summary["all_qualitative_pass"] else "CORE_QUALITATIVE_FAILURE")

    with (OUT / "stage3b_endpoint_comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = list(rows[0].keys())
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    (OUT / "stage3b_robustness_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["classification","all_numerical_pass","all_qualitative_pass","all_quantitative_pass","max_abs_delta_eta","max_abs_delta_kappa","spearman_signed_eta"]}, indent=2))

if __name__ == "__main__":
    main()
