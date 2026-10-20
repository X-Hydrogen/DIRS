from __future__ import annotations

import json
from pathlib import Path

import analysis.stage3a_unified_response_analysis as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis" / "stage3a_nine_system_outputs"
EXCLUDED = ["benzene_methane", "fluorobenzene_fluoroform"]


def main() -> None:
    # Reuse the frozen Stage 3A observable implementation without changing any
    # numerical tolerance or local-coordinate definition. Only the scientifically
    # rejected fluorobenzene-fluoroform source is removed from the accepted set.
    base.OUT = OUT
    base.OUT.mkdir(parents=True, exist_ok=True)
    base.ACCEPTED_SOURCES = {
        name: source
        for name, source in base.ACCEPTED_SOURCES.items()
        if name != "fluorobenzene_fluoroform"
    }
    base.main()

    summary_path = OUT / "stage3a_system_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["acceptance_decision"] = "theory/stage3a_acceptance_decision_r2.md"
    summary["accepted_system_count"] = len(base.ACCEPTED_SOURCES)
    summary["excluded_from_static_inversion"] = EXCLUDED
    summary["fluorobenzene_validation_disposition"] = (
        "excluded after preregistered analytical-vs-two-numerical-Hessian gate failure"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    gates_path = OUT / "stage3a_numerical_gates.json"
    gates = json.loads(gates_path.read_text(encoding="utf-8"))
    spectral = json.loads(
        (OUT / "stage3a_spectral_grid_audit.json").read_text(encoding="utf-8")
    )

    max_sum = max(float(row["finest_absolute_error"]) for row in spectral)
    max_mid_to_fine = max(
        float(row["refinement_difference_0p5_to_0p25"]) for row in spectral
    )
    retained = {
        key: bool(gates["pass"][key])
        for key in (
            "all_stationary",
            "direct_modal",
            "reciprocity",
            "eta_bound",
            "isotope_static_invariance",
        )
    }
    sum_tol = float(gates["declared_tolerances"]["spectral_sum_absolute"])
    refine_tol = float(
        gates["declared_tolerances"]["spectral_refinement_absolute"]
    )
    spectral_r2 = {
        "maximum_finest_sum_error": max_sum,
        "maximum_middle_to_finest_refinement_difference": max_mid_to_fine,
        "sum_tolerance": sum_tol,
        "refinement_tolerance": refine_tol,
        "sum_pass": max_sum <= sum_tol,
        "refinement_pass": max_mid_to_fine <= refine_tol,
    }
    overall_r2 = (
        all(retained.values())
        and spectral_r2["sum_pass"]
        and spectral_r2["refinement_pass"]
    )
    r2 = {
        "protocol": "theory/stage3a_numerical_r2_protocol.md",
        "acceptance_decision": "theory/stage3a_acceptance_decision_r2.md",
        "accepted_system_count": len(base.ACCEPTED_SOURCES),
        "excluded_from_static_inversion": EXCLUDED,
        "overall_numerical_pass": overall_r2,
        "retained_non_spectral_gates": retained,
        "spectral_r2": spectral_r2,
        "scientific_review_flags": gates.get("scientific_review_flags", {}),
    }
    (OUT / "stage3a_numerical_gates_r2.json").write_text(
        json.dumps(r2, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    systems = summary["systems"]
    review_flags = gates.get("scientific_review_flags", {})
    lines = [
        "# Stage 3A Nine-System DIRS Transferability Audit",
        "",
        "This rerun implements `theory/stage3a_acceptance_decision_r2.md` after the preregistered fluorobenzene···fluoroform numerical-Hessian validation failed. No tolerance, local coordinate, or response definition was changed.",
        "",
        "## Accepted primary set",
        "",
    ]
    for name in base.ACCEPTED_SOURCES:
        item = systems[name]
        lines.append(
            f"- `{name}`: minimum projected frequency {item['minimum_frequency_cm_minus1']:.6f} cm^-1; "
            f"stationary={item['stationary_minimum_pass']}; response={item['response_status']}."
        )
    lines.extend(
        [
            "",
            "## Explicit exclusions",
            "",
            "- `benzene_methane`: no accepted positive-definite stationary minimum after the frozen R2/R3 saddle-escape searches.",
            "- `fluorobenzene_fluoroform`: analytical soft mode was positive, but both preregistered fixed-geometry numerical Hessians were negative in the corresponding lowest projected curvature; excluded rather than rescued.",
            "",
            "## Numerical closure",
            "",
            f"- nine-system R2 overall numerical pass: {overall_r2}",
            f"- direct/modal: {retained['direct_modal']}",
            f"- reciprocity: {retained['reciprocity']}",
            f"- eta bound: {retained['eta_bound']}",
            f"- all accepted Hessians stationary: {retained['all_stationary']}",
            f"- isotope static invariance: {retained['isotope_static_invariance']}",
            f"- spectral signed-area maximum error: {max_sum:.6e} (tol {sum_tol:.1e})",
            f"- spectral middle-to-finest maximum difference: {max_mid_to_fine:.6e} (tol {refine_tol:.1e})",
            f"- raw-versus-rigid-corrected primary eta sign reversals retained for review: {review_flags.get('raw_regularized_primary_sign_reversal_count', 'NA')}",
            "",
            "## Interpretation boundary",
            "",
            "Passing the numerical gates establishes that the accepted nine-system harmonic-response dataset is internally computable and numerically closed under the frozen protocol. It does not by itself establish the final chemical narrative. Raw-versus-rigid-corrected sign reversals remain explicit review flags for the next curator decision and will be examined before manuscript-ready status is declared.",
            "",
        ]
    )
    (OUT / "STAGE3A_NINE_SYSTEM_TRANSFERABILITY_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    # The reused base implementation writes a legacy title into this filename.
    # Remove it so the branch cannot accidentally expose a misleading ten-system
    # report as the current decision artifact.
    legacy = OUT / "STAGE3A_TRANSFERABILITY_AUDIT.md"
    if legacy.exists():
        legacy.unlink()


if __name__ == "__main__":
    main()
