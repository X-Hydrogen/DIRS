#!/usr/bin/env python3
"""Non-invasive audit of HCN dimer near-linearity and coordinate response.

Uses the independent raw-ORCA Hessian reader and two hypothetical rigid-mode
rank conventions. The published values, Hessian, and coordinates are never
modified; forcing a linear projection on a noncollinear finite geometry is
only a sensitivity diagnostic, not a correction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import independent_check as audit  # noqa: E402


def main() -> None:
    source = audit.PRIMARY_HESS["hcn_dimer"]
    data = audit.read_hess(source)
    coords = audit.coordinate_maps()["hcn_dimer"]["coordinates"]
    names = list(coords)
    rows = np.array([audit.distance_row(data["pos"], *coords[n]) for n in names])
    K, pos, masses = data["K"], data["pos"], data["masses"]

    e6, _, singular_e, rank6 = audit.orthonormal_rigid(pos, np.ones(len(pos)), rel_tol=1e-8)
    _, _, singular_m, mass_rank6 = audit.orthonormal_rigid(pos, masses, rel_tol=1e-8)
    original_lam, original_amplitudes, _ = audit.route_modes(K, pos, masses, rows)

    e5, _, _, rank5 = audit.orthonormal_rigid(pos, np.ones(len(pos)), rel_tol=1e-5)
    _, z5, _, mass_rank5 = audit.orthonormal_rigid(pos, masses, rel_tol=1e-5)
    m = np.repeat(masses * audit.AMU_ME, 3)
    p5 = np.eye(len(m)) - e5 @ e5.T
    k5 = p5 @ (0.5 * (K + K.T)) @ p5
    d5 = k5 / np.sqrt(np.outer(m, m))
    lam5, u5 = np.linalg.eigh(0.5 * (z5.T @ d5 @ z5 + z5.T @ d5.T @ z5))
    amplitudes5 = (z5 @ u5).T @ (rows / np.sqrt(m)[None, :]).T

    result = {
        "input": str(source.relative_to(ROOT)),
        "frozen_rank_tolerance": 1e-8,
        "frozen_euclidean_rigid_rank": rank6,
        "frozen_mass_weighted_rigid_rank": mass_rank6,
        "frozen_vibrational_mode_count": len(original_lam),
        "forced_rank_tolerance": 1e-5,
        "forced_euclidean_rigid_rank": rank5,
        "forced_mass_weighted_rigid_rank": mass_rank5,
        "forced_vibrational_mode_count": len(lam5),
        "forced_minimum_projected_eigenvalue": float(lam5.min()),
        "forced_positive_vibrational_curvature": bool(np.all(lam5 > 0.0)),
        "euclidean_singular_values_rel": singular_e.tolist(),
        "mass_weighted_singular_values_rel": singular_m.tolist(),
        "endpoints": [],
        "interpretation": (
            "The archived, slightly noncollinear geometry is nonlinear with "
            "six rigid degrees of freedom. Forced rank five is only a "
            "counterfactual; it does not replace the published response."
        ),
    }
    if rank6 != 6 or mass_rank6 != 6:
        raise AssertionError("Archive baseline does not have frozen six-mode rigidity")
    if rank5 != 5 or mass_rank5 != 5:
        result["warning"] = "Looser tolerance did not produce a consistent rank-five counterfactual"
    elif np.any(lam5 <= 0.0):
        result["warning"] = (
            "Forced rank five includes a nonpositive-curvature direction. "
            "Its static harmonic response is not an admissible alternative."
        )
    else:
        references = audit.read_csv(ROOT / "reference/compliance_37_endpoints.csv")
        for row in references:
            if row["system"] != "hcn_dimer":
                continue
            a, c = names.index(row["coordinate_a"]), names.index(row["coordinate_b"])
            d0 = audit.descriptors(original_lam, original_amplitudes, a, c)
            d5 = audit.descriptors(lam5, amplitudes5, a, c)
            result["endpoints"].append({
                "coordinate_a": row["coordinate_a"],
                "coordinate_b": row["coordinate_b"],
                "relative_chi_change": audit.rel(d5["chi"], d0["chi"]),
                **{f"absolute_{k}_change": abs(d5[k] - d0[k])
                   for k in ("eta", "A", "kappa", "low", "mid", "high")},
            })
    path = ROOT / "reproduced" / "hcn_rank_sensitivity.json"
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
