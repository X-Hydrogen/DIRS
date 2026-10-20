"""Cross-checks of the publication pipeline against independent formulations.

The reference route below never diagonalizes a Hessian and never uses masses:
it solves the Lagrange (KKT) problem

    min 1/2 x^T K x   subject to   b_A . x = 1,   R^T x = 0,

whose solution gives the relaxed compliance C_AA = 1 / (x^T K x) and
C_BA = (b_B . x) C_AA.  Agreement with sum_s c_As c_Bs / omega_s^2 from the
mass-weighted normal modes is therefore a test of the physics, not a repeat of
the same algebra.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import analysis.stage3a_unified_response_analysis as s3  # noqa: E402

CALC = ROOT / "calculations"
SOURCES = {
    **{(n, "pbe0"): CALC / "prototypes_stage2/results/compact" / f"{n}.npz"
       for n in ("water_dimer", "methane_dimer", "fhf_minus")},
    **{(n, "pbe0"): CALC / "benchmark_stage3/results/compact" / f"{n}.npz"
       for n in ("hf_dimer", "hcn_dimer", "nh3_hf", "chloride_water", "h5o2_plus",
                 "ammonia_borane")},
    **{(n, "wb97x"): CALC / "stage3b_method_robustness/results/compact" / f"{n}_wb97x_d4.npz"
       for n in ("water_dimer", "methane_dimer", "chloride_water", "fhf_minus",
                 "ammonia_borane")},
}
HEXAMER = CALC / "revision_r1_water_hexamer_network/results/compact/water_hexamer_book_pbe0d4.npz"
HEX_GRAPH = CALC / "revision_r1_water_hexamer_network/results/preregistered_endpoint_graph.json"


def _euclidean_rigid_basis(pos: np.ndarray) -> np.ndarray:
    n = len(pos)
    centre = pos.mean(axis=0)
    cols = [np.tile(e, n) for e in np.eye(3)]
    cols += [np.cross(e, pos - centre).ravel() for e in np.eye(3)]
    u, s, _ = np.linalg.svd(np.array(cols).T, full_matrices=False)
    return u[:, : int(np.sum(s > 1e-8 * s[0]))]


def _distance_row(pos: np.ndarray, i: int, j: int) -> np.ndarray:
    e = (pos[i] - pos[j]) / np.linalg.norm(pos[i] - pos[j])
    row = np.zeros(pos.size)
    row[3 * i:3 * i + 3] = e
    row[3 * j:3 * j + 3] = -e
    return row


def _kkt_compliance(K: np.ndarray, pos: np.ndarray, rows: np.ndarray) -> np.ndarray:
    R = _euclidean_rigid_basis(pos)
    n, nr, nc = K.shape[0], R.shape[1], rows.shape[0]
    C = np.zeros((nc, nc))
    for a in range(nc):
        A = np.zeros((n + 1 + nr, n + 1 + nr))
        A[:n, :n] = 0.5 * (K + K.T)
        A[:n, n] = A[n, :n] = rows[a]
        A[:n, n + 1:] = R
        A[n + 1:, :n] = R.T
        rhs = np.zeros(n + 1 + nr)
        rhs[n] = 1.0
        x = np.linalg.solve(A, rhs)[:n]
        caa = 1.0 / float(x @ A[:n, :n] @ x)
        C[:, a] = (rows @ x) * caa
    return C


def _raw_record(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return np.array(data["positions_bohr"]), np.array(data["hessian_hartree_per_bohr2"])


def _pipeline_chi(path: Path, label: str, definitions: dict[str, list[int]]):
    registry = dict(s3.ACCEPTED_SOURCES)
    s3.ACCEPTED_SOURCES[label] = path
    try:
        hs = s3.analyse_hessian(label)
    finally:
        s3.ACCEPTED_SOURCES.clear()
        s3.ACCEPTED_SOURCES.update(registry)
    names, reduced, amplitudes, _ = s3.coordinate_data(hs, definitions)
    _, chi, _ = s3.static_matrices(hs, reduced, amplitudes)
    return names, chi, hs


def _hexamer_definitions() -> dict[str, list[int]]:
    graph = json.loads(HEX_GRAPH.read_text(encoding="utf-8"))
    return {
        f'{e["endpoint_id"]}_{side}': [int(a) - 1 for a in e[f"coord_{side}"]["atoms"]]
        for e in graph["endpoints"] for side in "AB"
    }


CASES = [(f"{n}:{m}", p, None) for (n, m), p in SOURCES.items()] + [
    ("water_hexamer_book", HEXAMER, "hexamer")
]


@pytest.mark.parametrize("label,path,kind", CASES, ids=[c[0] for c in CASES])
def test_modal_compliance_equals_constrained_relaxation(label, path, kind):
    maps = s3.load_coordinate_maps()
    definitions = _hexamer_definitions() if kind else maps[label.split(":")[0]]["coordinates"]
    names, chi, _ = _pipeline_chi(path, f"audit_{label}", definitions)
    pos, K = _raw_record(path)
    rows = np.array([_distance_row(pos, *definitions[n]) for n in names])
    reference = _kkt_compliance(K, pos, rows)
    scale = float(np.max(np.abs(reference)))
    assert np.max(np.abs(chi - reference)) / scale < 1e-9


def test_compliance_is_invariant_to_completion_of_internal_coordinates():
    """Decius: C = F^-1 for a complete set does not depend on the other
    coordinates, whereas the off-diagonal force constant does."""
    pos, K = _raw_record(SOURCES[("water_dimer", "pbe0")])
    definitions = s3.load_coordinate_maps()["water_dimer"]["coordinates"]
    a = _distance_row(pos, *definitions["hydrogen_bond_HO"])
    b = _distance_row(pos, *definitions["donor_acceptor_OO"])
    pairs = list(itertools.combinations(range(len(pos)), 2))
    results = []
    for seed in (1, 2, 3):
        order = np.random.default_rng(seed).permutation(len(pairs))
        chosen, basis = [a, b], np.linalg.qr(np.array([a, b]).T)[0]
        for k in order:
            r = _distance_row(pos, *pairs[k])
            resid = r - basis @ (basis.T @ r)
            if np.linalg.norm(resid) > 0.2:
                chosen.append(r)
                basis = np.column_stack([basis, resid / np.linalg.norm(resid)])
        B = np.array(chosen)
        assert B.shape[0] == 3 * len(pos) - 6
        Bplus = B.T @ np.linalg.inv(B @ B.T)
        F = Bplus.T @ K @ Bplus
        C = np.linalg.inv(0.5 * (F + F.T))
        results.append((C[0, 1], F[0, 1]))
    compliances = np.array([x[0] for x in results])
    force_constants = np.array([x[1] for x in results])
    assert np.ptp(compliances) / abs(compliances[0]) < 1e-10
    assert np.ptp(force_constants) > 1e-3 * abs(force_constants).max()
    names, chi, _ = _pipeline_chi(SOURCES[("water_dimer", "pbe0")], "audit_wd", definitions)
    assert compliances[0] == pytest.approx(
        chi[names.index("hydrogen_bond_HO"), names.index("donor_acceptor_OO")], rel=1e-10
    )


def test_functional_control_gate_summary_is_reproduced():
    """Recompute the published Stage 3B gates rather than only the 25 rows."""
    maps = s3.load_coordinate_maps()
    systems = ("water_dimer", "methane_dimer", "chloride_water", "fhf_minus", "ammonia_borane")
    rows = {}
    for method in ("pbe0", "wb97x"):
        for name in systems:
            definitions = maps[name]["coordinates"]
            names, chi, hs = _pipeline_chi(SOURCES[(name, method)], f"audit_{name}_{method}", definitions)
            amplitudes = s3.coordinate_data(hs, definitions)[2]
            eta = chi / np.sqrt(np.outer(np.diag(chi), np.diag(chi)))
            for left, right in maps[name]["primary_cross_pairs"]:
                metric, _ = s3.pair_metrics(hs, names, amplitudes, chi, eta, left, right, "audit")
                rows[(method, name, left, right)] = metric
    keys = [k[1:] for k in rows if k[0] == "pbe0"]
    noncancel = [k for k in keys if rows[("pbe0", *k)]["coherence_kappa"] >= 0.2]
    d_eta = [abs(rows[("wb97x", *k)]["eta"] - rows[("pbe0", *k)]["eta"]) for k in noncancel]
    d_kappa = [abs(rows[("wb97x", *k)]["coherence_kappa"] - rows[("pbe0", *k)]["coherence_kappa"])
               for k in noncancel]
    rho = stats.spearmanr([rows[("pbe0", *k)]["eta"] for k in noncancel],
                          [rows[("wb97x", *k)]["eta"] for k in noncancel]).statistic
    ref = json.loads((ROOT / "reference/stage3b_robustness_summary.json").read_text(encoding="utf-8"))
    assert len(keys) == 25
    assert len(noncancel) == ref["noncancellation_endpoint_count"]
    assert max(d_eta) == pytest.approx(ref["max_abs_delta_eta"], rel=1e-6)
    assert max(d_kappa) == pytest.approx(ref["max_abs_delta_kappa"], rel=1e-6)
    assert rho == pytest.approx(ref["spearman_signed_eta"], rel=1e-9)

    w = lambda *k: rows[("wb97x", *k)]  # noqa: E731
    qualitative = {
        "water_dimer": w("water_dimer", "donor_OH", "hydrogen_bond_HO")["eta"] < 0
        and w("water_dimer", "hydrogen_bond_HO", "donor_acceptor_OO")["eta"] > 0
        and w("water_dimer", "hydrogen_bond_HO", "donor_acceptor_OO")["coherence_kappa"] > 0.8,
        "methane_dimer": w("methane_dimer", "CH_A_in_1", "CH_B_in_1")["absolute_modal_activity_A"] > 0.5
        and w("methane_dimer", "CH_A_in_1", "CH_B_in_1")["coherence_kappa"] < 0.1
        and w("methane_dimer", "CC", "HH_A1_B1")["eta"] > 0.1
        and w("methane_dimer", "CC", "HH_A1_B1")["coherence_kappa"] > 0.8,
        "chloride_water": w("chloride_water", "donor_OH", "hydrogen_bond_HCl")["eta"] < 0
        and w("chloride_water", "hydrogen_bond_HCl", "donor_acceptor_OCl")["eta"] > 0
        and w("chloride_water", "hydrogen_bond_HCl", "donor_acceptor_OCl")["coherence_kappa"] > 0.8,
        "fhf_minus": w("fhf_minus", "FH_left", "HF_right")["eta"] < 0
        and all(w("fhf_minus", c, "FF")["eta"] > 0 and w("fhf_minus", c, "FF")["coherence_kappa"] > 0.8
                for c in ("FH_left", "HF_right")),
        "ammonia_borane": all(rows[k]["eta"] < 0 for k in rows
                              if k[:2] == ("wb97x", "ammonia_borane") and k[2] == "NB")
        and all(rows[k]["coherence_kappa"] < 0.2 for k in rows
                if k[:2] == ("wb97x", "ammonia_borane") and k[2].startswith("NH") and k[3].startswith("BH")),
    }
    assert qualitative == ref["qualitative_gates"]


def test_nine_system_spectral_and_isotope_gates_are_reproduced(tmp_path, monkeypatch):
    import analysis.stage3a_nine_system_reanalysis as nine

    monkeypatch.setattr(s3, "ACCEPTED_SOURCES", dict(s3.ACCEPTED_SOURCES))
    monkeypatch.setattr(s3, "OUT", s3.OUT)
    monkeypatch.setattr(nine, "OUT", tmp_path)
    nine.main()
    observed = json.loads((tmp_path / "stage3a_numerical_gates_r2.json").read_text(encoding="utf-8"))
    ref = json.loads((ROOT / "reference/stage3a_nine_system_gates.json").read_text(encoding="utf-8"))
    assert observed["overall_numerical_pass"] is True
    assert observed["accepted_system_count"] == ref["accepted_system_count"] == 9
    assert observed["retained_non_spectral_gates"] == ref["retained_non_spectral_gates"]
    assert observed["scientific_review_flags"] == ref["scientific_review_flags"]
    for key in ("maximum_finest_sum_error", "maximum_middle_to_finest_refinement_difference"):
        assert observed["spectral_r2"][key] == pytest.approx(ref["spectral_r2"][key], rel=1e-6)


def test_hexamer_endpoint_graph_is_regenerated_from_geometry(tmp_path):
    out = tmp_path / "graph.json"
    xyz = CALC / "revision_r1_water_hexamer_network/results/water_hexamer_book_pbe0d4.xyz"
    subprocess.run([sys.executable, str(ROOT / "analysis/revision_water_hexamer_prepare_endpoints.py"),
                    str(xyz), "--out", str(out)], check=True, capture_output=True)
    regenerated = json.loads(out.read_text(encoding="utf-8"))
    frozen = json.loads(HEX_GRAPH.read_text(encoding="utf-8"))
    # The graph identity is discrete and must match EXACTLY; distances and
    # angles are independently recalculated floats and can differ by a few ULP.
    for key in ("thresholds", "water_assignment", "endpoints", "endpoint_class_counts"):
        assert regenerated[key] == frozen[key]
    assert len(regenerated["contacts"]) == len(frozen["contacts"]) == 7
    for calculated, archived in zip(regenerated["contacts"], frozen["contacts"]):
        assert set(calculated) == set(archived)
        for key, expected in archived.items():
            if isinstance(expected, float):
                assert calculated[key] == pytest.approx(expected, rel=1e-12, abs=1e-10)
            else:
                assert calculated[key] == expected


def test_reproduction_cli_restores_the_primary_source_registry(tmp_path):
    spec = importlib.util.spec_from_file_location("reproduce_publication",
                                                  ROOT / "scripts/reproduce_publication.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    before = dict(cli.s3.ACCEPTED_SOURCES)
    cli.run_hexamer(tmp_path)
    cli.run_functional(tmp_path)
    assert cli.s3.ACCEPTED_SOURCES == before


@pytest.mark.xfail(strict=True, reason=(
    "Known limitation (docs/REPRODUCIBILITY.md): the HCN dimer is linear to "
    "2e-6 angstrom but the frozen LINEARITY_TOLERANCE=1e-8 classifies it as "
    "nonlinear, so one bending component is projected out as a rotation."))
def test_linear_hcn_dimer_keeps_all_thirteen_vibrations():
    hs = s3.analyse_hessian("hcn_dimer")
    assert hs.rigid_rank == 5
    assert len(hs.eigenvalues) == 13
