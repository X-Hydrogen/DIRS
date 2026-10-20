#!/usr/bin/env python3
"""Independent re-derivation of every published DIRS number from raw ORCA text.

This program deliberately does NOT import the ``dirs`` package or the
``analysis`` scripts.  It re-implements, from the definitions only:

* an ORCA ``.hess`` text parser (``$hessian``, ``$atoms``,
  ``$vibrational_frequencies``);
* Euclidean (mass-free) and Eckart (mass-weighted) rigid-motion bases;
* interatomic-distance Wilson rows, checked against central finite differences;
* three mathematically distinct routes to the static compliance C_AB:
    route M  mass-weighted normal modes, sum_s c_As c_Bs / omega_s^2;
    route K  Lagrange/KKT constrained relaxation of the raw Cartesian Hessian
             (no eigen-decomposition, no masses, no pseudo-inverse);
    route W  Wilson/Decius internal-coordinate route: F = (B^+)^T K B^+ for a
             complete non-redundant internal-coordinate set, C = F^-1, with two
             different completions to demonstrate completion invariance;
* sensitivity of every endpoint to the rigid-motion treatment;
* degenerate-subspace bounds on the basis-dependent descriptors A and kappa,
  and their ranges under random symmetric Hessian perturbations;
* isotope perturbation tests, spectral sum rule, unit conversions,
  geometry/Hessian provenance and the functional-control gates.

Reference CSV files are read only AFTER the computation, for comparison.
Run from the repository root:  python scripts/independent_check.py
Outputs are written to reproduced/independent_check/.
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, linalg, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduced" / "independent_check"
CALC = ROOT / "calculations"

# CODATA 2018 (NIST SP 961, 2019)
BOHR_ANGSTROM = 0.529177210903
HARTREE_J = 4.3597447222071e-18
AMU_ME = 1822.888486209
HARTREE_INV_CM = 219474.63136320

PRIMARY_HESS = {
    **{n: CALC / "prototypes_stage2/results/hessians" / f"{n}.hess"
       for n in ("water_dimer", "methane_dimer", "fhf_minus")},
    **{n: CALC / "benchmark_stage3/results/hessians" / f"{n}.hess"
       for n in ("hf_dimer", "hcn_dimer", "nh3_hf", "chloride_water",
                 "h5o2_plus", "ammonia_borane")},
}
FUNCTIONAL_HESS = {
    n: CALC / "stage3b_method_robustness/results/hessians" / f"{n}_wb97x_d4.hess"
    for n in ("water_dimer", "methane_dimer", "chloride_water", "fhf_minus",
              "ammonia_borane")
}
HEX_DIR = CALC / "revision_r1_water_hexamer_network"
HEX_HESS = HEX_DIR / "results/water_hexamer_book_pbe0d4.hess"

# Element masses as actually printed by archived ORCA 6.1.1 .hess files.
# Chlorine is 35.453 in the ORCA calculations (not the rounded 35.45).
STANDARD_WEIGHTS = {"H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007,
                    "O": 15.999, "F": 18.998, "Cl": 35.453}


# --------------------------------------------------------------------------
# ORCA text parsing (independent implementation)
# --------------------------------------------------------------------------
def _section(lines: list[str], tag: str) -> int:
    hits = [i for i, l in enumerate(lines) if l.strip() == tag]
    if len(hits) != 1:
        raise ValueError(f"{tag}: {len(hits)} occurrences")
    return hits[0]


def _block_matrix(lines: list[str], start: int, nrow: int, ncol: int) -> np.ndarray:
    out = np.full((nrow, ncol), np.nan)
    i = start
    filled = 0
    while filled < ncol:
        while not lines[i].strip():
            i += 1
        cols = [int(t) for t in lines[i].split()]
        i += 1
        for r in range(nrow):
            parts = lines[i].split()
            assert int(parts[0]) == r
            out[r, cols] = [float(x) for x in parts[1:]]
            i += 1
        filled += len(cols)
    if np.isnan(out).any():
        raise ValueError("unfilled matrix element")
    return out


def read_hess(path: Path) -> dict:
    lines = path.read_text().splitlines()
    h = _section(lines, "$hessian")
    dim = int(lines[h + 1].split()[0])
    hess = _block_matrix(lines, h + 2, dim, dim)
    a = _section(lines, "$atoms")
    nat = int(lines[a + 1].split()[0])
    sym, mass, pos = [], [], []
    for l in lines[a + 2:a + 2 + nat]:
        p = l.split()
        sym.append(p[0]); mass.append(float(p[1])); pos.append([float(x) for x in p[2:5]])
    v = _section(lines, "$vibrational_frequencies")
    nv = int(lines[v + 1].split()[0])
    freqs = np.array([float(lines[v + 2 + k].split()[1]) for k in range(nv)])
    return {"symbols": sym, "masses": np.array(mass), "pos": np.array(pos),
            "K": hess, "orca_freqs": freqs}


def read_xyz(path: Path) -> tuple[list[str], np.ndarray, str]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    sym, xyz = [], []
    for l in lines[2:2 + n]:
        p = l.split()
        sym.append(p[0]); xyz.append([float(x) for x in p[1:4]])
    return sym, np.array(xyz), lines[1]


def read_inp_geometry(path: Path) -> tuple[list[str], np.ndarray, str]:
    text = path.read_text()
    m = re.search(r"^\*\s*xyz\s+(-?\d+)\s+(\d+)\s*\n(.*?)^\*", text, re.S | re.M)
    sym, xyz = [], []
    for l in m.group(3).strip().splitlines():
        p = l.split()
        sym.append(p[0]); xyz.append([float(x) for x in p[1:4]])
    keywords = " ".join(l for l in text.splitlines() if l.startswith("!"))
    return sym, np.array(xyz), keywords


# --------------------------------------------------------------------------
# Geometry, rigid motion and coordinates
# --------------------------------------------------------------------------
def rigid_vectors(pos: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Columns: 3 translations and 3 infinitesimal rotations, sqrt(w)-scaled."""
    n = len(pos)
    sw = np.sqrt(weights)
    centre = (weights[:, None] * pos).sum(0) / weights.sum()
    cols = []
    for e in np.eye(3):
        t = np.zeros((n, 3)); t[:] = e
        cols.append((sw[:, None] * t).ravel())
    for e in np.eye(3):
        cols.append((sw[:, None] * np.cross(e, pos - centre)).ravel())
    return np.array(cols).T


def orthonormal_rigid(pos: np.ndarray, weights: np.ndarray, rel_tol: float = 1e-8):
    v = rigid_vectors(pos, weights)
    u, s, _ = np.linalg.svd(v, full_matrices=True)
    rank = int(np.sum(s > rel_tol * s[0]))
    return u[:, :rank], u[:, rank:], s / s[0], rank


def distance_row(pos: np.ndarray, i: int, j: int) -> np.ndarray:
    d = pos[i] - pos[j]
    e = d / np.linalg.norm(d)
    row = np.zeros(pos.size)
    row[3 * i:3 * i + 3] = e
    row[3 * j:3 * j + 3] = -e
    return row


def distance_row_fd(pos: np.ndarray, i: int, j: int, h: float = 1e-5) -> np.ndarray:
    row = np.zeros(pos.size)
    flat = pos.ravel()
    for k in range(pos.size):
        p, m = flat.copy(), flat.copy()
        p[k] += h; m[k] -= h
        rp = np.linalg.norm(p.reshape(-1, 3)[i] - p.reshape(-1, 3)[j])
        rm = np.linalg.norm(m.reshape(-1, 3)[i] - m.reshape(-1, 3)[j])
        row[k] = (rp - rm) / (2 * h)
    return row


# --------------------------------------------------------------------------
# Three independent routes to the static compliance
# --------------------------------------------------------------------------
def route_modes(K, pos, masses_amu, rows, *, treatment="euclid_then_eckart"):
    """Mass-weighted normal modes; returns eigenvalues (a.u.) and amplitudes."""
    m = np.repeat(masses_amu * AMU_ME, 3)
    Ks = 0.5 * (K + K.T)
    if treatment == "euclid_then_eckart":
        Qe, _, _, _ = orthonormal_rigid(pos, np.ones(len(pos)))
        P = np.eye(len(m)) - Qe @ Qe.T
        Ks = P @ Ks @ P
    D = Ks / np.sqrt(np.outer(m, m))
    D = 0.5 * (D + D.T)
    G = rows / np.sqrt(m)[None, :]
    if treatment == "full_space_drop_smallest":
        lam, L = np.linalg.eigh(D)
        nr = orthonormal_rigid(pos, masses_amu)[3]
        keep = np.sort(np.argsort(np.abs(lam))[nr:])
        lam, L = lam[keep], L[:, keep]
    else:
        _, Zm, _, _ = orthonormal_rigid(pos, masses_amu)
        lam, U = np.linalg.eigh(Zm.T @ D @ Zm)
        L = Zm @ U
    amps = L.T @ G.T  # (n_modes, n_coords)
    return lam, amps, L


def compliance_from_modes(lam, amps):
    return amps.T @ (amps / lam[:, None])


def route_kkt(K, pos, rows):
    """min 1/2 x^T K x  s.t.  b_A . x = 1 and R_e^T x = 0  (Lagrange/KKT)."""
    Qe, _, _, nr = orthonormal_rigid(pos, np.ones(len(pos)))
    Ks = 0.5 * (K + K.T)
    n = Ks.shape[0]
    nc = rows.shape[0]
    C = np.zeros((nc, nc))
    for a in range(nc):
        A = np.zeros((n + 1 + nr, n + 1 + nr))
        A[:n, :n] = Ks
        A[:n, n] = rows[a]; A[n, :n] = rows[a]
        A[:n, n + 1:] = Qe; A[n + 1:, :n] = Qe.T
        rhs = np.zeros(n + 1 + nr); rhs[n] = 1.0
        x = np.linalg.solve(A, rhs)[:n]
        caa = 1.0 / float(x @ Ks @ x)
        C[:, a] = (rows @ x) * caa
    return C


def complete_internal_set(pos, rows_first, seed):
    """Complete non-redundant internal coordinates: distances first, then
    rigid-free linear combinations for any missing (bend/torsion) rank."""
    n = len(pos)
    _, Ze, _, _ = orthonormal_rigid(pos, np.ones(n))
    nvib = Ze.shape[1]
    pairs = list(itertools.combinations(range(n), 2))
    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    candidates = list(rows_first) + [distance_row(pos, i, j) for i, j in pairs]
    chosen: list[np.ndarray] = []
    basis = np.zeros((pos.size, 0))
    for k, r in enumerate(candidates):
        resid = r - basis @ (basis.T @ r)
        # Forced endpoint rows only need to be independent; extra distances
        # must be well conditioned so that F = (B^+)^T K B^+ is not ill-posed.
        threshold = 1e-8 if k < len(rows_first) else 0.2
        if np.linalg.norm(resid) > threshold * np.linalg.norm(r):
            chosen.append(r)
            basis = np.column_stack([basis, resid / np.linalg.norm(resid)])
        if len(chosen) == nvib:
            break
    if len(chosen) < nvib:
        Bc = np.array(chosen)
        resid = Ze - Bc.T @ np.linalg.lstsq(Bc.T, Ze, rcond=None)[0] if chosen else Ze
        u, s, _ = np.linalg.svd(resid, full_matrices=False)
        for k in range(nvib - len(chosen)):
            chosen.append(u[:, k])
    B = np.array(chosen)
    assert np.linalg.matrix_rank(B, tol=1e-8) == nvib
    return B


def route_wilson(K, pos, masses_amu, rows, seed):
    """Decius compliance C = F^-1 in a complete internal set, transformed to
    the requested (possibly redundant) coordinates; also GF eigenvalues."""
    independent = []
    basis = np.zeros((pos.size, 0))
    for r in rows:
        resid = r - basis @ (basis.T @ r)
        if np.linalg.norm(resid) > 1e-3 * np.linalg.norm(r):
            independent.append(r)
            basis = np.column_stack([basis, resid / np.linalg.norm(resid)])
    B = complete_internal_set(pos, independent, seed)
    Bplus = B.T @ np.linalg.inv(B @ B.T)
    F = Bplus.T @ (0.5 * (K + K.T)) @ Bplus
    F = 0.5 * (F + F.T)
    Cint = np.linalg.inv(F)
    alpha = np.linalg.lstsq(B.T, rows.T, rcond=None)[0]  # rows^T = B^T alpha
    resid = float(np.max(np.abs(B.T @ alpha - rows.T)))
    C = alpha.T @ Cint @ alpha
    Gm = B @ np.diag(1.0 / np.repeat(masses_amu * AMU_ME, 3)) @ B.T
    gf = np.sort(np.linalg.eigvals(Gm @ F).real)
    return C, gf, resid, F[0, 1] if F.shape[0] > 1 else float("nan")


# --------------------------------------------------------------------------
# Descriptors
# --------------------------------------------------------------------------
def descriptors(lam, amps, i, j):
    chi = compliance_from_modes(lam, amps)
    w = amps[:, i] * amps[:, j] / lam
    norm = math.sqrt(chi[i, i] * chi[j, j])
    nw = w / norm
    A = float(np.abs(nw).sum())
    nu = np.sign(lam) * np.sqrt(np.abs(lam)) * HARTREE_INV_CM
    absw = np.abs(nw)
    frac = lambda mask: float(absw[mask].sum() / absw.sum())
    return {
        "chi": float(chi[i, j]), "eta": float(chi[i, j] / norm), "A": A,
        "kappa": abs(float(chi[i, j] / norm)) / A,
        "low": frac(nu < 200), "mid": frac((nu >= 200) & (nu < 1000)),
        "high": frac(nu >= 1000), "chi_aa": float(chi[i, i]), "chi_bb": float(chi[j, j]),
    }


def degenerate_bounds(lam, amps, i, j, delta_cm):
    """Diagnostic equal-frequency proxy for rotations of NEAR-degenerate modes.

    Frequency gaps < delta_cm create clusters, which are hypothetically
    collapsed to one shared eigenvalue to explore rotations. These are
    rigorous bounds ONLY in an EXACTLY degenerate subspace, not rigorous
    bounds for the actual nondegenerate Hessian (nor for arbitrary noise).
    Do not present the proxy extrema as certified physical bounds.
    """
    chi = compliance_from_modes(lam, amps)
    norm = math.sqrt(chi[i, i] * chi[j, j])
    nu = np.sqrt(lam) * HARTREE_INV_CM
    order = np.argsort(nu)
    clusters, cur = [], [order[0]]
    for a, b in zip(order[:-1], order[1:]):
        if nu[b] - nu[a] < delta_cm:
            cur.append(b)
        else:
            clusters.append(cur); cur = [b]
    clusters.append(cur)
    lo = hi = 0.0
    largest = 1
    for c in clusters:
        c = np.array(c)
        lam_c = float(np.mean(lam[c]))
        if len(c) == 1:
            v = abs(amps[c[0], i] * amps[c[0], j] / lam[c[0]])
            lo += v; hi += v
        else:
            largest = max(largest, len(c))
            # Hypothetical equal-frequency cluster; never a rigorous bound
            # for unequal actual eigenvalues.
            lo += abs(float(np.dot(amps[c, i], amps[c, j]))) / lam_c
            hi += float(np.linalg.norm(amps[c, i]) * np.linalg.norm(amps[c, j])) / lam_c
    eta = chi[i, j] / norm
    return {"A_equal_freq_proxy_lower": lo / norm,
            "A_equal_freq_proxy_upper": hi / norm,
            "kappa_equal_freq_proxy_upper": abs(eta) / (lo / norm) if lo > 0 else float("inf"),
            "kappa_equal_freq_proxy_lower": abs(eta) / (hi / norm),
            "largest_cluster": largest}


# --------------------------------------------------------------------------
# Hessian diagnostics
# --------------------------------------------------------------------------
def hessian_diagnostics(K, pos):
    n = len(pos)
    Ks = 0.5 * (K + K.T)
    T = rigid_vectors(pos, np.ones(n))[:, :3] / math.sqrt(n)
    R = rigid_vectors(pos, np.ones(n))[:, 3:]
    trans_res = float(np.max(np.abs(Ks @ T)))
    # Rotational covariance of the gradient: sum_j K_ij (e x R_j) = e x g_i.
    KR = (Ks @ R).reshape(n, 3, 3)  # atom, component, axis
    g_est_a = np.stack([KR[:, 1, 2], KR[:, 2, 0], KR[:, 0, 1]], axis=1)
    g_est_b = -np.stack([KR[:, 2, 1], KR[:, 0, 2], KR[:, 1, 0]], axis=1)
    diag_res = np.max(np.abs(np.stack([KR[:, 0, 0], KR[:, 1, 1], KR[:, 2, 2]])))
    return {
        "max_asymmetry": float(np.max(np.abs(K - K.T))),
        "max_abs_hessian": float(np.max(np.abs(K))),
        "translational_sum_rule_max_residual": trans_res,
        "rotation_residual_max": float(np.max(np.abs(KR))),
        "rotation_diagonal_residual_max": float(diag_res),
        "implied_gradient_max_estimate_a": float(np.max(np.abs(g_est_a))),
        "implied_gradient_max_estimate_b": float(np.max(np.abs(g_est_b))),
        "implied_gradient_estimates_disagreement": float(np.max(np.abs(g_est_a - g_est_b))),
    }


# --------------------------------------------------------------------------
# Main audit
# --------------------------------------------------------------------------
def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def coordinate_maps():
    a = json.loads((CALC / "prototypes_stage2/local_coordinates.json").read_text())["systems"]
    b = json.loads((CALC / "benchmark_stage3/local_coordinates.json").read_text())["systems"]
    return {**b, **a}


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def analyse_system(label, data, names, defs, endpoints, deltas=(0.1, 1.0, 5.0)):
    pos, masses, K = data["pos"], data["masses"], data["K"]
    rows = np.array([distance_row(pos, *defs[n]) for n in names])
    rows_fd = np.array([distance_row_fd(pos, *defs[n]) for n in names])
    _, _, sv_e, rank_e = orthonormal_rigid(pos, np.ones(len(pos)))
    _, _, sv_m, rank_m = orthonormal_rigid(pos, masses)

    lam, amps, L = route_modes(K, pos, masses, rows)
    C_M = compliance_from_modes(lam, amps)
    C_K = route_kkt(K, pos, rows)
    C_W1, gf1, res1, f01_1 = route_wilson(K, pos, masses, rows, seed=1)
    C_W2, gf2, res2, f01_2 = route_wilson(K, pos, masses, rows, seed=2)

    alt = {}
    for t in ("mass_weighted_projection_of_raw", "full_space_drop_smallest"):
        try:
            l2, a2, _ = route_modes(K, pos, masses, rows, treatment=t)
            alt[t] = (l2, a2)
        except Exception as exc:  # pragma: no cover
            alt[t] = exc
    nu = np.sqrt(lam) * HARTREE_INV_CM
    scale = float(np.max(np.abs(C_M)))
    summary = {
        "system": label,
        "n_atoms": len(pos),
        "euclidean_rigid_rank": rank_e,
        "eckart_rigid_rank": rank_m,
        "euclidean_rigid_singular_values_rel": sv_e.tolist(),
        "min_eigenvalue": float(lam.min()),
        "lowest_frequencies_cm": nu[:6].tolist(),
        "highest_frequency_cm": float(nu.max()),
        "orca_unprojected_frequencies_first": data["orca_freqs"][:rank_e + 3].tolist(),
        "orca_negative_count": int(np.sum(data["orca_freqs"] < 0)),
        "orca_vs_audit_vibrational_max_abs_cm": float(np.max(np.abs(
            np.sort(data["orca_freqs"])[rank_e:] - np.sort(nu)))),
        "wilson_rows_analytic_vs_fd_max": float(np.max(np.abs(rows - rows_fd))),
        "route_K_vs_M_max_rel": float(np.max(np.abs(C_K - C_M)) / scale),
        "route_W1_vs_M_max_rel": float(np.max(np.abs(C_W1 - C_M)) / scale),
        "route_W2_vs_M_max_rel": float(np.max(np.abs(C_W2 - C_M)) / scale),
        "wilson_completion_residual": max(res1, res2),
        "GF_vs_modes_eigenvalue_max_rel": float(np.max(np.abs(gf1 - np.sort(lam)) / np.sort(lam))),
        "F01_completion1": f01_1, "F01_completion2": f01_2,
        "modes_near_200_cm": [float(x) for x in nu if 180 <= x <= 220],
        "modes_near_1000_cm": [float(x) for x in nu if 950 <= x <= 1050],
        "hessian": hessian_diagnostics(K, pos),
    }
    for t, v in alt.items():
        if isinstance(v, Exception):
            summary[f"{t}_error"] = str(v)
        else:
            summary[f"{t}_lowest_cm"] = float(np.sign(v[0].min()) * math.sqrt(abs(v[0].min())) * HARTREE_INV_CM)
    endpoint_rows = []
    for (ea, eb, ident) in endpoints:
        i, j = names.index(ea), names.index(eb)
        d = descriptors(lam, amps, i, j)
        row = {"set": label.split(":")[0], "system": label.split(":")[1], "endpoint": ident,
               "coordinate_a": ea, "coordinate_b": eb, **d,
               "C_angstrom_per_mdyn": d["chi"] * CONV_COMPLIANCE,
               "chi_route_K": float(C_K[i, j]), "chi_route_W": float(C_W1[i, j])}
        for t, v in alt.items():
            if isinstance(v, Exception) or np.any(v[0] <= 0):
                row[f"eta_{t}"] = float("nan")
                continue
            d2 = descriptors(v[0], v[1], i, j)
            row[f"eta_{t}"] = d2["eta"]
            row[f"chi_{t}"] = d2["chi"]
            row[f"kappa_{t}"] = d2["kappa"]
        for delta in deltas:
            b = degenerate_bounds(lam, amps, i, j, delta)
            for k, val in b.items():
                row[f"{k}_d{delta:g}"] = val
        endpoint_rows.append(row)
    noise_ranges(K, pos, masses, rows, names, endpoints, endpoint_rows)
    return summary, endpoint_rows, (lam, amps, L, names, rows)


NOISE_LEVELS = (1e-7, 1e-6)
NOISE_SAMPLES = 100


def noise_ranges(K, pos, masses, rows, names, endpoints, endpoint_rows):
    """Ranges of descriptors under random symmetric Hessian perturbations far
    below the size of the rigid-invariance correction and of DFT grid noise."""
    rng = np.random.default_rng(20261019)
    targets = endpoint_rows[-len(endpoints):]
    for amp in NOISE_LEVELS:
        acc = {e[2]: [] for e in endpoints}
        for _ in range(NOISE_SAMPLES):
            E = rng.normal(size=K.shape) * amp
            lam, amps, _ = route_modes(K + 0.5 * (E + E.T), pos, masses, rows)
            for ea, eb, ident in endpoints:
                d = descriptors(lam, amps, names.index(ea), names.index(eb))
                acc[ident].append((d["eta"], d["A"], d["kappa"], d["low"], d["high"]))
        for row, (_, _, ident) in zip(targets, endpoints):
            v = np.array(acc[ident])
            tag = f"noise{amp:.0e}"
            row[f"eta_min_{tag}"], row[f"eta_max_{tag}"] = float(v[:, 0].min()), float(v[:, 0].max())
            row[f"A_min_{tag}"], row[f"A_max_{tag}"] = float(v[:, 1].min()), float(v[:, 1].max())
            row[f"kappa_min_{tag}"], row[f"kappa_max_{tag}"] = float(v[:, 2].min()), float(v[:, 2].max())
            row[f"low_range_{tag}"] = float(np.ptp(v[:, 3]))
            row[f"high_range_{tag}"] = float(np.ptp(v[:, 4]))


def compare(observed: dict, reference: dict, keys: dict, tag: str, out: list):
    """Fail loudly if independently recomputed publication values disagree.

    The thresholds equal the established publication reproduction gate:
    compliance in Angstrom/mdyn uses rtol=atol=1e-8; all other
    quantities use rtol=2e-6 and atol=2e-8. No reference values are used
    to calculate the response itself.
    """
    for ok, rk in keys.items():
        o, r = float(observed[ok]), float(reference[rk])
        rtol, atol = (1e-8, 1e-8) if ok == "C_angstrom_per_mdyn" else (2e-6, 2e-8)
        if not (np.isfinite(o) and np.isfinite(r)):
            raise AssertionError(f"{tag}/{ok}: nonfinite result audit={o}, published={r}")
        out.append({"tag": tag, "quantity": ok, "audit": o, "reference": r,
                    "abs_diff": abs(o - r), "rel_diff": rel(o, r)})
        if not np.isclose(o, r, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{tag}/{ok}: independent={o:.16g} publication={r:.16g}; "
                f"absolute_difference={abs(o-r):.6g} (rtol={rtol:g}, atol={atol:g})"
            )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"python": sys.version.split()[0], "numpy": np.__version__}

    # ---- units ---------------------------------------------------------
    global CONV_COMPLIANCE
    eh_mdyn_angstrom = HARTREE_J * 1e8 * 1e10
    CONV_COMPLIANCE = BOHR_ANGSTROM**2 / eh_mdyn_angstrom
    report["units"] = {
        "a0^2/Eh_to_angstrom/mdyn_CODATA2018": CONV_COMPLIANCE,
        "repository_value": 0.0642304854990309,
        "relative_difference": rel(0.0642304854990309, CONV_COMPLIANCE),
        "Eh/a0^2_in_mdyn/angstrom": 1 / CONV_COMPLIANCE,
        "repository_BOHR_PER_ANGSTROM_vs_CODATA2018": rel(1.8897261254578281, 1 / BOHR_ANGSTROM),
        "repository_HARTREE_WAVENUMBER_vs_CODATA2018": rel(219474.6313705, HARTREE_INV_CM),
        "repository_AMU_ME_vs_CODATA2018": rel(1822.888486209, AMU_ME),
    }

    maps = coordinate_maps()
    systems_summary = []
    endpoint_rows: list[dict] = []
    comparisons: list[dict] = []
    keep: dict = {}

    # ---- primary 37 ----------------------------------------------------
    ref37 = read_csv(ROOT / "reference/compliance_37_endpoints.csv")
    if len(ref37) != 37 or len({r["endpoint_id"] for r in ref37}) != 37:
        raise AssertionError("Expected exactly 37 distinct published primary endpoints")
    for name, path in PRIMARY_HESS.items():
        data = read_hess(path)
        defs = {k: tuple(v) for k, v in maps[name]["coordinates"].items()}
        names = list(defs)
        eps = [(r["coordinate_a"], r["coordinate_b"], r["endpoint_id"]) for r in ref37 if r["system"] == name]
        s, rows, k = analyse_system(f"primary:{name}", data, names, defs, eps)
        systems_summary.append(s); endpoint_rows.extend(rows); keep[("primary", name)] = (data, k, defs)
    for r in ref37:
        o = next(x for x in endpoint_rows if x["set"] == "primary" and x["endpoint"] == r["endpoint_id"])
        compare(o, r, {"chi": "chi_ab_bohr2_per_hartree", "C_angstrom_per_mdyn": "C_AB_angstrom_per_mdyn",
                       "eta": "eta", "A": "A", "kappa": "kappa", "low": "low_lt200_fraction",
                       "mid": "mid_200_1000_fraction", "high": "high_gt1000_fraction"},
                f"primary:{r['endpoint_id']}", comparisons)

    # ---- functional controls 25 -----------------------------------------
    ref25 = read_csv(ROOT / "reference/stage3b_endpoint_comparison.csv")
    if len(ref25) != 25 or len({(r["system"],r["coordinate_a"],r["coordinate_b"]) for r in ref25}) != 25:
        raise AssertionError("Expected exactly 25 distinct functional controls")
    for name, path in FUNCTIONAL_HESS.items():
        data = read_hess(path)
        defs = {k: tuple(v) for k, v in maps[name]["coordinates"].items()}
        names = list(defs)
        eps = [(r["coordinate_a"], r["coordinate_b"], f"{name}::{r['coordinate_a']}__{r['coordinate_b']}")
               for r in ref25 if r["system"] == name]
        s, rows, k = analyse_system(f"functional:{name}", data, names, defs, eps)
        systems_summary.append(s); endpoint_rows.extend(rows); keep[("functional", name)] = (data, k, defs)
    for r in ref25:
        ident = f"{r['system']}::{r['coordinate_a']}__{r['coordinate_b']}"
        o = next(x for x in endpoint_rows if x["set"] == "functional" and x["endpoint"] == ident)
        compare(o, r, {"chi": "chi_ab_bohr2_per_hartree", "eta": "eta", "A": "absolute_modal_activity_A",
                       "kappa": "coherence_kappa", "low": "abs_activity_below_200_cm_fraction",
                       "mid": "abs_activity_200_1000_cm_fraction", "high": "abs_activity_above_1000_cm_fraction"},
                f"functional:{ident}", comparisons)

    # ---- water hexamer 6 -------------------------------------------------
    graph = json.loads((HEX_DIR / "results/preregistered_endpoint_graph.json").read_text())
    data = read_hess(HEX_HESS)
    defs, eps = {}, []
    for e in graph["endpoints"]:
        for L_ in "AB":
            defs[f"{e['endpoint_id']}_{L_}"] = tuple(int(x) - 1 for x in e[f"coord_{L_}"]["atoms"])
        eps.append((f"{e['endpoint_id']}_A", f"{e['endpoint_id']}_B", e["endpoint_id"]))
    names = list(defs)
    s, rows, k = analyse_system("hexamer:water_hexamer_book", data, names, defs, eps)
    systems_summary.append(s); endpoint_rows.extend(rows); keep[("hexamer", "hex")] = (data, k, defs)
    refhex_rows = read_csv(ROOT / "reference/water_hexamer_endpoint_metrics.csv")
    refhex = {r["endpoint_id"]: r for r in refhex_rows}
    if len(refhex_rows) != 6 or len(refhex) != 6 or [e["endpoint_id"] for e in graph["endpoints"]] != [f"E{i}" for i in range(1,7)]:
        raise AssertionError("Expected exactly six preregistered hexamer endpoints")
    for e in graph["endpoints"]:
        o = next(x for x in endpoint_rows if x["set"] == "hexamer" and x["endpoint"] == e["endpoint_id"])
        compare(o, refhex[e["endpoint_id"]], {
            "chi": "chi_ab_bohr2_per_hartree", "C_angstrom_per_mdyn": "compliance_A_per_mdyn",
            "eta": "eta", "A": "absolute_modal_activity_A", "kappa": "coherence_kappa",
            "low": "abs_activity_below_200_cm_fraction", "mid": "abs_activity_200_1000_cm_fraction",
            "high": "abs_activity_above_1000_cm_fraction", "chi_aa": "chi_aa_bohr2_per_hartree",
            "chi_bb": "chi_bb_bohr2_per_hartree"}, f"hexamer:{e['endpoint_id']}", comparisons)

    # ---- hexamer IPR, mode gaps, alternative participation ------------------
    lam, amps, Lmodes, names, rows_h = keep[("hexamer", "hex")][1]
    nu = np.sqrt(lam) * HARTREE_INV_CM
    waters = [[graph["water_assignment"][f"W{j}"]["O"] - 1] +
              [h - 1 for h in graph["water_assignment"][f"W{j}"]["H"]] for j in range(1, 7)]
    m3 = np.repeat(data["masses"] * AMU_ME, 3)
    ref_modes = read_csv(ROOT / "reference/water_hexamer_top_modes_with_ipr.csv")
    ipr_rows = []
    count_mw = count_cart = 0
    for e in graph["endpoints"]:
        eid = e["endpoint_id"]
        i, j = names.index(f"{eid}_A"), names.index(f"{eid}_B")
        chi = compliance_from_modes(lam, amps)
        w = amps[:, i] * amps[:, j] / lam / math.sqrt(chi[i, i] * chi[j, j])
        order = np.argsort(np.abs(w))[::-1]
        for rank, s_ in enumerate(order[:10], 1):
            mw = Lmodes[:, s_].reshape(-1, 3)
            cart = (Lmodes[:, s_] / np.sqrt(m3)).reshape(-1, 3)
            p_mw = np.array([np.sum(mw[g] ** 2) for g in waters]); p_mw /= p_mw.sum()
            p_c = np.array([np.sum(cart[g] ** 2) for g in waters]); p_c /= p_c.sum()
            gaps = np.abs(nu - nu[s_]); gaps[s_] = np.inf
            ref = next(r for r in ref_modes if r["endpoint_id"] == eid and int(r["rank"]) == rank)
            ipr_rows.append({
                "endpoint": eid, "rank": rank, "mode": int(s_), "freq_cm": float(nu[s_]),
                "weight": float(w[s_]), "ipr_mass_weighted": 1 / float(np.sum(p_mw ** 2)),
                "ipr_cartesian": 1 / float(np.sum(p_c ** 2)),
                "nearest_gap_cm": float(gaps.min()),
                "ref_ipr": float(ref["water_monomer_ipr"]), "ref_freq": float(ref["frequency_cm_minus1"]),
                "ref_mode": int(ref["mode_index_zero_based"]), "ref_weight": float(ref["normalized_static_modal_weight"]),
            })
            if rank <= 5:
                count_mw += ipr_rows[-1]["ipr_mass_weighted"] >= 3
                count_cart += ipr_rows[-1]["ipr_cartesian"] >= 3
    report["hexamer_ipr"] = {
        "top5_ipr_ge3_mass_weighted": int(count_mw),
        "top5_ipr_ge3_cartesian_displacement": int(count_cart),
        "max_abs_ipr_diff_vs_reference": max(abs(r["ipr_mass_weighted"] - r["ref_ipr"]) for r in ipr_rows),
        "mode_index_mismatches_vs_reference": sum(r["mode"] != r["ref_mode"] for r in ipr_rows),
        "max_abs_weight_diff_vs_reference": max(abs(r["weight"] - r["ref_weight"]) for r in ipr_rows),
        "min_nearest_gap_top5_cm": min(r["nearest_gap_cm"] for r in ipr_rows if r["rank"] <= 5),
    }
    with (OUT / "hexamer_top_modes_audit.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(ipr_rows[0])); wtr.writeheader(); wtr.writerows(ipr_rows)

    # ---- hexamer spectral integrals vs reference -----------------------------
    spec_ref = read_csv(ROOT / "reference/water_hexamer_spectral_integral_audit.csv")
    spec_rows = []
    upper = max(10000.0, 3 * nu.max())
    for r in spec_ref:
        eid, gamma = r["endpoint_id"], float(r["gamma_cm_minus1"])
        i, j = names.index(f"{eid}_A"), names.index(f"{eid}_B")
        chi = compliance_from_modes(lam, amps)
        coef = amps[:, i] * amps[:, j] / math.sqrt(chi[i, i] * chi[j, j])
        g = gamma / HARTREE_INV_CM

        def dens(wcm):
            w_ = np.atleast_1d(wcm) / HARTREE_INV_CM
            val = (4 * g / math.pi) * coef[:, None] / ((lam[:, None] - w_[None] ** 2) ** 2 + (2 * g * w_[None]) ** 2)
            return val.sum(0) / HARTREE_INV_CM
        grid = np.arange(0.0, upper + 0.125, 0.25)
        trap = float(np.trapezoid(dens(grid), grid))
        exact_inf = sum(integrate.quad(lambda x: float(dens(x)[0]), a, b, limit=500, epsabs=1e-13)[0]
                        for a, b in zip([0, 500, 1500, 3000, 4500], [500, 1500, 3000, 4500, np.inf]))
        spec_rows.append({"endpoint": eid, "gamma": gamma, "eta": float(chi[i, j] / math.sqrt(chi[i, i] * chi[j, j])),
                          "trapezoid_0p25": trap, "reference_trapezoid_0p25": float(r["integral_step_0p25_cm"]),
                          "quad_0_to_inf": exact_inf})
    report["hexamer_spectral"] = {
        "max_abs_trapezoid_vs_reference": max(abs(x["trapezoid_0p25"] - x["reference_trapezoid_0p25"]) for x in spec_rows),
        "max_abs_quad_inf_minus_eta": max(abs(x["quad_0_to_inf"] - x["eta"]) for x in spec_rows),
        "max_abs_trapezoid_minus_eta": max(abs(x["trapezoid_0p25"] - x["eta"]) for x in spec_rows),
    }

    # ---- isotope / mass perturbation tests --------------------------------------
    iso_rows = []
    tests = [
        ("primary", "methane_dimer", 6, 2.01410177812, [("CH_A_in_1", "CH_B_in_1"), ("CC", "HH_A1_B1")]),
        ("primary", "methane_dimer", 6, 1.008 * (1 + 1e-4), [("CH_A_in_1", "CH_B_in_1")]),
        ("primary", "ammonia_borane", 7, 2.01410177812, [("NH1", "BH1"), ("NB", "NH1")]),
        ("primary", "fhf_minus", 1, 2.01410177812, [("FH_left", "HF_right")]),
        ("primary", "water_dimer", 4, 2.01410177812, [("donor_OH", "hydrogen_bond_HO")]),
    ]
    for set_, name, atom, newmass, pairs in tests:
        data_s, (lam0, amps0, _, names_s, rows_s), defs_s = keep[(set_, name)]
        masses = data_s["masses"].copy(); masses[atom] = newmass
        lam1, amps1, _ = route_modes(data_s["K"], data_s["pos"], masses, rows_s)
        for a_, b_ in pairs:
            i, j = names_s.index(a_), names_s.index(b_)
            d0 = descriptors(lam0, amps0, i, j); d1 = descriptors(lam1, amps1, i, j)
            iso_rows.append({"system": name, "atom_mass_changed_zero_based": atom, "new_mass_amu": newmass,
                             "pair": f"{a_}/{b_}", "eta_ref": d0["eta"], "eta_perturbed": d1["eta"],
                             "A_ref": d0["A"], "A_perturbed": d1["A"],
                             "kappa_ref": d0["kappa"], "kappa_perturbed": d1["kappa"]})
    report["isotope_tests"] = iso_rows

    # ---- linear-geometry classification margin ---------------------------------
    report["linearity"] = {s["system"]: {"rank": s["euclidean_rigid_rank"],
                                         "smallest_rigid_singular_value_rel": min(s["euclidean_rigid_singular_values_rel"])}
                           for s in systems_summary}

    # ---- functional control gates recomputed --------------------------------------
    prim = {(r["system"], r["coordinate_a"], r["coordinate_b"]): r for r in ref37}
    func = [r for r in endpoint_rows if r["set"] == "functional"]
    deltas = []
    for r in func:
        b = next(x for x in endpoint_rows if x["set"] == "primary" and x["system"] == r["system"]
                 and x["coordinate_a"] == r["coordinate_a"] and x["coordinate_b"] == r["coordinate_b"])
        deltas.append({"system": r["system"], "pair": f"{r['coordinate_a']}/{r['coordinate_b']}",
                       "eta_pbe0": b["eta"], "eta_wb97x": r["eta"], "kappa_pbe0": b["kappa"],
                       "kappa_wb97x": r["kappa"], "noncancellation": b["kappa"] >= 0.2})
    nc = [d for d in deltas if d["noncancellation"]]
    rho = stats.spearmanr([d["eta_pbe0"] for d in nc], [d["eta_wb97x"] for d in nc]).statistic
    report["functional_gates"] = {
        "noncancellation_count": len(nc),
        "max_abs_delta_eta": max(abs(d["eta_wb97x"] - d["eta_pbe0"]) for d in nc),
        "max_abs_delta_kappa": max(abs(d["kappa_wb97x"] - d["kappa_pbe0"]) for d in nc),
        "spearman_signed_eta": float(rho),
        "sign_changes_all_25": [d for d in deltas if np.sign(d["eta_pbe0"]) != np.sign(d["eta_wb97x"])],
        "endpoints_crossing_kappa_0p2": [d for d in deltas if (d["kappa_pbe0"] >= 0.2) != (d["kappa_wb97x"] >= 0.2)],
    }

    # ---- provenance: inputs vs final geometries vs Hessian geometries ---------------
    prov = []
    def add_prov(label, inp, xyz, hess):
        si, gi, kw = read_inp_geometry(inp)
        sx, gx, title = read_xyz(xyz)
        h = read_hess(hess)
        gh = h["pos"] * BOHR_ANGSTROM
        def kabsch_rmsd(a, b):
            a0, b0 = a - a.mean(0), b - b.mean(0)
            u, _, vt = np.linalg.svd(a0.T @ b0)
            dmat = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
            return float(np.sqrt(np.mean(np.sum((a0 @ (u @ dmat @ vt) - b0) ** 2, 1))))
        prov.append({
            "label": label, "keywords": kw, "xyz_title": title,
            "symbols_inp_xyz_hess_equal": si == sx == h["symbols"],
            "xyz_vs_hess_max_abs_angstrom": float(np.max(np.abs(gx - gh))),
            "inp_vs_hess_max_abs_angstrom": float(np.max(np.abs(gi - gh))),
            "inp_vs_hess_rmsd_aligned_angstrom": kabsch_rmsd(gi, gh),
            "masses_match_standard_weights": bool(all(abs(m - STANDARD_WEIGHTS[s]) < 2e-3 for s, m in zip(h["symbols"], h["masses"]))),
            "masses": sorted({f"{s}:{m}" for s, m in zip(h["symbols"], h["masses"])}),
        })
    for name in PRIMARY_HESS:
        stage = "prototypes_stage2" if name in ("water_dimer", "methane_dimer", "fhf_minus") else "benchmark_stage3"
        add_prov(f"primary:{name}", CALC / stage / "inputs/orca" / f"{name}.inp",
                 CALC / stage / "results/structures" / f"{name}.xyz", PRIMARY_HESS[name])
    for name in FUNCTIONAL_HESS:
        add_prov(f"functional:{name}", CALC / "stage3b_method_robustness/inputs/orca" / f"{name}_wb97x_d4.inp",
                 CALC / "stage3b_method_robustness/results/structures" / f"{name}_wb97x_d4.xyz", FUNCTIONAL_HESS[name])
        stage = "prototypes_stage2" if name in ("water_dimer", "methane_dimer", "fhf_minus") else "benchmark_stage3"
        _, g_inp, _ = read_inp_geometry(CALC / "stage3b_method_robustness/inputs/orca" / f"{name}_wb97x_d4.inp")
        _, g_pbe, _ = read_xyz(CALC / stage / "results/structures" / f"{name}.xyz")
        prov[-1]["wb97x_start_equals_pbe0_final_max_abs"] = float(np.max(np.abs(g_inp - g_pbe)))
    add_prov("hexamer", HEX_DIR / "inputs/water_hexamer_book_pbe0d4.inp",
             HEX_DIR / "results/water_hexamer_book_pbe0d4.xyz", HEX_HESS)
    _, gs, _ = read_xyz(HEX_DIR / "inputs/water_hexamer_book_start.xyz")
    _, gi, _ = read_inp_geometry(HEX_DIR / "inputs/water_hexamer_book_pbe0d4.inp")
    prov[-1]["start_xyz_equals_inp_max_abs"] = float(np.max(np.abs(gs - gi)))
    report["provenance"] = prov

    # ---- comparisons summary ------------------------------------------------
    by_q: dict = {}
    for c in comparisons:
        q = (c["tag"].split(":")[0], c["quantity"])
        cur = by_q.setdefault(q, {"n": 0, "max_abs": 0.0, "max_rel": 0.0})
        cur["n"] += 1; cur["max_abs"] = max(cur["max_abs"], c["abs_diff"]); cur["max_rel"] = max(cur["max_rel"], c["rel_diff"])
    report["reference_comparison"] = {f"{k[0]}/{k[1]}": v for k, v in sorted(by_q.items())}
    report["endpoint_counts"] = {s: sum(r["set"] == s for r in endpoint_rows) for s in ("primary", "functional", "hexamer")}
    if report["endpoint_counts"] != {"primary": 37, "functional": 25, "hexamer": 6}:
        raise AssertionError(f"Incomplete independent audit: {report['endpoint_counts']}")

    # ---- sensitivity summary ----------------------------------------------
    sens = {}
    for t in ("mass_weighted_projection_of_raw", "full_space_drop_smallest"):
        vals = [r for r in endpoint_rows if not math.isnan(r.get(f"eta_{t}", float("nan")))]
        sens[t] = {
            "n_evaluable": len(vals),
            "max_abs_delta_eta": max(abs(r[f"eta_{t}"] - r["eta"]) for r in vals),
            "worst": max(vals, key=lambda r: abs(r[f"eta_{t}"] - r["eta"]))["endpoint"],
            "sign_reversals": [r["endpoint"] for r in vals if np.sign(r[f"eta_{t}"]) != np.sign(r["eta"])],
            "max_rel_delta_chi": max(rel(r[f"chi_{t}"], r["chi"]) for r in vals),
        }
    report["rigid_treatment_sensitivity"] = sens
    fragile = []
    for r in endpoint_rows:
        tag = "noise1e-06"
        dk = r[f"kappa_max_{tag}"] - r[f"kappa_min_{tag}"]
        dA = (r[f"A_max_{tag}"] - r[f"A_min_{tag}"]) / r["A"]
        if dk > 0.02 or dA > 0.05:
            fragile.append({"set": r["set"], "endpoint": r["endpoint"], "eta": r["eta"], "A": r["A"],
                            "kappa": r["kappa"], "A_range_noise1e-6": [r[f"A_min_{tag}"], r[f"A_max_{tag}"]],
                            "kappa_range_noise1e-6": [r[f"kappa_min_{tag}"], r[f"kappa_max_{tag}"]],
                            "eta_range_noise1e-6": [r[f"eta_min_{tag}"], r[f"eta_max_{tag}"]],
                            "A_bounds_degenerate_1cm": [r["A_equal_freq_proxy_lower_d1"], r["A_equal_freq_proxy_upper_d1"]]})
    report["noise_fragile_endpoints"] = fragile
    report["noise_max_abs_eta_shift"] = {
        lvl: max(max(abs(r[f"eta_max_noise{lvl}"] - r["eta"]), abs(r[f"eta_min_noise{lvl}"] - r["eta"]))
                 for r in endpoint_rows) for lvl in ("1e-07", "1e-06")}
    report["route_agreement"] = {
        "max_route_K_vs_M_rel": max(s["route_K_vs_M_max_rel"] for s in systems_summary),
        "max_route_W_vs_M_rel": max(max(s["route_W1_vs_M_max_rel"], s["route_W2_vs_M_max_rel"]) for s in systems_summary),
        "max_GF_eigen_rel": max(s["GF_vs_modes_eigenvalue_max_rel"] for s in systems_summary),
        "max_wilson_row_fd_error": max(s["wilson_rows_analytic_vs_fd_max"] for s in systems_summary),
    }
    # Cross-method checks MUST fail on disagreement; otherwise independent
    # implementation could silently print a discrepancy while CI stays green.
    route_limits = {
        "max_route_K_vs_M_rel": 1e-7,
        "max_route_W_vs_M_rel": 1e-7,
        "max_GF_eigen_rel": 1e-7,
        "max_wilson_row_fd_error": 1e-7,
    }
    for quantity, threshold in route_limits.items():
        actual = report["route_agreement"][quantity]
        if not np.isfinite(actual) or actual > threshold:
            raise AssertionError(
                f"Independent {quantity}={actual:.9g} exceeds {threshold:g}; "
                "inspect Hessian rigidity/conditioning instead of relaxing limits"
            )
    for item in prov:
        if not item["symbols_inp_xyz_hess_equal"] or not item["masses_match_standard_weights"]:
            raise AssertionError(f"{item['label']}: inconsistent atom identity/masses")
        if item["xyz_vs_hess_max_abs_angstrom"] > 1e-6:
            raise AssertionError(f"{item['label']}: final XYZ differs from ORCA Hessian geometry")
    report["independent_validation_passed"] = True

    (OUT / "independent_report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    (OUT / "system_diagnostics.json").write_text(json.dumps(systems_summary, indent=2, default=float) + "\n")
    fields = []
    for r in endpoint_rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with (OUT / "endpoint_audit.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=fields); wtr.writeheader(); wtr.writerows(endpoint_rows)
    with (OUT / "reference_comparison.csv").open("w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=list(comparisons[0])); wtr.writeheader(); wtr.writerows(comparisons)
    print(json.dumps({k: report[k] for k in ("endpoint_counts", "route_agreement", "reference_comparison",
                                             "rigid_treatment_sensitivity")}, indent=1, default=float))


CONV_COMPLIANCE = float("nan")

if __name__ == "__main__":
    main()
