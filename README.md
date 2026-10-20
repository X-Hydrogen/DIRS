# DIRS

**Dynamic Interaction Response Spectra (DIRS)** is a Python implementation of
harmonic response analysis for selected molecular coordinates. It calculates
off-diagonal compliance coupling constants, signed normal-mode contributions
to mechanical response, and the associated response descriptors.

This repository accompanies the manuscript *Dynamic Interaction Response
Spectra Reveal the Modal Organization of Intermolecular Interactions*
(submitted to **The Journal of Physical Chemistry Letters**). It contains the
analysis software, the molecular Hessians, and the local-coordinate
definitions needed to reproduce the reported results. The electronic-structure
calculations were performed with ORCA 6.1.1; ORCA is not required for
postprocessing the archived Hessians.

## Installation

Use Python 3.11 or newer. For the exact numerical environment used in the
publication checks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproduction.txt
python -m pip install --no-deps -e .
```

Alternatively, install the package and test dependencies with
`python -m pip install -e ".[test]"`.

## Reproducing the calculations

```bash
python scripts/reproduce_publication.py --verify-raw
python scripts/independent_check.py
python scripts/hcn_rank_sensitivity.py
python -m pytest -q
```

The first command evaluates the reported quantities from the archived
Hessians and compares them with the reference tables:

| Dataset | Coordinate pairs | Systems |
|---|---:|---:|
| Primary PBE0-D4 | 37 | 9 |
| Water-hexamer network | 6 | 1 |
| ωB97X-D4 functional comparison | 25 | 5 |

The resulting tables and a machine-readable verification report are written
to `reproduced/`. The reference tables are used to identify the published
coordinate pairs and to check the newly calculated results; they are not
used as computed response values. The `--verify-raw` option also compares
the archived numerical Hessian files with the original ORCA `.hess` records.

`scripts/independent_check.py` parses the ORCA text Hessians without
importing the DIRS package or the original analysis scripts. It evaluates
compliance using mass-weighted normal modes, Cartesian constrained
relaxation, and two completions of the Wilson internal-coordinate
formulation. The checks reject incomplete datasets, numerical disagreement,
and inconsistent source geometries.

No new density-functional calculations are carried out by these commands.
They typically require only a standard desktop or laptop computer.

## Response quantities

Let `K` be the Cartesian Hessian, `M` the diagonal mass matrix, and `Z`
an orthonormal basis for the mass-weighted vibrational subspace. For an
interatomic-distance coordinate `A` with Cartesian gradient `b_A`, define

```text
H      = Z.T @ M^(-1/2) @ K @ M^(-1/2) @ Z
g_A    = Z.T @ M^(-1/2) @ b_A

chi_AB(0) = g_A.T @ H^(-1) @ g_B
          = sum_s (c_As * c_Bs) / omega_s^2
          = C_AB
```

`C_AB` is the cross-coordinate compliance in the adopted distance
convention. The signed modal weights `w_s` and descriptors are

```text
w_s   = c_As * c_Bs / omega_s^2
N     = sqrt(chi_AA(0) * chi_BB(0))
eta   = chi_AB(0) / N
A     = sum_s abs(w_s) / N
kappa = abs(eta) / A
```

For distance–distance coordinates, the conversion is
`1 a0²/Eh = 0.0642304854990309 Å/mdyn`. The signed response is not an
interaction energy or a direct measure of bond strength.

The treatment of rigid modes, near-degenerate eigenvectors, and frequency
partitions is described in [the reproducibility notes](docs/REPRODUCIBILITY.md).
In particular, the absolute modal activity can be sensitive to rotations
within degenerate mode subspaces even when the static compliance is
unchanged.

## Repository contents

- `src/dirs/` — response analysis, coordinate gradients, Hessian readers,
  rigid-mode projection and model systems
- `analysis/` — computational scripts used for the reported data
- `scripts/` — publication reproduction and independent numerical checks
- `calculations/` — ORCA input files, optimized geometries, Cartesian
  Hessians, compact Hessian records and coordinate specifications
- `reference/` — published comparison values
- `tests/` — unit tests, numerical invariance tests and independent
  compliance comparisons
- `docs/REPRODUCIBILITY.md` — methods, data provenance and known limitations
- `docs/VALIDATION_REPORT.md` — results of the numerical verification

The deposited Hessians correspond to optimized molecular structures.
Some ORCA `.inp` files contain the **starting geometry** used for
optimization; the `results/structures/` XYZ files and `results/hessians/`
records contain the final geometry used for DIRS analysis.

The repository does not include complete ORCA optimization logs, so
the software reproduces the *Hessian postprocessing* rather than a fresh
DFT geometry optimization.

## License and citation

The Python software is available under the MIT License. When citing a
released version, use the matching GitHub release and `CITATION.cff`
together with the associated research article.
