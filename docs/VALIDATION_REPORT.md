# Numerical validation

## Reproduction environment

The publication tests were run on GitHub Actions (Ubuntu, Python 3.11)
using the dependency versions pinned in
`requirements-reproduction.txt`. The verification suite consists of

```bash
python -m pytest -q
python scripts/reproduce_publication.py --verify-raw
python scripts/independent_check.py
python scripts/hcn_rank_sensitivity.py
```

The reproducibility workflow runs the numerical checks in a clean
Python environment. Results for the current repository are available
under [GitHub Actions](https://github.com/X-Hydrogen/DIRS/actions/workflows/reproduce-paper.yml).
The source archive's verified baseline had **70 passed and 1 expected failure**. The expected failure documents
the exact-linearity assumption in the HCN dimer test; it does not arise
from an unsuccessful reproduction of a reported endpoint.

## Published data

| Calculation | Scope | Verification |
|---|---|---|
| Primary data | 37 coordinate pairs, 9 molecular systems | Passed |
| Water-hexamer extension | 6 coordinate pairs, 48 projected vibrations | Passed |
| Alternative functional | 25 coordinate pairs, 5 molecular systems | Passed |
| Source Hessians | 15 ORCA `.hess` / compact `.npz` pairs | Matched |

The primary-data maximum absolute difference against the frozen numerical
reference is `1.1540350897121243e-9`; for the six water-hexamer endpoints
it is `2.1662671656486054e-12`. The comparisons are made after
recalculating the response from the archived Hessians.

## Independent compliance calculations

The independent routine starts from ORCA `.hess` text rather than the
original analysis implementation. For every selected coordinate pair,
it compares a normal-mode expression, a Cartesian constrained-relaxation
solution and the Wilson internal-coordinate compliance matrix.

| Comparison | Maximum relative discrepancy |
|---|---:|
| Cartesian constrained relaxation vs modal response | `1.4492654780941292e-12` |
| Wilson compliance vs modal response | `2.605611687766527e-12` |
| Wilson GF vs vibrational eigenvalues | `8.743122938892058e-11` |

The largest absolute difference between analytic and finite-difference
distance-coordinate gradients is `7.635103660419418e-11`.
Numerical disagreement beyond the program's published tolerances stops the
verification run. Separate negative-control tests confirm that this
comparison rejects perturbed and nonfinite reference values.

## Sensitivity to numerical conventions

The HCN dimer is very nearly linear, with a small but finite departure
from exact collinearity. Its archived Hessian is treated as nonlinear
(six rigid modes). A diagnostic calculation with five rigid modes retains
one additional bending mode; among the three published HCN coordinate
pairs, the maximum relative change in compliance is
`2.4594323379355935e-9` and the maximum absolute change in
`kappa` is `6.369432203889858e-9`. The published calculations retain
the documented six-mode convention.

Alternative rigid-projection schemes need not give identical results:
in the 68-endpoint analysis, the largest absolute change in `eta` was
`0.006935943898264629` and the largest relative change in `chi`
was `0.029970812374513336`. No sign reversal of the signed `eta`
was observed under these checks. These values characterize sensitivity
to the analysis convention, not failure of a specified reproduction.

The detailed limits concerning near-degenerate modal contributions,
frequency-bin assignments and the absence of full ORCA optimization
logs are described in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Scope

The checks establish reproducibility of harmonic-response postprocessing
from the deposited molecular Hessians. They do not constitute independent
reruns of the electronic-structure calculations or prove that the harmonic
model is applicable outside the reported systems.
