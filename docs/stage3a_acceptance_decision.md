# Stage 3A Scientific Acceptance Decision R2

## Decision

`fluorobenzene_fluoroform__n3_pmp` is removed from the **primary static DIRS transferability set**. It remains permanently retained as a traceable boundary/failure case. No further geometry search, larger displacement, alternative functional, constrained rescue, frequency clipping, or post-hoc regularization is authorized for the first-paper Stage 3A benchmark.

The primary transferability analysis therefore proceeds with **nine accepted systems**:

1. `water_dimer`
2. `methane_dimer`
3. `fhf_minus`
4. `hf_dimer`
5. `hcn_dimer`
6. `nh3_hf`
7. `chloride_water`
8. `h5o2_plus`
9. `ammonia_borane`

Two predeclared weak-complex stress tests are retained but excluded from primary static inversion: `benzene_methane` and `fluorobenzene_fluoroform`.

## New evidence forcing this revision

The prior R2 analytical Hessian for fluorobenzene···fluoroform gave a very soft but positive rigid-projected minimum frequency of approximately `+10.925 cm^-1`. Because the response was unusually sensitive to rigid-invariance correction, Stage 3A-V1 preregistered two independent fixed-geometry ORCA 6.1.1 numerical Hessians at the **same accepted R2 geometry and same electronic-structure level**, differing only in central-difference displacement:

- `DX = 0.005 bohr`; Hessian SHA-256 `a699f2bb92a8f9c0fc5d9202594e6b32e7d57d0934b0b7a54afecabd13cc6ccb`;
- `DX = 0.010 bohr`; Hessian SHA-256 `98d8e788e8d85d6b626fdb34dae4b60aecf450d55bb598683b0fd68b1f8e943e`.

Both jobs terminated normally and their Hessians passed checksum verification. Independent three-Hessian analysis gave corrected-Hessian relative Frobenius differences of approximately `9.16e-4` (analytic vs DX005), `8.62e-4` (analytic vs DX010), and `4.79e-4` (DX005 vs DX010). However, the lowest rigid-projected frequencies were:

- analytical: `+10.9248 cm^-1`;
- numerical DX005: `-9.1354 cm^-1`;
- numerical DX010: `-4.7254 cm^-1`.

Thus both independent numerical Hessians are indefinite in the very mode that controls the prior minimum classification. The preregistered gates `all_three_positive_definite`, `closure_reciprocity_eta_bound`, `primary_eta_sign_agreement`, and `primary_eta_max_abs_difference_le_0p05` therefore fail by construction because static inversion is not defined for the two numerical Hessians.

## Scientific interpretation

The three Hessians are globally close, but the sign of the softest curvature is not numerically robust. This is precisely the regime in which a normalized static susceptibility can become dominated by a near-zero eigenvalue and therefore cease to be a defensible benchmark observable. Treating the analytical positive curvature as authoritative after two independent central-difference Hessians cross zero would be a post-hoc relaxation of the acceptance criterion.

The appropriate action is exclusion, not rescue. The system remains scientifically useful as a **method-boundary example** showing that DIRS requires a well-conditioned positive reduced Hessian and that apparently tiny Hessian discrepancies can become qualitative near a zero-curvature intermolecular mode.

## Consequence for Stage 3A

The nine-system accepted set still spans the mechanistic contrasts needed for the first paper: dispersion-dominated weak binding (`methane_dimer`), conventional and unconventional hydrogen bonding (`water_dimer`, `hf_dimer`, `hcn_dimer`, `nh3_hf`), ionic hydrogen bonding (`chloride_water`), symmetric proton sharing (`fhf_minus`, `h5o2_plus`), and a coordinate-bond control (`ammonia_borane`).

The aromatic blue-shifting example is no longer used to support a general transferability claim. Any discussion of fluorobenzene···fluoroform must be explicitly framed as a numerical/admissibility boundary case rather than primary response data.

## Next frozen action

Recompute the complete preregistered Stage 3A response/audit package on the nine-system set only. No new quantum-chemistry calculation is required for this exclusion reanalysis. Numerical acceptance remains governed by the existing Stage 3A R2 tolerances; the nine-system rerun may not loosen them. Raw-versus-rigid-corrected sign reversals remain explicit curator-review flags and are not silently thresholded away.
