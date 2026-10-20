# Stage 3B Method-Robustness Design R1

## Scientific question

Stage 3A now contains nine admissible systems and passes every frozen numerical closure gate after exclusion of the two weak-complex boundary cases. The next uncertainty is no longer numerical inversion; it is electronic-structure model dependence. Stage 3B therefore asks the narrow question required for the first manuscript:

> Are the interaction-class DIRS fingerprints and the separation of modal activity from coherence stable when the electronic-structure functional is changed, without changing the observable definition?

This stage is a robustness test, not a new benchmark survey. No additional interaction class is added.

## Frozen representative subset

Exactly five systems are selected before seeing alternative-method results:

1. `water_dimer` — neutral hydrogen bond;
2. `methane_dimer` — dispersion-dominated weak binding and cancellation control;
3. `chloride_water` — ionic hydrogen bond;
4. `fhf_minus` — symmetric shared proton;
5. `ammonia_borane` — coordinate-bond control.

This is the smallest subset spanning every qualitative response regime used in the planned paper. `hf_dimer`, `hcn_dimer`, `nh3_hf`, and `h5o2_plus` remain primary PBE0-D4 transferability data but are not repeated merely to increase calculation count.

## Alternative electronic-structure level

Use ORCA 6.1.1 with:

- `WB97X-D4`;
- `def2-TZVPPD`;
- `VeryTightSCF`;
- `DEFGRID3`;
- `NoSym`;
- full geometry optimization followed by an analytical frequency/Hessian calculation at the optimized geometry;
- charge and multiplicity identical to the accepted PBE0-D4 calculation;
- atom identity and order identical to the accepted source system.

The basis is intentionally held fixed. This isolates functional dependence rather than conflating functional and basis changes. ORCA 6.1 documentation lists `WB97X-D4` as an available range-separated hybrid and permits frequency calculations for range-separated hybrids. No double-hybrid or RI-JK Hessian is authorized.

Reference documentation used when freezing this design:

- https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html
- https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html

## Starting geometries

For each system, start from the already accepted PBE0-D4 optimized geometry. The alternative-method optimization is unconstrained. Do not transfer the PBE0 Hessian or force the alternative method to remain in the original basin.

If the optimization exits into a chemically different connectivity or dissociated basin, preserve the result and classify that system as method-sensitive/unresolved; do not repair it by constraints or manual repositioning.

## Observable definitions

All DIRS definitions are frozen from Stage 3A:

- same atom-index local coordinates for the same named system;
- same rigid-body invariance correction;
- same mass weighting;
- same direct and modal static response definitions;
- same normalized `eta`;
- same absolute modal activity `A`;
- same coherence `kappa = |eta|/A`;
- same spectral normalization and damping/grid conventions;
- same isotope interpretation.

No local coordinate may be replaced after observing the WB97X-D4 geometry unless atom connectivity itself changes, in which case the system is reported unresolved rather than remapped.

## Predeclared numerical gates

Each alternative-method result must satisfy all of the following before scientific comparison:

1. normal ORCA termination and converged optimization;
2. positive-definite rigid-projected vibrational Hessian;
3. direct/modal relative agreement <= `1e-8`;
4. reciprocity absolute error <= `1e-10`;
5. `|eta| <= 1 + 1e-10`;
6. no raw-versus-rigid-corrected sign reversal for a non-cancellation manuscript endpoint;
7. the existing Stage 3A spectral sum and grid-refinement tolerances if spectra are regenerated.

A failing system is retained as method-sensitive evidence. No threshold is relaxed after calculation.

## Frozen manuscript endpoints and qualitative gates

### Water dimer

- `donor_OH` vs `hydrogen_bond_HO`: `eta < 0`;
- `hydrogen_bond_HO` vs `donor_acceptor_OO`: `eta > 0`, `kappa > 0.8`.

### Methane dimer

The tiny signed `CH_A_in_1`/`CH_B_in_1` eta is deliberately **not** assigned a sign gate because Stage 3A shows near-perfect cancellation. The invariant claim is instead:

- `CH_A_in_1` vs `CH_B_in_1`: `A > 0.5` and `kappa < 0.1`;
- `CC` vs `HH_A1_B1`: `eta > 0.1` and `kappa > 0.8`.

This tests the central distinction between abundant modal activity and coherent signed response.

### Chloride-water

- `donor_OH` vs `hydrogen_bond_HCl`: `eta < 0`;
- `hydrogen_bond_HCl` vs `donor_acceptor_OCl`: `eta > 0`, `kappa > 0.8`.

### FHF-

- `FH_left` vs `HF_right`: `eta < 0`;
- each proton-bond coordinate vs `FF`: `eta > 0`, `kappa > 0.8`.

### Ammonia-borane

- each `NB` vs `NH_i` and `NB` vs `BH_i`: `eta < 0`;
- the pre-existing `NH_i` vs `BH_j` cancellation controls: `kappa < 0.2`.

## Quantitative cross-method gates

For all designated non-cancellation endpoints (`kappa >= 0.2` at PBE0-D4):

- maximum absolute `eta` change between PBE0-D4 and WB97X-D4 must be <= `0.15` for a clean quantitative-robustness pass;
- maximum absolute `kappa` change must be <= `0.20`;
- Spearman correlation of the signed `eta` values across these endpoints must be >= `0.80`.

The cancellation endpoints are evaluated by their preregistered `A`/`kappa` regime rather than relative error in an eta that is intentionally near zero.

These quantitative thresholds govern whether the manuscript may state that the fingerprints are quantitatively robust. If the qualitative gates pass but one quantitative gate fails, the manuscript must restrict the claim to qualitative/topological robustness and report the method dependence explicitly.

## No-rescue rule

Stage 3B does not authorize:

- changing the alternative functional after seeing an unfavorable result;
- changing the basis or grid to rescue a system;
- frequency clipping;
- constrained reoptimization;
- deleting a method-sensitive representative;
- redefining local coordinates;
- adding extra representatives to dilute a failed comparison.

## Completion decision

If all five representatives pass numerical admissibility and all qualitative gates, the Stage 3A interaction-class interpretation is considered functionally transferable for first-paper purposes. Quantitative wording is governed separately by the three cross-method quantitative gates.

After Stage 3B, the remaining work should be manuscript-data synthesis (figures, tables, source-data manifests, provenance/readiness audit), not expansion of the quantum-chemistry benchmark, unless Stage 3B falsifies a core qualitative claim.
