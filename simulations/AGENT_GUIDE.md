# What Rules Equilibrium Simulation Agent Guide

This suite validates structural claims with explicit model definitions and exact references.

## Exact references used

1. ED trace-out reference
`rho_S^ED(beta) = Tr_B[e^{-beta H_tot}] / Tr[e^{-beta H_tot}]`.

2. Exact static-mixture reference
`rho_S^ED(beta) = sum_B p(B) exp[-beta(H_S + B f)] / Tr[...]` for static traced fields.

## Test models and plotted quantities

### WRE-MA-1 (closure)
- Systems: qubit non-commuting, qutrit non-commuting, qutrit commuting.
- Plot: computed closure order and `dim(A)` versus `d^2` bound.

### WRE-NU-1 (truncation inequivalence)
- Non-commuting non-Gaussian qutrit benchmark from exact static-mixture trace-out.
- Plot: residuals of polynomial-basis and commutator-basis truncations over `(beta,coupling)`.

### WRE-QC-1 (quantum-classical matching)
- Quantum target: non-commuting Gaussian qubit from bosonic ED trace-out.
- Classical construction: commuting diagonal construction in target eigenbasis.
- Plots: ED vs constructed `p0(coupling)` overlay and residual trace distance.

### WRE-NCNG-1 (basis saturation)
- Non-commuting non-Gaussian qutrit exact reference.
- Plot: residual vs basis size as candidate master-basis elements are added.

## Commands

```powershell
py -3 simulations/src/run_what_rules_suite.py --regime all --profile full --seed 42
py -3 simulations/src/plot_what_rules_suite.py
py -3 simulations/src/validate_what_rules_claims.py
```

## Output paths

- Data: `simulations/results/data/*.csv`
- Figures: `simulations/results/figures/wre_*.pdf` and `.png`
- QC compatibility figure for manuscript: `simulations/results/figures/qc_equivalence_demo.*`
- Metrics: `simulations/results/claim_metrics_what_rules.json`
- Manifest: `simulations/results/manifest.json`

## Failure modes

- `WRE-QC-1` mismatch: check eigenbasis normalization and positivity clipping.
- `WRE-NU-1` one-sided dominance: expand `(beta,coupling)` grid.
- `WRE-NCNG-1` weak saturation: increase candidate basis depth.
