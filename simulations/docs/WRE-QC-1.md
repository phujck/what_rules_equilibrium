# WRE-QC-1

## Model
- Quantum target: qubit with bosonic ED trace-out,
  `H_tot = H_S + H_B + f \otimes X_B`,
  `H_S=0.6 sigma_z`, `f=sigma_x`.
- Classical construction: commuting diagonal model in the eigenbasis of
  `rho_S^target` to match equilibrium state.

## Exact quantity
`rho_S^target = Tr_B[e^{-beta H_tot}] / Tr[e^{-beta H_tot}]`.

## Constructed quantity
`rho_S^cl` obtained from matched diagonal energies and transformed back to the
physical basis.

## What is plotted
- Overlay of `p0` from `rho_S^target` and `rho_S^cl`.
- Residual trace distance and required effective `kappa4` cost.
