# WRE-MA-1

## Model set
- Qubit: `H_S=0.6 sigma_z`, `f=sigma_x`.
- Qutrit (non-commuting): `H_S=diag(0,0.9,1.8)`, `f=lambda_1+0.45 lambda_6+0.2 lambda_4`.
- Qutrit (commuting): `H_S=diag(0,0.7,1.6)`, `f=diag(0,1,2)`.

## Computation
- Build adjoint chain from repeated commutators with `H_S`.
- Build associative closure basis from products of chain elements.

## What is plotted
- `adjoint_dim`, `closure_order`, and `master_dim` against `d^2` bound.
