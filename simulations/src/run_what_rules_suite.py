import argparse
import csv
import os
import time

import numpy as np

from suite_common import (
    SuiteProfile,
    ensure_dir,
    exact_reduced_density_bosonic,
    exact_static_mixture,
    fit_hmf_in_basis,
    gauss_hermite_expectation_matrix,
    hermitize,
    hmf_from_rho,
    qutrit_models,
    sigma_x,
    sigma_z,
    static_spin_distribution,
    trace_distance,
    write_json,
)


PROFILES = {
    "quick": SuiteProfile(
        name="quick",
        cutoff_list=[4, 6],
        beta_list=[0.8, 1.2],
        coupling_list=[0.08, 0.14],
        mc_samples=20000,
        quad_order=24,
        edgeworth_order=4,
    ),
    "full": SuiteProfile(
        name="full",
        cutoff_list=[4, 6, 8, 10],
        beta_list=[0.7, 1.0, 1.3, 1.7],
        coupling_list=[0.05, 0.08, 0.12, 0.16, 0.2],
        mc_samples=60000,
        quad_order=40,
        edgeworth_order=4,
    ),
    "publish": SuiteProfile(
        name="publish",
        cutoff_list=[4, 6, 8, 10, 12],
        beta_list=[0.6, 0.8, 1.0, 1.2, 1.6, 2.0],
        coupling_list=[0.04, 0.06, 0.09, 0.12, 0.16, 0.2, 0.24],
        mc_samples=120000,
        quad_order=64,
        edgeworth_order=4,
    ),
}

REGIMES = ("cg", "cng", "ncg", "ncng")


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mat_residual(op, basis):
    _, resid = fit_hmf_in_basis(op, basis)
    return resid


def independent_append(basis, op, tol=1e-10):
    if not basis:
        basis.append(op)
        return True
    resid = mat_residual(op, basis)
    if resid > tol:
        basis.append(op)
        return True
    return False


def commutator(a, b):
    return a @ b - b @ a


def adjoint_chain_independent(hs, f, max_depth=16, tol=1e-10):
    chain = []
    cur = f.copy()
    independent_append(chain, hermitize(cur), tol=tol)
    closure_order = None
    for d in range(1, max_depth + 1):
        cur = commutator(hs, cur)
        candidate = hermitize(1j * cur)
        if not independent_append(chain, candidate, tol=tol):
            closure_order = d
            break
    if closure_order is None:
        closure_order = max_depth
    return chain, closure_order


def master_algebra_basis(hs, f, max_depth=8, tol=1e-10):
    d = hs.shape[0]
    chain, closure_order = adjoint_chain_independent(hs, f, max_depth=max_depth, tol=tol)
    gens = [hermitize(g) for g in chain]

    basis = [np.eye(d, dtype=np.complex128)]
    for g in gens:
        independent_append(basis, g, tol=tol)

    changed = True
    while changed and len(basis) < d * d:
        changed = False
        current = list(basis)
        for b in current:
            for g in gens:
                for cand in (hermitize(b @ g), hermitize(g @ b), hermitize(b @ g @ b)):
                    if independent_append(basis, cand, tol=tol):
                        changed = True
                        if len(basis) >= d * d:
                            break
                if len(basis) >= d * d:
                    break
            if len(basis) >= d * d:
                break
    return basis, len(gens), closure_order


def run_ma(profile, data_dir):
    models = qutrit_models()

    systems = {
        "qubit_noncomm": (
            0.6 * sigma_z(),
            sigma_x(),
        ),
        "qutrit_noncomm": (
            models["hs_noncomm"],
            models["f_noncomm"],
        ),
        "qutrit_comm": (
            models["hs_comm"],
            models["f_comm"],
        ),
    }

    rows = []
    for name, (hs, f) in systems.items():
        basis, adj_dim, closure_order = master_algebra_basis(hs, f, max_depth=10)
        d = hs.shape[0]
        rows.append(
            {
                "test_id": "WRE-MA-1",
                "system": name,
                "d": d,
                "adjoint_dim": adj_dim,
                "closure_order": closure_order,
                "master_dim": len(basis),
                "bound_d2": d * d,
                "saturation_ratio": float(len(basis) / float(d * d)),
            }
        )

    write_csv(os.path.join(data_dir, "cg_wre_ma_1.csv"), rows)


def run_nu(profile, data_dir):
    models = qutrit_models()
    hs = models["hs_noncomm"]
    f = models["f_noncomm"]
    f2 = hermitize(f @ f)
    f4 = hermitize(f2 @ f2)
    c1 = hermitize(1j * commutator(hs, f))

    poly_basis = [np.eye(3, dtype=np.complex128), f2, f4]
    comm_basis = [np.eye(3, dtype=np.complex128), f, c1]

    template = np.array([0.08, 0.08, 0.16, 0.16], dtype=float)
    rows = []
    for beta in profile.beta_list:
        for g in profile.coupling_list:
            values, probs = static_spin_distribution(g * template)
            rho = exact_static_mixture(hs, f, beta, values, probs)
            hmf = hmf_from_rho(rho, beta)

            r_poly = mat_residual(hmf, poly_basis)
            r_comm = mat_residual(hmf, comm_basis)
            dominance = "poly" if r_poly < r_comm else "comm"

            rows.append(
                {
                    "test_id": "WRE-NU-1",
                    "beta": beta,
                    "coupling": g,
                    "residual_poly": r_poly,
                    "residual_comm": r_comm,
                    "residual_gap": r_poly - r_comm,
                    "dominance": dominance,
                }
            )

    write_csv(os.path.join(data_dir, "ncng_wre_nu_1.csv"), rows)


def run_qc(profile, data_dir):
    hs_q = 0.6 * sigma_z()
    f_q = sigma_x()

    rows = []
    for beta in profile.beta_list:
        for g in profile.coupling_list:
            rho_q = exact_reduced_density_bosonic(
                hs_q,
                f_q,
                bath_omegas=np.array([1.05], dtype=float),
                bath_cs=np.array([g], dtype=float),
                cutoff=max(profile.cutoff_list),
                beta=beta,
            )

            evals, u = np.linalg.eigh(hermitize(rho_q))
            evals = np.clip(np.real_if_close(evals), 1e-14, None)
            evals = evals / np.sum(evals)

            h_diag = -(1.0 / beta) * np.log(evals)
            c0 = float(h_diag[0])
            c1 = float(h_diag[1] - h_diag[0])

            # Projector coupling f_cl^n = f_cl, so one coefficient is enough.
            f_cl = np.diag([0.0, 1.0]).astype(np.complex128)
            rho_diag = np.diag(np.exp(-beta * h_diag))
            rho_diag = rho_diag / np.trace(rho_diag)
            rho_cl = hermitize(u @ rho_diag @ u.conj().T)

            td = trace_distance(rho_q, rho_cl)

            # One possible "cost" mapping if encoded as a fourth-cumulant correction.
            kappa4_required = -24.0 * beta * c1
            feasible = 1 if abs(kappa4_required) < 25.0 else 0

            rows.append(
                {
                    "test_id": "WRE-QC-1",
                    "beta": beta,
                    "coupling": g,
                    "trace_distance": td,
                    "kappa4_cost": kappa4_required,
                    "constraint_feasible": feasible,
                    "c0": c0,
                    "c1": c1,
                    "p0_exact": float(np.real_if_close(rho_q[0, 0])),
                    "p1_exact": float(np.real_if_close(rho_q[1, 1])),
                    "p0_pred": float(np.real_if_close(rho_cl[0, 0])),
                    "p1_pred": float(np.real_if_close(rho_cl[1, 1])),
                }
            )

    write_csv(os.path.join(data_dir, "ncg_wre_qc_1.csv"), rows)


def run_ncng(profile, data_dir):
    models = qutrit_models()
    hs = models["hs_noncomm"]
    f = models["f_noncomm"]

    chain, _ = adjoint_chain_independent(hs, f, max_depth=6)
    candidates = [np.eye(3, dtype=np.complex128)] + [hermitize(x) for x in chain]
    for a in chain:
        for b in chain:
            candidates.append(hermitize(a @ b))

    # Deduplicate by independence.
    basis = []
    for c in candidates:
        independent_append(basis, c, tol=1e-10)

    template = np.array([0.08, 0.08, 0.16, 0.16], dtype=float)
    rows = []
    for beta in profile.beta_list:
        for g in profile.coupling_list:
            values, probs = static_spin_distribution(g * template)
            rho = exact_static_mixture(hs, f, beta, values, probs)
            hmf = hmf_from_rho(rho, beta)

            for k in range(1, len(basis) + 1):
                resid = mat_residual(hmf, basis[:k])
                rows.append(
                    {
                        "test_id": "WRE-NCNG-1",
                        "beta": beta,
                        "coupling": g,
                        "basis_size": k,
                        "residual": resid,
                    }
                )

    write_csv(os.path.join(data_dir, "ncng_wre_ncng_1.csv"), rows)


def write_manifest(out_dir, args, profile_name, duration):
    payload = {
        "suite": "what_rules_equilibrium",
        "profile": profile_name,
        "regime": args.regime,
        "seed": args.seed,
        "n_workers": args.n_workers,
        "duration_sec": duration,
    }
    write_json(os.path.join(out_dir, "manifest.json"), payload)


def parse_args():
    p = argparse.ArgumentParser(description="Run what_rules suite")
    p.add_argument("--regime", choices=["cg", "cng", "ncg", "ncng", "all"], default="all")
    p.add_argument("--profile", choices=list(PROFILES.keys()), default="quick")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--n-workers", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)
    ensure_dir(data_dir)

    profile = PROFILES[args.profile]

    t0 = time.time()
    regimes = REGIMES if args.regime == "all" else (args.regime,)

    if "cg" in regimes:
        run_ma(profile, data_dir)
    if "cng" in regimes:
        run_nu(profile, data_dir)
    if "ncg" in regimes:
        run_qc(profile, data_dir)
    if "ncng" in regimes:
        run_ncng(profile, data_dir)

    elapsed = time.time() - t0
    write_manifest(out_dir, args, profile.name, elapsed)
    print(f"what_rules suite complete in {elapsed:.2f} s")


if __name__ == "__main__":
    main()
