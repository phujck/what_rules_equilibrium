import argparse
import os

import numpy as np

from suite_common import ensure_dir, write_json


def read_csv(path):
    return np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")


def slope_fit_log(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log(np.maximum(x, 1e-12))
    ly = np.log(np.maximum(y, 1e-15))
    a = np.vstack([lx, np.ones_like(lx)]).T
    coef, _, _, _ = np.linalg.lstsq(a, ly, rcond=None)
    return float(coef[0]), float(coef[1])


def validate(data_dir):
    ma = read_csv(os.path.join(data_dir, "cg_wre_ma_1.csv"))
    nu = read_csv(os.path.join(data_dir, "ncng_wre_nu_1.csv"))
    qc = read_csv(os.path.join(data_dir, "ncg_wre_qc_1.csv"))
    ncng = read_csv(os.path.join(data_dir, "ncng_wre_ncng_1.csv"))

    metrics = {
        "trace_distance": float(np.max(qc["trace_distance"])),
        "fro_error": float(np.max(np.abs(nu["residual_gap"]))),
        "hmf_basis_residual": float(np.max(np.r_[nu["residual_poly"], nu["residual_comm"], ncng["residual"]])),
        "collapse_max_deviation": float(np.max(np.abs(ma["saturation_ratio"] - np.mean(ma["saturation_ratio"])))),
        "slope_fit": {},
        "improvement_ratio": 0.0,
        "pass_fail": True,
    }

    # basis saturation slope at one representative (largest beta,coupling)
    b0 = np.max(ncng["beta"])
    g0 = np.max(ncng["coupling"])
    sub = ncng[(ncng["beta"] == b0) & (ncng["coupling"] == g0)]
    s, c = slope_fit_log(sub["basis_size"], sub["residual"])
    metrics["slope_fit"] = {"basis_saturation_log_slope": {"slope": s, "intercept": c}}

    # Non-uniqueness improvement indicator.
    better_poly = np.mean((nu["residual_poly"] < nu["residual_comm"]).astype(float))
    better_comm = np.mean((nu["residual_comm"] < nu["residual_poly"]).astype(float))
    metrics["improvement_ratio"] = float((better_poly + 1e-12) / (better_comm + 1e-12))

    checks = []
    checks.append(metrics["trace_distance"] < 1e-9)
    checks.append(np.all(ma["master_dim"] <= ma["bound_d2"]))
    checks.append(better_poly > 0.05 and better_comm > 0.05)
    checks.append(s < -0.15)

    metrics["pass_fail"] = bool(all(checks))
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description="Validate what_rules claims")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    ensure_dir(out_dir)

    metrics = validate(data_dir)
    write_json(os.path.join(out_dir, "claim_metrics_what_rules.json"), metrics)
    print("what_rules validation:", "PASS" if metrics["pass_fail"] else "FAIL")
    print(metrics)


if __name__ == "__main__":
    main()
