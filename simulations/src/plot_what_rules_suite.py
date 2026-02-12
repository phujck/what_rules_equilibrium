import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from suite_common import ensure_dir


def read_csv(path):
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if arr.shape == ():
        arr = np.array([arr], dtype=arr.dtype)
    return arr


def plot_ma(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "cg_wre_ma_1.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    x = np.arange(len(arr["system"]))

    axes[0].bar(x - 0.15, arr["adjoint_dim"], width=0.3, label="adjoint dim")
    axes[0].bar(x + 0.15, arr["master_dim"], width=0.3, label="master dim")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(arr["system"], rotation=20)
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("WRE-MA-1: closure dimensions")
    axes[0].legend()

    axes[1].bar(x, arr["saturation_ratio"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(arr["system"], rotation=20)
    axes[1].set_ylabel(r"$\dim(\mathcal{A})/d^2$")
    axes[1].set_title("WRE-MA-1: algebra saturation")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "wre_ma_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "wre_ma_1.png"), dpi=150)
    plt.close(fig)


def plot_nu(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "ncng_wre_nu_1.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    sc = axes[0].scatter(arr["beta"], arr["coupling"], c=arr["residual_gap"], cmap="coolwarm")
    axes[0].axhline(0, color="k", linewidth=0.5)
    axes[0].set_xlabel(r"$\beta$")
    axes[0].set_ylabel("coupling")
    axes[0].set_title("WRE-NU-1: residual gap (poly-comm)")
    fig.colorbar(sc, ax=axes[0], label="gap")

    axes[1].scatter(arr["residual_poly"], arr["residual_comm"], c=arr["beta"], cmap="viridis")
    mn = min(np.min(arr["residual_poly"]), np.min(arr["residual_comm"]))
    mx = max(np.max(arr["residual_poly"]), np.max(arr["residual_comm"]))
    axes[1].plot([mn, mx], [mn, mx], "k--")
    axes[1].set_xlabel("poly residual")
    axes[1].set_ylabel("comm residual")
    axes[1].set_title("WRE-NU-1: truncation inequivalence")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "wre_nu_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "wre_nu_1.png"), dpi=150)
    plt.close(fig)


def plot_qc(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "ncg_wre_qc_1.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    for beta in np.unique(arr["beta"]):
        sub = arr[arr["beta"] == beta]
        order = np.argsort(sub["coupling"])
        sub = sub[order]
        axes[0].plot(sub["coupling"], sub["p0_pred"], linestyle="-", label=fr"pred, $\beta={beta}$")
        axes[0].plot(sub["coupling"], sub["p0_exact"], marker="o", linestyle="None", label=fr"ED, $\beta={beta}$")

    axes[0].set_xlabel("Quantum coupling")
    axes[0].set_ylabel(r"$p_0$")
    axes[0].set_title("WRE-QC-1: ED vs classical construction")
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].scatter(arr["coupling"], arr["trace_distance"], c=arr["beta"], cmap="viridis", label="match residual")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Quantum coupling")
    axes[1].set_ylabel("Trace distance")
    axes[1].set_title("WRE-QC-1: state residual")

    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "wre_qc_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "wre_qc_1.png"), dpi=150)
    plt.close(fig)

    # Backward-compatible filename for manuscript include.
    fig2, axes2 = plt.subplots(1, 2, figsize=(8, 3.5))
    betas = np.unique(arr["beta"])
    for beta in betas[: min(3, len(betas))]:
        sub = arr[arr["beta"] == beta]
        order = np.argsort(sub["coupling"])
        sub = sub[order]
        axes2[0].plot(sub["coupling"], sub["p0_pred"], "-")
        axes2[0].plot(sub["coupling"], sub["p0_exact"], "o")
    axes2[0].set_xlabel("Quantum coupling")
    axes2[0].set_ylabel(r"$p_0$")
    axes2[0].set_title("ED vs classical construction")
    axes2[1].plot(arr["coupling"], arr["kappa4_cost"], "r-")
    axes2[1].set_xlabel("Quantum coupling")
    axes2[1].set_ylabel("required kappa4")
    axes2[1].set_title("Classical non-Gaussian cost")
    fig2.tight_layout()
    fig2.savefig(os.path.join(fig_dir, "qc_equivalence_demo.pdf"), dpi=150)
    fig2.savefig(os.path.join(fig_dir, "qc_equivalence_demo.png"), dpi=150)
    plt.close(fig2)


def plot_ncng(data_dir, fig_dir):
    arr = read_csv(os.path.join(data_dir, "ncng_wre_ncng_1.csv"))

    fig, ax = plt.subplots(figsize=(7, 4.6))
    betas = np.unique(arr["beta"])
    couplings = np.unique(arr["coupling"])

    # Plot representative subset to keep figure readable.
    for b in betas[: min(3, len(betas))]:
        for g in couplings[: min(2, len(couplings))]:
            sub = arr[(arr["beta"] == b) & (arr["coupling"] == g)]
            order = np.argsort(sub["basis_size"])
            sub = sub[order]
            ax.plot(sub["basis_size"], sub["residual"], marker="o", linestyle="-", label=fr"$\beta={b}, g={g}$")

    ax.set_yscale("log")
    ax.set_xlabel("Basis size")
    ax.set_ylabel("Residual")
    ax.set_title("WRE-NCNG-1: basis saturation")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "wre_ncng_1.pdf"), dpi=150)
    fig.savefig(os.path.join(fig_dir, "wre_ncng_1.png"), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Plot what_rules suite")
    p.add_argument("--outdir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.dirname(__file__)
    out_dir = args.outdir or os.path.abspath(os.path.join(base_dir, "..", "results"))
    data_dir = os.path.join(out_dir, "data")
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    plot_ma(data_dir, fig_dir)
    plot_nu(data_dir, fig_dir)
    plot_qc(data_dir, fig_dir)
    plot_ncng(data_dir, fig_dir)
    print("what_rules figures generated")


if __name__ == "__main__":
    main()
