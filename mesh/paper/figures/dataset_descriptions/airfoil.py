"""Airfoil dataset-description figures (appendix: mesh-refinement convergence).

Reads the released dataset through data/airfoil/processed/metadata.csv (one row per case and level) and
writes, next to this file,

    airfoil_discretization_convergence_fields.pdf   median rel. L2 error of u_x, u_y, p and their
                                                    average vs. mesh nodes, 10-90% band
    airfoil_discretization_convergence_coeffs.pdf   median relative error of Cd, Cl and their
                                                    average vs. mesh nodes, interquartile band

Field errors come from the metadata (every level resampled onto the case's finest mesh);
coefficient errors are computed here against the finest level of the same case. The fitted
power-law exponents shown in the legends are also printed.

Usage:  python airfoil.py
"""
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import color_baseline, color_ours  # noqa: E402

matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 5, "ytick.labelsize": 5,
    "legend.fontsize": 5, "axes.linewidth": 0.5, "grid.linewidth": 0.3,
    "lines.linewidth": 1, "lines.markersize": 2, "grid.alpha": 0.3,
    "legend.framealpha": 0.8, "legend.edgecolor": "0.8", "legend.fancybox": False,
    "savefig.dpi": 300,
})
COLUMN_WIDTH = 5.5
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
METADATA = osp.join(REPO, "data", "airfoil", "processed", "metadata.csv")


def add_coefficient_errors(df, hf_value=1.0, coefficients=("cd_openfoam", "cl_openfoam"), eps=1e-12):
    """Relative error of the force coefficients against the finest level of the same case,
    written to rel_l2_error_cd / rel_l2_error_cl (naming kept for consistency with the fields)."""
    hf = df[df["meshing_fidelity"] == hf_value][["case_id", *coefficients]]
    if hf.empty:
        raise ValueError(f"no rows with meshing_fidelity == {hf_value}")
    hf = hf.drop_duplicates("case_id").rename(columns={c: f"_hf_{c}" for c in coefficients})
    df = df.merge(hf, on="case_id", how="left")
    for c in coefficients:
        df[f"rel_l2_error_{c.split('_')[0]}"] = (df[c] - df[f"_hf_{c}"]).abs() / df[f"_hf_{c}"].abs().clip(lower=eps)
    return df.drop(columns=[f"_hf_{c}" for c in coefficients])


def plot_convergence(df, error_columns, column_labels, out_pdf, band, y_label, average_label="Avg."):
    """Median error per level vs. mean node count (finest level excluded), percentile band,
    power law fitted to the medians in log-log space. Returns {label: exponent}."""
    lo_q, hi_q = {"iqr": (0.25, 0.75), "p10_p90": (0.10, 0.90)}[band]
    df = df.copy()
    df["_avg"] = df[error_columns].mean(axis=1)
    g = df.groupby("meshing_fidelity")
    n_nodes = g["n_nodes"].mean().iloc[:-1]
    keep = n_nodes.index
    n = n_nodes.values

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.48, COLUMN_WIDTH * 0.48 / GOLDEN_RATIO), constrained_layout=True)
    slopes = {}
    series = list(zip(error_columns, column_labels, ["o", "s", "^", "D"], [color_baseline] * 4)) + [("_avg", average_label, "o", color_ours)]
    for col, label, marker, color in series:
        med = g[col].median().loc[keep].values
        lo, hi = g[col].quantile(lo_q).loc[keep].values, g[col].quantile(hi_q).loc[keep].values
        slope, _ = np.polyfit(np.log(n), np.log(med), 1)
        slopes[label] = slope
        ax.plot(n, med, label=f"{label} $\\mathcal{{O}}(N^{{{slope:.2f}}})$", marker=marker, color=color, linewidth=1.5)
        ax.fill_between(n, np.clip(lo, 1e-12, None), hi, alpha=0.18, color=color, linewidth=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator([5e3, 1e4, 2e4, 5e4]))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Number of nodes ($N$)"); ax.set_ylabel(y_label)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True, edgecolor="0.8", fancybox=False)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    fig.savefig(out_pdf, format="pdf"); plt.close(fig)
    return slopes


def main():
    df = pd.read_csv(METADATA, dtype={"sample_id": str, "fidelity_id": str})
    df = add_coefficient_errors(df)
    print(f"airfoil: {df.case_id.nunique()} cases x {df.fidelity_id.nunique()} levels, "
          f"mean nodes {df.groupby('fidelity_id').n_nodes.mean().min():.0f}..{df.groupby('fidelity_id').n_nodes.mean().max():.0f}")
    s = plot_convergence(df, ["rel_l2_error_ux", "rel_l2_error_uy", "rel_l2_error_p"], ["$u_x$", "$u_y$", "$p$"],
                         osp.join(HERE, "airfoil_discretization_convergence_fields.pdf"), band="p10_p90", y_label=r"Relative $L_2$ error")
    print("fields:       " + ", ".join(f"{k} O(N^{v:.2f})" for k, v in s.items()))
    s = plot_convergence(df, ["rel_l2_error_cd", "rel_l2_error_cl"], ["$C_d$", "$C_l$"],
                         osp.join(HERE, "airfoil_discretization_convergence_coeffs.pdf"), band="iqr", y_label="Relative error")
    print("coefficients: " + ", ".join(f"{k} O(N^{v:.2f})" for k, v in s.items()))
    print(f"wrote airfoil_discretization_convergence_{{fields,coeffs}}.pdf to {HERE}")


if __name__ == "__main__":
    main()
