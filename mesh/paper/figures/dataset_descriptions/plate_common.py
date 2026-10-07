"""Shared plotting code for the plate (solid-mechanics) dataset-description figures.

Reads the released datasets through the repo's data/ layout:

    data/<problem>/discretization/processed      (mesh-refinement axis, 21 levels)
    data/<problem>/solver_truncation/processed   (solver-truncation axis)

Each processed sample is  <processed>/<sample_id>/<fidelity_id>.h5  with
data/fields (N, C), channels/<name> -> column indices, mesh/connectivity (T, 3).

Produces, per problem:
    <problem>_discretization_convergence.pdf        error vs. N with the fitted O(N^-p)
    <problem>_discretization_lf_hf_comparison.pdf   von Mises field, coarsest vs finest mesh
    <problem>_solver_convergence.pdf                error vs. iterations with the fitted O(rho^k)
    <problem>_solver_lf_hf_comparison.pdf           von Mises field, lowest vs highest budget + error
and prints the fitted exponent / rho, which are the numbers quoted in the appendix.
"""
import os.path as osp
import sys

import h5py
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as mtri
import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
DATA = osp.join(REPO, "data")
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import (  # noqa: E402
    cmap_diverging, cmap_sequential, color_baseline_2, color_ours,
)

matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 6, "axes.labelsize": 7, "xtick.labelsize": 5, "ytick.labelsize": 5,
    "legend.fontsize": 5, "axes.linewidth": 0.5, "grid.linewidth": 0.3,
    "lines.linewidth": 1.5, "lines.markersize": 4, "grid.alpha": 0.3,
    "legend.framealpha": 0.8, "legend.edgecolor": "0.8", "legend.fancybox": False,
    "savefig.dpi": 300,
})
COLUMN_WIDTH = 5.5
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2
ERROR_COLUMN = "rel_l2_error_vm"


# ----------------------------------------------------------------------------- data
def processed_dir(problem, axis):
    return osp.join(DATA, problem, axis, "processed")


def load_metadata(problem, axis):
    return pd.read_csv(osp.join(processed_dir(problem, axis), "metadata.csv"),
                       dtype={"sample_id": str, "fidelity_id": str})


def load_sample(problem, axis, sample_id, fidelity_id):
    """Returns coords (N, 2), connectivity (T, 3), von Mises (N,) of one h5 sample."""
    with h5py.File(osp.join(processed_dir(problem, axis), sample_id, f"{fidelity_id}.h5"), "r") as f:
        fields = f["data"]["fields"][:]
        ch = {k: f["channels"][k][()] for k in f["channels"]}
        conn = f["mesh"]["connectivity"][:]
    return fields[:, ch["coords"]], conn, fields[:, ch["vm"]].squeeze()


def extreme_levels(df):
    ids = sorted(df.fidelity_id.unique())
    return ids[0], ids[-1]


# ----------------------------------------------------------------------- convergence
def plot_discretization_convergence(df, out_pdf, error_column=ERROR_COLUMN):
    """Mean +- std of the error vs. mean node count per level (reference level excluded),
    with a least-squares power law fitted in log-log space. Returns the fitted exponent."""
    g = df.groupby("meshing_fidelity")[["n_nodes", error_column]]
    mean, std = g.mean().iloc[:-1], g.std().iloc[:-1]
    n_nodes, e = mean["n_nodes"].values, mean[error_column].values
    slope, intercept = np.polyfit(np.log(n_nodes), np.log(e), 1)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.48, COLUMN_WIDTH * 0.48 / GOLDEN_RATIO),
                           constrained_layout=True)
    ax.plot(n_nodes, e, label=r"Mean $\pm$ std", marker="o", linewidth=2, color=color_ours)
    ax.fill_between(n_nodes, e - std[error_column].values, e + std[error_column].values,
                    alpha=0.3, color=color_ours)
    dense = np.logspace(np.log10(n_nodes.min()), np.log10(n_nodes.max()), 100)
    ax.plot(dense, np.exp(intercept) * dense**slope, label=f"$\\mathcal{{O}}(N^{{{slope:.2f}}})$",
            linestyle="--", color=color_baseline_2, linewidth=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # minor labels collide over 212..7859
    ax.set_xlabel("Number of Nodes ($N$)"); ax.set_ylabel(r"Rel. $L_2$ error")
    ax.legend(); ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(out_pdf, format="pdf"); plt.close(fig)
    return slope


def plot_solver_convergence(df, out_pdf, rho_decimals, error_column=ERROR_COLUMN):
    """Mean +- std of the error vs. iteration budget (reference level excluded), with
    rho fitted from log(mean error) vs. k. Returns the fitted rho."""
    sub = df[df["solver_max_its"] < df["solver_max_its"].max()]
    g = sub.groupby("solver_max_its")[error_column]
    mean, std = g.mean(), g.std()
    k = mean.index.values.astype(float)
    slope, _ = np.polyfit(k, np.log(mean.values), 1)
    rho = float(np.exp(slope))

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH * 0.48, COLUMN_WIDTH * 0.48 / GOLDEN_RATIO),
                           constrained_layout=True)
    ax.plot(k, mean.values, label=r"Mean $\pm$ std", linewidth=2, marker="o", color=color_ours)
    ax.fill_between(k, mean.values - std.values, mean.values + std.values, alpha=0.3, color=color_ours)
    ax.plot(k, mean.values[0] * rho ** (k - k[0]), color="black", linestyle="--", alpha=0.7,
            label=f"$\\mathcal{{O}}({{{rho:.{rho_decimals}f}}}^k)$")
    ax.set_xlabel(r"Solver iterations ($k$)"); ax.set_ylabel(r"Rel. $L_2$ error")
    ax.set_yscale("log"); ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    fig.savefig(out_pdf, format="pdf"); plt.close(fig)
    return rho


# ------------------------------------------------------------------ field comparisons
def _sci_colorbar(fig, mappable, cax, label, orientation):
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation,
                        format=ticker.ScalarFormatter(useMathText=True))
    cbar.ax.ticklabel_format(style="scientific", scilimits=(0, 0))
    (cbar.ax.set_xlabel if orientation == "horizontal" else cbar.ax.set_ylabel)(label)
    return cbar


def plot_discretization_comparison(problem, sample_id, out_pdf):
    """Coarsest (top) vs finest (bottom) mesh, von Mises, shared colour scale."""
    df = load_metadata(problem, "discretization")
    lo, hi = extreme_levels(df)
    c_lf, t_lf, vm_lf = load_sample(problem, "discretization", sample_id, lo)
    c_hf, t_hf, vm_hf = load_sample(problem, "discretization", sample_id, hi)
    vmin, vmax = min(vm_lf.min(), vm_hf.min()), max(vm_lf.max(), vm_hf.max())

    fig = plt.figure(figsize=(COLUMN_WIDTH * 0.48, COLUMN_WIDTH * 0.48 * 0.95), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.04], wspace=0.02, hspace=0.02)
    ax_lf, ax_hf, cax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    xlim, ylim = (c_hf[:, 0].min(), c_hf[:, 0].max()), (c_hf[:, 1].min(), c_hf[:, 1].max())
    for ax, c, t, vm in ((ax_lf, c_lf, t_lf, vm_lf), (ax_hf, c_hf, t_hf, vm_hf)):
        tp = ax.tripcolor(mtri.Triangulation(c[:, 0], c[:, 1], t), vm, shading="gouraud",
                          vmin=vmin, vmax=vmax, cmap=cmap_sequential)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    _sci_colorbar(fig, tp, cax, "Von Mises stress", "vertical")
    fig.savefig(out_pdf, format="pdf"); plt.close(fig)
    row = df[(df.sample_id == sample_id) & (df.fidelity_id == lo)].iloc[0]
    return int(row.n_nodes), int(df[(df.sample_id == sample_id) & (df.fidelity_id == hi)].iloc[0].n_nodes), float(row[ERROR_COLUMN])


def plot_solver_comparison(problem, sample_id, out_pdf):
    """Lowest vs highest iteration budget on the fixed mesh, plus their difference."""
    df = load_metadata(problem, "solver_truncation")
    lo, hi = extreme_levels(df)
    c_lf, t_lf, vm_lf = load_sample(problem, "solver_truncation", sample_id, lo)
    c_hf, t_hf, vm_hf = load_sample(problem, "solver_truncation", sample_id, hi)
    vmin, vmax = min(vm_lf.min(), vm_hf.min()), max(vm_lf.max(), vm_hf.max())
    xlim = (min(c_lf[:, 0].min(), c_hf[:, 0].min()), max(c_lf[:, 0].max(), c_hf[:, 0].max()))
    ylim = (min(c_lf[:, 1].min(), c_hf[:, 1].min()), max(c_lf[:, 1].max(), c_hf[:, 1].max()))

    fig = plt.figure(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.32))
    panel_w, panel_h = 0.28, 0.52
    ax_lf = fig.add_axes([0.02, 0.35, panel_w, panel_h])
    ax_hf = fig.add_axes([0.32, 0.35, panel_w, panel_h])
    ax_err = fig.add_axes([0.66, 0.35, panel_w, panel_h])
    cax, cax_err = fig.add_axes([0.05, 0.25, 0.52, 0.03]), fig.add_axes([0.68, 0.25, 0.24, 0.03])

    tri_lf, tri_hf = (mtri.Triangulation(c[:, 0], c[:, 1], t) for c, t in ((c_lf, t_lf), (c_hf, t_hf)))
    ax_lf.tripcolor(tri_lf, vm_lf, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap_sequential)
    tp = ax_hf.tripcolor(tri_hf, vm_hf, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap_sequential)
    err = vm_hf - vm_lf; err_max = float(np.max(np.abs(err)))
    tp_err = ax_err.tripcolor(tri_hf, err, shading="gouraud", vmin=-err_max, vmax=err_max, cmap=cmap_diverging)
    for ax, title in ((ax_lf, "Lowest fidelity"), (ax_hf, "Highest fidelity"), (ax_err, "Error")):
        ax.set_title(title, pad=4); ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
    _sci_colorbar(fig, tp, cax, "Von Mises stress", "horizontal")
    _sci_colorbar(fig, tp_err, cax_err, "Von Mises stress", "horizontal")
    fig.savefig(out_pdf, format="pdf"); plt.close(fig)
    row = df[(df.sample_id == sample_id) & (df.fidelity_id == lo)].iloc[0]
    return int(row.n_nodes), float(row[ERROR_COLUMN])


# ----------------------------------------------------------------------------- driver
def run(problem, rho_decimals, sample_id):
    out = lambda name: osp.join(HERE, f"{problem}_{name}.pdf")  # noqa: E731
    print(f"== {problem} ==")

    df = load_metadata(problem, "discretization")
    p = plot_discretization_convergence(df, out("discretization_convergence"))
    print(f"discretization: {df.sample_id.nunique()} samples x {df.fidelity_id.nunique()} levels, "
          f"N = {df.n_nodes.min()}..{df.n_nodes.max()};  fitted e(N) = O(N^{p:.3f})")
    n_lo, n_hi, e_lo = plot_discretization_comparison(problem, sample_id, out("discretization_lf_hf_comparison"))
    print(f"  comparison figure: sample {sample_id}, {n_lo} vs {n_hi} nodes, lowest-level vm error {e_lo:.3f}")

    df = load_metadata(problem, "solver_truncation")
    rho = plot_solver_convergence(df, out("solver_convergence"), rho_decimals)
    print(f"solver:         {df.sample_id.nunique()} samples x {df.fidelity_id.nunique()} levels, "
          f"budgets {df.solver_max_its.min()}..{df.solver_max_its.max()};  fitted e(k) = O(rho^k), rho = {rho:.5f}")
    n, e_lo = plot_solver_comparison(problem, sample_id, out("solver_lf_hf_comparison"))
    print(f"  comparison figure: sample {sample_id}, {n} nodes, lowest-level vm error {e_lo:.3f}")
    print(f"wrote {problem}_*.pdf to {HERE}")
