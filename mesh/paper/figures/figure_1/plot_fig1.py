"""Figure 1: scaling schematic, HF-only vs multi-fidelity at lambda*.

Three stacked log-log panels (no tick labels, schematic):
  row 1  Poisson 3D, solver truncation, U-Net
  row 2  linear elasticity, solver truncation, AB-UPT (annotated with the savings at
         eps* = 10%, taken from savings_summary.csv)
  row 3  airfoil volume field, AB-UPT

Inputs:  paper/results/*.csv, paper/figures/results/savings_summary.csv (from condensed.py)
Outputs: fig1_scaling_hf_vs_mf.{pdf,png,svg} (+ transparent) and fig1_legend.* (this directory)
Usage:   python plot_fig1.py   (after results/condensed.py)
"""
import os.path as osp
import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
RESULTS = f"{REPO}/paper/results"
sys.path.insert(0, REPO)
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import color_baseline, color_ours  # noqa: E402

matplotlib.rcParams.update({
    "text.usetex": False, "font.family": "serif", "mathtext.fontset": "cm",
    "font.size": 6, "axes.labelsize": 10, "xtick.labelsize": 5, "ytick.labelsize": 5,
    "legend.fontsize": 5, "axes.linewidth": 0.5, "grid.linewidth": 0.3,
    "lines.linewidth": 1.0, "lines.markersize": 3, "grid.alpha": 0.3,
    "legend.framealpha": 0.8, "legend.edgecolor": "0.8", "legend.fancybox": False,
    "savefig.dpi": 300,
})
STAR_MS = 8
PANEL_W, PANEL_H, PANEL_GAP = 2.0, 1.2, 4.0


def hf_curve(df_hf, err):
    g = df_hf.groupby("b_tilde")[err].agg(["mean", "std"]).fillna(0.0).sort_index()
    return g.index.values, g["mean"].values, g["std"].values


def mf_optimal(df_mf, err):
    bts, mu, sd = [], [], []
    for bt in sorted(df_mf.b_tilde.unique()):
        st = df_mf[df_mf.b_tilde == bt].groupby("lambd")[err].agg(["mean", "std"]).fillna(0.0)
        best = st.loc[st["mean"].idxmin()]
        bts.append(bt); mu.append(best["mean"]); sd.append(best["std"])
    return np.array(bts), np.array(mu), np.array(sd)


def poisson_row():
    """Row 1: Poisson 3D solver (truncation), U-Net.
    HF baseline = max-lambda cell per budget."""
    p = pd.read_csv(f"{RESULTS}/poisson.csv")
    d = p[(p.dim == 3) & (p.axis == "truncation") & (p.architecture == "unet")].copy()
    err = "mean_prediction_error"
    hf_b, hf_m, hf_s = [], [], []
    for bt in sorted(d.B_tilde.unique()):
        b = d[d.B_tilde == bt]
        hf = b[b.lambd == b.lambd.max()]
        hf_b.append(bt); hf_m.append(hf[err].mean())
        hf_s.append(hf[err].std(ddof=1) if len(hf) > 1 else 0.0)
    d = d.rename(columns={"B_tilde": "b_tilde"})
    return (np.array(hf_b), np.array(hf_m), np.array(hf_s)), mf_optimal(d, err)


def plate_row(csv, model, err):
    df = pd.read_csv(f"{RESULTS}/{csv}")
    df.loc[df.model_name == "ABUPT", "model_name"] = "AB-UPT"
    df = df[df.model_name == model]
    return hf_curve(df[df.hf_only], err), mf_optimal(df[~df.hf_only], err)


SAVINGS_CSV = f"{REPO}/paper/figures/results/savings_summary.csv"


def btilde_at_error(bts, errs, thr):
    """Log-log interpolation of the budget where a curve crosses `thr`."""
    o = np.argsort(bts); b, e = np.asarray(bts)[o], np.asarray(errs)[o]
    for i in range(len(e) - 1):
        if e[i] >= thr >= e[i + 1]:
            t = (np.log10(thr) - np.log10(e[i])) / (np.log10(e[i + 1]) - np.log10(e[i]))
            return 10 ** (np.log10(b[i]) + t * (np.log10(b[i + 1]) - np.log10(b[i])))
    return np.nan


def annotate_savings(ax, hf, mf, thr, label):
    """Horizontal bracket at error `thr` from the MF crossing to the HF crossing,
    with the savings ratio written above its (geometric) midpoint."""
    b_mf = btilde_at_error(mf[0], mf[1], thr)
    b_hf = btilde_at_error(hf[0], hf[1], thr)
    # arrow MF -> HF: "you need <label> the left budget to reach the same error"
    ax.annotate("", xy=(b_hf, thr), xytext=(b_mf, thr), zorder=20,
                arrowprops=dict(arrowstyle="-|>", color="0.15", lw=0.9,
                                mutation_scale=7, shrinkA=0, shrinkB=0))
    ax.plot([b_mf], [thr], color="0.15", marker="|", ms=5, markeredgewidth=0.9,
            ls="none", zorder=20)  # start cap on the MF curve
    ax.text(np.sqrt(b_mf * b_hf), thr * 1.12, label, ha="center", va="bottom",
            fontsize=8, fontweight="bold", color="0.15", zorder=21)
    return b_mf, b_hf


def schematic(ax):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_locator(mticker.LogLocator(base=10.0))
        a.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1,
                                               numticks=10))
        a.set_major_formatter(mticker.NullFormatter())
        a.set_minor_formatter(mticker.NullFormatter())


def draw(ax, hf, mf):
    hb, hm, hs = hf; mb, mm, ms = mf
    ax.plot(hb, hm, color=color_baseline, ls="--", marker="o", ms=3, lw=1.0)
    ax.fill_between(hb, np.maximum(hm - hs, 1e-10), hm + hs, color=color_baseline, alpha=0.15)
    ax.plot(mb, mm, color=color_ours, ls="-", marker="*", ms=STAR_MS, lw=1.0,
            markeredgecolor="white", markeredgewidth=0.2)
    ax.fill_between(mb, np.maximum(mm - ms, 1e-10), mm + ms, color=color_ours, alpha=0.15)
    schematic(ax)


def main():
    rows = [
        poisson_row(),
        plate_row("linear_elasticity_solver_truncation.csv", "AB-UPT",
                  "val/rel_l2_loss_fields_avg"),
        plate_row("airfoil.csv", "AB-UPT",
                  "val/volume_rel_l2_loss_fields_avg"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(PANEL_W, 3 * PANEL_H))
    for ax, (hf, mf) in zip(axes, rows):
        draw(ax, hf, mf)
    # eps*=10% savings bracket on the linear-elasticity panel; label = the paper's
    # reported ratio (bootstrap median from savings_summary.csv), endpoints = the
    # plotted seed-mean curves (their ratio, 6.28x, rounds to the same 6.3x)
    sv = pd.read_csv(SAVINGS_CSV)
    ratio = float(sv[(sv.problem == "linear_elasticity") & (sv.axis == "Solver error")
                     & (sv.model == "AB-UPT")].ratio.iloc[0])
    b_mf, b_hf = annotate_savings(axes[1], rows[1][0], rows[1][1], 0.10, f"{ratio:.1f}\u00d7")
    print(f"bracket: eps*=10%  B~_MF={b_mf:.0f}  B~_HF={b_hf:.0f}  (endpoint ratio {b_hf/b_mf:.2f}x, label {ratio:.1f}x)")
    fig.tight_layout(pad=0.5, h_pad=PANEL_GAP)
    for ext in ("pdf", "png", "svg"):
        fig.savefig(f"{HERE}/fig1_scaling_hf_vs_mf.{ext}")
        fig.savefig(f"{HERE}/fig1_scaling_hf_vs_mf_transparent.{ext}", transparent=True)
    plt.close(fig)
    print("wrote fig1_scaling_hf_vs_mf.*")

    handle_mf = Line2D([], [], color=color_ours, ls="-", marker="*", ms=STAR_MS, lw=1.0,
                       markeredgecolor="white", markeredgewidth=0.2)
    handle_hf = Line2D([], [], color=color_baseline, ls="--", marker="o", ms=3, lw=1.0)
    fig_leg = plt.figure(figsize=(PANEL_W, 0.3))
    fig_leg.legend([handle_mf, handle_hf],
                   [r"Optimal mixture ($\lambda^{\star}$)", "High-fidelity only"],
                   loc="center", ncol=2, frameon=False, handlelength=2.2,
                   columnspacing=1.5, handletextpad=0.6,
                   prop={"weight": "bold", "size": 8})
    for ext in ("pdf", "png", "svg"):
        fig_leg.savefig(f"{HERE}/fig1_legend.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_leg)
    print("wrote fig1_legend.*")


if __name__ == "__main__":
    main()
