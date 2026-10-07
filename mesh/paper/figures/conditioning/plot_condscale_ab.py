"""Conditioning-encoding resolution ablation panels at a fixed matched cell.

One panel per resolution arm (ell_min = 2pi/s for s in {1, 50, 1000}): GT solver
error (dashed gray), the arm's curve through the 16 TRAINING (menu) levels with a
seed band, and hollow circles at ALL 15 midpoints (the object of interest). All
panels of a cell share identical y-limits so they compare side by side. Prints a
midpoint-smoothness metric per arm: excess = err(mid) - mean(err of adjacent menu levels).

Outputs (next to this script):
  condscale_<s>_ab_b<B~>.pdf   (one per arm with data)

Inputs: paper/results/xi_sweep_hyperelasticity_solver.csv, solver error from
data/hyperelasticity/solver_truncation/processed/metadata.csv.
Usage:  python plot_condscale_ab.py [--b-tilde 1600]
"""
import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import color_baseline_2, color_ours  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "results"))
import common as C  # noqa: E402

REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
SWEEP_CSV = osp.join(REPO, "paper", "results", "xi_sweep_hyperelasticity_solver.csv")
GT_CSV = osp.join(REPO, "data/hyperelasticity/solver_truncation/processed/metadata.csv")
ERROR_COLUMN = "val/rel_l2_loss_fields_avg"
VARIANTS = [  # (experiment_id, s, label, color); arms without swept rows are skipped
    ("camera_ready_hyperelasticity_solver_condscale1_0", 1,
     r"$\ell_{\min}=2\pi$", "#C05746"),
    ("camera_ready_hyperelasticity_solver_condscale50_0", 50,
     r"$\ell_{\min}=2\pi/50$", "#4878A8"),
    ("camera_ready_hyperelasticity_solver_truncation_0", 1000,
     r"$\ell_{\min}=2\pi/1000$", color_ours),
]


def agg(df):
    return (df.groupby(["conditioning_fidelity", "in_menu", "extrapolation"])
            [ERROR_COLUMN].agg(["mean", "std"]).reset_index()
            .sort_values("conditioning_fidelity").reset_index(drop=True))


def midpoint_excess(surr):
    menu = surr[surr.in_menu].reset_index(drop=True)
    mids = surr[~surr.in_menu & ~surr.extrapolation].reset_index(drop=True)
    rows = []
    for _, m in mids.iterrows():
        lo = menu[menu.conditioning_fidelity < m.conditioning_fidelity].iloc[-1]
        hi = menu[menu.conditioning_fidelity > m.conditioning_fidelity].iloc[0]
        rows.append(dict(xi=m.conditioning_fidelity,
                         excess=m["mean"] - (lo["mean"] + hi["mean"]) / 2))
    return pd.DataFrame(rows)


COLOR_INTERPOLATION = "#4878A8"


def _finish_axes(ax, ylim):
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xlabel(r"$\hat{\xi}$")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.grid(True, alpha=0.3, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_arm_panel(surr, gt, menu_x, ylim, out):
    """Single arm in the MAIN figure's visual language (green Training line/dots,
    blue hollow Interpolation circles); the arm identity goes in the subcaption."""
    fig, ax = plt.subplots(figsize=(1.8, 1.8), constrained_layout=True)
    ax.plot(menu_x, gt["mean"], color=color_baseline_2, linestyle="--",
            label="Numerical solver", zorder=2)
    menu = surr[surr.in_menu]
    mids = surr[~surr.in_menu & ~surr.extrapolation]
    ax.plot(menu.conditioning_fidelity, menu["mean"], color=color_ours,
            linewidth=0.8, zorder=3)
    ax.fill_between(menu.conditioning_fidelity, menu["mean"] - menu["std"],
                    menu["mean"] + menu["std"], color=color_ours, alpha=0.18,
                    linewidth=0, zorder=2)
    ax.scatter(menu.conditioning_fidelity, menu["mean"], marker="o",
               color=color_ours, s=8, label="Training", zorder=6)
    ax.scatter(mids.conditioning_fidelity, mids["mean"], marker="o",
               facecolors="none", edgecolors=COLOR_INTERPOLATION, s=8,
               label="Interpolation", zorder=7)
    _finish_axes(ax, ylim)
    ax.legend(loc="upper right", frameon=True, edgecolor="0.8", fancybox=False,
              borderpad=0.3, handlelength=1.5, handletextpad=0.5)
    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b-tilde", type=int, default=1600)
    args = p.parse_args()

    C.apply_style()
    df = pd.read_csv(SWEEP_CSV)
    df = df[(df.b_tilde == args.b_tilde) & df.swept]
    arms = []
    for exp, s, label, color in VARIANTS:
        sub = df[df.experiment == exp]
        if sub.empty:
            continue
        surr = agg(sub)
        ex = midpoint_excess(surr)
        hi = ex[ex.xi >= 0.5]
        menu = surr[surr.in_menu]
        print(f"s={s} ({label}): cond@1.0 {menu['mean'].iloc[-1]:.4f} | midpoint "
              f"excess (xi>=0.5) mean {hi.excess.mean():+.4f} max {hi.excess.max():+.4f}")
        arms.append((s, surr, label, color))
    print(f"B~={args.b_tilde}: {len(arms)} arms")

    menu_x = arms[0][1][arms[0][1].in_menu].conditioning_fidelity.values
    gt = (pd.read_csv(GT_CSV).groupby("fidelity_raw")["rel_l2_error_vm"]
          .agg(["mean"]).reset_index().sort_values("fidelity_raw"))

    # shared y-limits across all panels of this cell (small headroom, floor at 0)
    hi = max(max((s[1]["mean"] + s[1]["std"].fillna(0)).max() for s in
                 [(None, a[1]) for a in arms]), gt["mean"].max())
    ylim = (-0.01, hi * 1.06)

    for s, surr, label, color in arms:
        plot_arm_panel(surr, gt, menu_x, ylim,
                       osp.join(HERE, f"condscale_{s}_ab_b{args.b_tilde}.pdf"))


if __name__ == "__main__":
    main()
