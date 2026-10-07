"""Conditioning figure: surrogate test error vs the fidelity input.

Test error of the conditioned model at a matched cell (Transolver at B~ = 400, 1600 and 6400,
each at its optimal mixture) as a function of the fidelity input xi-hat it is given, overlaid with
the numerical solver's error per fidelity and the cost-matched unconditioned baseline (flat by
construction: one evaluation at its training-time input).

Inputs:  paper/results/xi_sweep_hyperelasticity_solver.csv (from xi_sweep_eval.py)
         data/hyperelasticity/solver_truncation/processed/metadata.csv (solver error)
Outputs: conditioning_hyperelasticity_solver_b<B~>_main.pdf for B~ = 400, 1600, 6400 (this directory)
Usage:   python plot_conditioning.py [--b-tilde 1600] [--log-y]
"""
import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import color_baseline, color_baseline_2, color_ours  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "results"))
import common as C  # noqa: E402

REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
SWEEP_CSV = osp.join(REPO, "paper", "results", "xi_sweep_hyperelasticity_solver.csv")
GT_CSV = osp.join(REPO, "data/hyperelasticity/solver_truncation/processed/metadata.csv")
# main hyperelasticity solver results -- source of the HF-only reference line
MAIN_EVAL_CSV = osp.join(REPO, "paper", "results", "hyperelasticity_solver_truncation.csv")
MODEL = "Transolver"
ERROR_COLUMN = "val/rel_l2_loss_fields_avg"
GT_ERROR_COLUMN = "rel_l2_error_vm"

COLOR_EXTRAPOLATION = "#C05746"
COLOR_INTERPOLATION = "#4878A8"


def parse_args():
    p = argparse.ArgumentParser(description="Conditioning figure.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--b-tilde", type=int, nargs="+", default=[400, 1600, 6400],
                   help="matched cells to plot (must be swept in the sweep CSV)")
    p.add_argument("--experiment",
                   default="camera_ready_hyperelasticity_solver_truncation_0",
                   help="experiment_id of the swept (conditioned) runs; e.g. the "
                        "condscale50 pilot lives at the same cell and must not be mixed")
    p.add_argument("--log-y", action="store_true", help="log-scale y axis")
    p.add_argument("--sweep-csv", default=SWEEP_CSV)
    p.add_argument("--gt-csv", default=GT_CSV)
    return p.parse_args()


def main():
    args = parse_args()
    C.apply_style()
    tag = ("" if args.experiment.endswith("truncation_0")
           else "_" + args.experiment.split("solver_")[-1].removesuffix("_0"))
    for b_tilde in args.b_tilde:
        plot_cell(args, b_tilde, osp.join(HERE, f"conditioning_hyperelasticity_solver_b{b_tilde}{tag}_main.pdf"))


def plot_cell(args, b_tilde, out):
    df = pd.read_csv(args.sweep_csv)
    df = df[df.b_tilde == b_tilde]
    cond = df[df.swept & (df.experiment == args.experiment)]
    uncond = df[~df.swept]  # uncond ablation rows (their own experiment_id)
    if cond.empty:
        raise SystemExit(f"no swept rows for B~={b_tilde} / {args.experiment} "
                         f"in {args.sweep_csv}")

    # conditioned sweep: mean/std across seeds per forced xi-hat value
    surr = (cond.groupby(["conditioning_fidelity", "in_menu", "extrapolation"])
            [ERROR_COLUMN].agg(["mean", "std"]).reset_index()
            .sort_values("conditioning_fidelity").reset_index(drop=True))
    surr_in = surr[surr.in_menu].reset_index(drop=True)
    surr_interp = surr[~surr.in_menu & ~surr.extrapolation].reset_index(drop=True)
    surr_extrap = surr[surr.extrapolation].reset_index(drop=True)

    # ground truth: per-fidelity solver error, rank-aligned to the menu xi-hat values
    gt = (pd.read_csv(args.gt_csv).groupby("fidelity_raw")[GT_ERROR_COLUMN]
          .agg(["mean", "std"]).reset_index().sort_values("fidelity_raw")
          .reset_index(drop=True))
    assert len(gt) == len(surr_in), (
        f"GT has {len(gt)} fidelity levels but the menu sweep has {len(surr_in)}")
    gt_x = surr_in.conditioning_fidelity.values

    # HF-only reference at the same B~ (canonical eval of the main campaign)
    ev = pd.read_csv(MAIN_EVAL_CSV)
    hf_rows = ev[(ev.model_name == MODEL) & (ev.b_tilde == b_tilde) & ev.hf_only]
    hf_mean, hf_std = hf_rows[ERROR_COLUMN].mean(), hf_rows[ERROR_COLUMN].std()

    # uncond baseline: one natural eval per run, replicated over the axis -> dedupe
    per_run = uncond.groupby("run_id")[ERROR_COLUMN].first()
    no_cond_mean, no_cond_std = per_run.mean(), per_run.std()
    n_uncond = uncond.run_id.nunique()
    print(f"B~={b_tilde} (lambda={cond.lambd.iloc[0]:g}): cond seeds "
          f"{sorted(cond.seed.unique())}, uncond runs: {n_uncond}, "
          f"uncond mean: {no_cond_mean:.4f}, HF-only: {hf_mean:.4f}, cond@1.0: "
          f"{surr_in['mean'].iloc[-1]:.4f}, GT@HF: {gt['mean'].iloc[-1]:.4f}")

    # line/band through TRAINING (menu) points only, ALL menu levels marked (the GT
    # line is unmarked, so the green dots carry the trained-level positions);
    # interpolation/extrapolation are free-standing subsampled markers
    in_markers = surr_in
    interp_markers = surr_interp.iloc[1::3]
    extrap_markers = surr_extrap.iloc[[1]]  # single marker at xi-hat = 1.10

    fig, ax = plt.subplots(figsize=(1.8, 1.8), constrained_layout=True)

    ax.plot(gt_x, gt["mean"], color=color_baseline_2, linestyle="--",
            label="Numerical solver", zorder=2)
    ax.fill_between(gt_x, gt["mean"] - gt["std"], gt["mean"] + gt["std"],
                    color=color_baseline_2, alpha=0.15, linewidth=0, zorder=1)

    if pd.notna(hf_mean):
        ax.axhline(y=hf_mean, color=color_baseline, linestyle="--",
                   linewidth=0.8, label="HF only", zorder=2.5)
        ax.axhspan(hf_mean - hf_std, hf_mean + hf_std, color=color_baseline,
                   alpha=0.12, linewidth=0, zorder=1.5)

    if n_uncond:
        ax.axhline(y=no_cond_mean, color="black", linestyle="--",
                   linewidth=0.8, label="No conditioning", zorder=2.5)
        ax.axhspan(no_cond_mean - no_cond_std, no_cond_mean + no_cond_std,
                   color="black", alpha=0.10, linewidth=0, zorder=1.5)

    ax.plot(surr_in.conditioning_fidelity, surr_in["mean"], color=color_ours,
            linewidth=0.8, zorder=3)
    ax.fill_between(surr_in.conditioning_fidelity, surr_in["mean"] - surr_in["std"],
                    surr_in["mean"] + surr_in["std"], color=color_ours, alpha=0.18,
                    linewidth=0, zorder=2)

    ax.scatter(in_markers.conditioning_fidelity, in_markers["mean"], marker="o",
               color=color_ours, s=8, label="Training", zorder=6)
    ax.scatter(interp_markers.conditioning_fidelity, interp_markers["mean"],
               marker="o", facecolors="none", edgecolors=COLOR_INTERPOLATION, s=8,
               label="Interpolation", zorder=7)
    ax.scatter(extrap_markers.conditioning_fidelity, extrap_markers["mean"],
               marker="o", facecolors="none", edgecolors=COLOR_EXTRAPOLATION, s=8,
               label="Extrapolation", zorder=7)

    if args.log_y:
        ax.set_yscale("log")
    else:
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.set_xlabel(r"$\hat{\xi}$")
    ax.set_ylabel(r"Relative $L_2$ error")
    ax.grid(True, alpha=0.3, linestyle="--")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    leg = ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2,
                    frameon=True, edgecolor="0.8", fancybox=False, borderpad=0.3,
                    handlelength=1.5, columnspacing=1.2, handletextpad=0.5)
    # align the legend's left edge with the left edge of the y tick labels
    fig.canvas.draw()
    tick_left = min(t.get_window_extent().x0 for t in ax.get_yticklabels()
                    if t.get_text())
    x_axes = ax.transAxes.inverted().transform((tick_left, 0))[0]
    leg.set_bbox_to_anchor((x_axes, 1.0), transform=ax.transAxes)

    fig.savefig(out, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
