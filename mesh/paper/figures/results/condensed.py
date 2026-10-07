"""Condensed result figures of the main paper.

Per problem, one panel pair per fidelity axis for one model (Transolver in the paper; the airfoil
version is made by airfoil.py): iso-budget lambda sweeps (left, B~ colorbar) and the
Pareto/scaling comparison HF-only vs MF at lambda* (right). Plus the savings bar chart S(eps*)
across experiments with seed-bootstrap 95% CIs.

Outputs (this directory): results_<problem>_condensed.pdf, savings_summary.pdf,
savings_summary.csv (read by figure_1/plot_fig1.py)
Usage:  python condensed.py [--model Transolver] [--threshold 0.1] [--n-boot 2000]
"""
import argparse
import os.path as osp

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import common as C

HERE = osp.dirname(osp.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Condensed main-paper result figures.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="Transolver", choices=C.MODELS,
                   help="model shown in the condensed per-problem figures")
    p.add_argument("--threshold", type=float, default=0.1,
                   help="error threshold eps* for the savings bars")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap draws for the savings CIs (0 = no CIs)")
    p.add_argument("--skip-savings", action="store_true",
                   help="only regenerate the condensed figures (skip the savings chart)")
    p.add_argument("--results-dir", default=C.RESULTS_DIR)
    p.add_argument("-o", "--out-dir", default=HERE)
    return p.parse_args()


def plot_condensed(dfs_by_axis, model_name, save_path):
    """dfs_by_axis: {'Discretization error': df, 'Solver error': df} (full dfs)."""
    groups = [(lab, dfs_by_axis[lab][dfs_by_axis[lab].model_name == model_name])
              for lab in ["Discretization error", "Solver error"]]
    all_bt = sorted(set(np.concatenate([g[1][~g[1].hf_only].b_tilde.unique() for g in groups])))
    cmap, norm, sm = C.btilde_colormap(all_bt)

    fig = plt.figure(figsize=(C.COLUMN_WIDTH, C.COLUMN_WIDTH * 0.31))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.2,
                     left=0.10, right=0.995, top=0.78, bottom=0.18)

    for gi, (label, df) in enumerate(groups):
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, gi],
                                        width_ratios=[1, 1], wspace=0.12)
        df_mf, df_hf = df[~df.hf_only], df[df.hf_only]

        ax_l = fig.add_subplot(inner[0, 0])
        for bt, stats in C.iso_budget_stats(df_mf).items():
            color = cmap(norm(bt))
            ax_l.plot(stats.lambd, stats["mean"], color=color, linewidth=0.8)
            ax_l.fill_between(stats.lambd, stats["mean"] - stats["std"],
                              stats["mean"] + stats["std"], color=color, alpha=0.12)
            best = stats.loc[stats["mean"].idxmin()]
            ax_l.plot(best.lambd, best["mean"], marker="*", markersize=5, color=color,
                      markeredgecolor="white", markeredgewidth=0.2, zorder=10)
        ax_l.set_yscale("log")
        ax_l.set_xlabel(r"$\lambda$", labelpad=0, fontsize=8)
        ax_l.grid(True, alpha=0.3, ls="--")
        if gi == 0:
            ax_l.set_ylabel(r"Rel. $L_2$ error", labelpad=2)

        ax_r = fig.add_subplot(inner[0, 1], sharey=ax_l)
        hb, hm, hs = C.hf_curve(df_hf)
        ax_r.plot(hb, hm, color=C.color_baseline, linestyle="--", marker="o",
                  markersize=2, linewidth=0.8, label="HF only")
        ax_r.fill_between(hb, hm - hs, hm + hs, color=C.color_baseline, alpha=0.12)
        ob, om, os_ = C.mf_optimal_curve(df_mf)
        ax_r.plot(ob, om, color=C.color_ours, linestyle="-", marker="*", markersize=5,
                  linewidth=0.8, markeredgecolor="white", markeredgewidth=0.2,
                  label=r"MF ($\lambda^{*}$)")
        ax_r.fill_between(ob, om - os_, om + os_, color=C.color_ours, alpha=0.12)
        ax_r.set_xscale("log")
        ax_r.set_yscale("log")
        ax_r.set_xlabel(r"$\tilde{B}$", labelpad=0, fontsize=8)
        ax_r.grid(True, alpha=0.3, ls="--")
        ax_r.tick_params(axis="y", labelleft=False, left=False, which="both")
        ax_r.legend(loc="upper right", frameon=True, edgecolor="0.8", fancybox=False,
                    fontsize=6)
        C.decade_extreme_ticks(ax_l)
        C.decade_xticks(ax_r, minor_marks=True)
        ax_l.tick_params(axis="y", which="major", labelsize=6)
        ax_l.tick_params(axis="x", which="major", labelsize=6)
        ax_r.tick_params(axis="x", which="major", labelsize=6)

        # horizontal colorbar ABOVE the panel pair, centered
        pl, pr = ax_l.get_position(fig), ax_r.get_position(fig)
        span = pr.x1 - pl.x0
        cb_w = span * 0.62
        cax = fig.add_axes([pl.x0 + (span - cb_w) / 2, pl.y1 + 0.035, cb_w, 0.032])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], which="both", pad=1)
        cax.text(1.04, 0.4, r"$\tilde{B}$", transform=cax.transAxes, va="center",
                 ha="left", fontsize=plt.rcParams["axes.labelsize"])

        fig.text((pl.x0 + pr.x1) / 2, 0.995, label, ha="center", va="top",
                 fontsize=7, fontweight="bold")

    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"wrote {save_path}")


# discretization bars then solver bars per group, gray n/a placeholders for missing combos
COLOR_DISC, COLOR_SOLVER, COLOR_MISSING = "#C05746", "#4878A8", "#CCCCCC"
SAVINGS_GROUPS = [  # (label, problem key, archs, has_solver)
    ("Poisson 1D", "poisson1d", ["U-Net", "FNO"], True),
    ("Poisson 2D", "poisson2d", ["U-Net", "FNO"], True),
    ("Poisson 3D", "poisson3d", ["U-Net"], True),
    ("Lin. elast. 2D", "linear_elasticity", ["Transolver", "AB-UPT"], True),
    ("Hyperelast. 2D", "hyperelasticity", ["Transolver", "AB-UPT"], True),
    ("Airfoil 2D", "airfoil", ["Transolver", "AB-UPT"], False),
]


def plot_savings(data, save_path, threshold):
    """data: list of dicts (problem, axis, model, ratio, lo, hi). Verbatim port of
    bars with bootstrap-CI whiskers."""
    from matplotlib.patches import Patch

    def lookup(prob, axis, model):
        for d in data:
            if (d["problem"], d["axis"], d["model"]) == (prob, axis, model):
                return d
        return None

    bars = []  # (pos, value, color, label, lo, hi)
    group_centers, group_labels = [], []
    x, bar_width, bar_gap, axis_gap, group_gap = 0.0, 0.55, 0.1, 0.35, 1.2
    for label, prob, archs, has_solver in SAVINGS_GROUPS:
        group_start = x
        axes = ["Discretization error"] + (["Solver error"] if has_solver else [])
        for ai, axis in enumerate(axes):
            if ai:
                x += axis_gap
            color = COLOR_DISC if axis == "Discretization error" else COLOR_SOLVER
            for model in archs:
                d = lookup(prob, axis, model)
                missing = d is None or not np.isfinite(d["ratio"]) or d["ratio"] <= 0
                bars.append((x, 1.0 if missing else d["ratio"],
                             COLOR_MISSING if missing else color, model,
                             np.nan if missing else d.get("lo", np.nan),
                             np.nan if missing else d.get("hi", np.nan)))
                x += bar_width + bar_gap
        group_end = x - bar_gap
        group_centers.append((group_start + group_end - bar_width) / 2)
        group_labels.append(label)
        x += group_gap

    fig, ax = plt.subplots(figsize=(C.COLUMN_WIDTH, C.COLUMN_WIDTH * 0.36))
    pos = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    cols = [b[2] for b in bars]
    ax.bar(pos, vals, width=bar_width, color=cols, edgecolor="white", linewidth=0.3)
    for p, v, c, _, lo, hi in bars:
        if np.isfinite(lo):
            # capless whisker in a darkened shade of the bar color
            r, g, b = mcolors.to_rgb(c)
            ax.vlines(p, lo, hi, color=(r * 0.55, g * 0.55, b * 0.55),
                      linewidth=0.7, zorder=4)
        top = hi if np.isfinite(hi) else v
        if c == COLOR_MISSING:
            ax.text(p, v + 0.08, "n/a", ha="center", va="bottom", fontsize=5,
                    color="0.5")
        else:
            ax.text(p, top + 0.08, f"{v:.1f}x", ha="center", va="bottom",
                    fontsize=5, color="0.2", fontweight="bold")

    ax.set_xticks(pos)
    ax.set_xticklabels([b[3] for b in bars], fontsize=5.5, ha="right", rotation=35)
    for center, label in zip(group_centers, group_labels):
        ax.text(center, -0.40, label, ha="center", va="top", fontsize=6.5,
                fontweight="bold", transform=ax.get_xaxis_transform())
    ax.set_ylabel(r"$S(\mathcal{E}^\star) = \tilde{B}_{\mathrm{HF}} \,/\, \tilde{B}_{\mathrm{MF}}$")
    ax.axhline(y=1.0, color="0.5", linewidth=0.4, linestyle="--", zorder=0)
    ax.grid(axis="y", alpha=0.3, linewidth=0.3, linestyle="--")
    ax.set_axisbelow(True)
    handles = [Patch(facecolor=COLOR_DISC, edgecolor="white",
                     label="Discretization error"),
               Patch(facecolor=COLOR_SOLVER, edgecolor="white",
                     label="Solver error")]
    ax.legend(handles=handles, loc="upper left", frameon=True, edgecolor="0.8",
              fancybox=False, fontsize=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.subplots_adjust(bottom=0.32, left=0.08, right=0.98, top=0.95)
    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"wrote {save_path}")


def main():
    args = parse_args()
    C.apply_style()
    data = C.load_experiments(args.results_dir)

    # main paper carries only the hyperelasticity condensed figure
    plot_condensed(data["hyperelasticity"], args.model,
                   f"{args.out_dir}/results_hyperelasticity_condensed.pdf")
    plot_condensed(C.load_poisson()["poisson3d"], "U-Net",
                   f"{args.out_dir}/results_poisson3d_condensed.pdf")
    if args.skip_savings:
        return

    data.update(C.load_poisson())  # poisson joins the savings chart
    rows = []
    for prob, dfs in data.items():
        for axis_label in ["Discretization error", "Solver error"]:
            df = dfs[axis_label]
            for model in C.POISSON_MODELS.get(prob, C.MODELS):
                s = C.compute_savings(df, args.threshold, model)
                lo, med, hi = (C.bootstrap_savings(df, args.threshold, model, args.n_boot)
                               if args.n_boot else (np.nan, s["savings_ratio"], np.nan))
                rows.append(dict(problem=prob, axis=axis_label, model=model,
                                 ratio=med, lo=lo, hi=hi,
                                 ratio_meancurve=s["savings_ratio"],
                                 b_tilde_hf=s["b_tilde_hf"], b_tilde_mf=s["b_tilde_mf"]))
                print(f"{prob:20s} {axis_label:22s} {model:10s} "
                      f"S(med)={med:.2f}x  CI=[{lo:.2f}, {hi:.2f}]  "
                      f"(mean-curve {s['savings_ratio']:.2f})")
    # airfoil savings (one disc-colored bar per arch, per-problem eps* = 1.5%);
    # uses the field-avg error, present in both field-only and coeff eval CSVs
    af_csv = f"{args.results_dir}/airfoil.csv"
    if osp.exists(af_csv):
        af = pd.read_csv(af_csv)
        af.loc[af.model_name == "ABUPT", "model_name"] = "AB-UPT"
        # savings on the VOLUME field error (field-avg never reaches 1.5%); coeff-eval
        # column name after --force re-eval, field-only name before it
        vcol = ("val/volume_rel_l2_loss_fields_avg"
                if "val/volume_rel_l2_loss_fields_avg" in af.columns
                else "val/rel_l2_volume")
        for model in C.MODELS:
            s = C.compute_savings(af, C.AIRFOIL_THRESHOLD, model, error_col=vcol)
            lo, med, hi = (C.bootstrap_savings(af, C.AIRFOIL_THRESHOLD, model,
                                               args.n_boot, error_col=vcol)
                           if args.n_boot else (np.nan, s["savings_ratio"], np.nan))
            rows.append(dict(problem="airfoil", axis="Discretization error",
                             model=model, ratio=med, lo=lo, hi=hi,
                             ratio_meancurve=s["savings_ratio"],
                             b_tilde_hf=s["b_tilde_hf"], b_tilde_mf=s["b_tilde_mf"]))
            print(f"{'airfoil':20s} {f'(eps*={C.AIRFOIL_THRESHOLD:.1%})':22s} {model:10s} "
                  f"S(med)={med:.2f}x  CI=[{lo:.2f}, {hi:.2f}]  "
                  f"(mean-curve {s['savings_ratio']:.2f})")

    pd.DataFrame(rows).to_csv(f"{args.out_dir}/savings_summary.csv", index=False)
    plot_savings(rows, f"{args.out_dir}/savings_summary.pdf", args.threshold)


if __name__ == "__main__":
    main()
