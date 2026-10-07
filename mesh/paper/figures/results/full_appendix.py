"""Full result figures of the appendix.

One overview per problem with all budgets, lambdas, architectures and both fidelity axes, and
one Poisson figure per fidelity axis.

Outputs (this directory): <problem>_overview.pdf, Poisson figures
Usage:  python full_appendix.py
"""
import argparse
import os.path as osp

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import common as C

HERE = osp.dirname(osp.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="Full appendix overview figures.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results-dir", default=C.RESULTS_DIR)
    p.add_argument("-o", "--out-dir", default=HERE)
    return p.parse_args()


def plot_overview(dfs_by_axis, save_path, models=None, error_col=C.ERROR_COL):
    models = models or C.MODELS
    navia_trunc = C.navia_truncated()
    axes_labels = ["Discretization", "Solver"]
    axes_dfs = [dfs_by_axis["Discretization error"], dfs_by_axis["Solver error"]]

    fig = plt.figure(figsize=(C.COLUMN_WIDTH, C.COLUMN_WIDTH * 0.45))
    outer_gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1],
                        wspace=0.3, hspace=0.35,
                        left=0.10, right=0.90, top=0.90, bottom=0.13)

    bot_axes = []
    group_cmaps = []

    for axis_idx, df_exp in enumerate(axes_dfs):
        grp_btildes = sorted(df_exp["b_tilde"].dropna().unique())
        grp_norm = mcolors.LogNorm(vmin=min(grp_btildes), vmax=max(grp_btildes))
        grp_sm = cm.ScalarMappable(cmap=navia_trunc, norm=grp_norm)
        grp_sm.set_array([])
        group_cmaps.append((grp_sm, grp_norm))

        inner_top = GridSpecFromSubplotSpec(1, len(models),
                                            subplot_spec=outer_gs[0, axis_idx], wspace=0.1)
        inner_bot = GridSpecFromSubplotSpec(1, len(models),
                                            subplot_spec=outer_gs[1, axis_idx], wspace=0.1)

        for arch_idx, model_name in enumerate(models):
            subset = df_exp[df_exp["model_name"] == model_name].copy()

            # ---- Row 0: error vs lambda (MF runs only) ----
            if arch_idx == 0:
                ax_top = fig.add_subplot(inner_top[0, 0])
                ax_top_left = ax_top
            else:
                ax_top = fig.add_subplot(inner_top[0, arch_idx], sharey=ax_top_left)

            subset_mf = subset[~subset["hf_only"]]
            for btilde in sorted(subset_mf["b_tilde"].unique()):
                color = navia_trunc(grp_norm(btilde))
                bsub = subset_mf[subset_mf["b_tilde"] == btilde]
                stats = (bsub.groupby("lambd")[error_col].agg(["mean", "std"])
                         .reset_index().sort_values("lambd"))
                ax_top.plot(stats["lambd"], stats["mean"], color=color, linewidth=0.8)
                ax_top.fill_between(stats["lambd"], stats["mean"] - stats["std"],
                                    stats["mean"] + stats["std"], color=color, alpha=0.12)
                best_row = stats.loc[stats["mean"].idxmin()]
                ax_top.plot(best_row["lambd"], best_row["mean"], marker="*", markersize=5,
                            color=color, markeredgecolor="white", markeredgewidth=0.2,
                            zorder=10)

            ax_top.set_yscale("log")
            ax_top.set_title(model_name, pad=2)
            ax_top.grid(True, alpha=0.3, ls="--")
            ax_top.set_xlabel(r"$\lambda$", labelpad=0)
            if arch_idx == 0 and axis_idx == 0:
                ax_top.set_ylabel("Rel. $L_2$ error")
            elif arch_idx == 0:
                ax_top.set_ylabel("")
            else:
                ax_top.tick_params(axis="y", labelleft=False, left=False, which="both")

            # ---- Row 1: Pareto front ----
            if arch_idx == 0:
                ax_bot = fig.add_subplot(inner_bot[0, 0])
                ax_bot_left = ax_bot
            else:
                ax_bot = fig.add_subplot(inner_bot[0, arch_idx], sharey=ax_bot_left)
            bot_axes.append(ax_bot)

            subset_hf = subset[subset["hf_only"]]
            opt_bt, opt_mean, opt_std = C.mf_optimal_curve(subset_mf, error_col)
            hf_bt, hf_mean, hf_std = C.hf_curve(subset_hf, error_col)
            is_first = (axis_idx == 0 and arch_idx == 0)

            if len(hf_bt):
                ax_bot.plot(hf_bt, hf_mean, color=C.color_baseline, linestyle="--",
                            marker="o", markersize=2, linewidth=0.8,
                            label="HF only" if is_first else None)
                ax_bot.fill_between(hf_bt, hf_mean - hf_std, hf_mean + hf_std,
                                    color=C.color_baseline, alpha=0.12)
            if len(opt_bt):
                ax_bot.plot(opt_bt, opt_mean, color=C.color_ours, linestyle="-",
                            marker="*", markersize=5, linewidth=0.8,
                            markeredgecolor="white", markeredgewidth=0.2,
                            label=r"MF ($\lambda^{*}$)" if is_first else None)
                ax_bot.fill_between(opt_bt, opt_mean - opt_std, opt_mean + opt_std,
                                    color=C.color_ours, alpha=0.12)

            ax_bot.set_xscale("log")
            ax_bot.set_yscale("log")
            ax_bot.set_xlabel(r"$\tilde{B}$", labelpad=0)
            ax_bot.grid(True, alpha=0.3, ls="--")
            if arch_idx == 0 and axis_idx == 0:
                ax_bot.set_ylabel("Rel. $L_2$ error")
            elif arch_idx == 0:
                ax_bot.set_ylabel("")
            else:
                ax_bot.tick_params(axis="y", labelleft=False, left=False, which="both")

    # ---- Y ticks: decades + visible extremes; X: all decades ----
    for ax in [a for a in fig.axes if a.get_yscale() == "log"]:
        C.decade_extreme_ticks(ax)
    for ax in bot_axes:
        C.decade_xticks(ax)

    # ---- Per-group colorbars next to top row ----
    for axis_idx in range(len(axes_labels)):
        grp_sm, _ = group_cmaps[axis_idx]
        top_pos = outer_gs[0, axis_idx].get_position(fig)
        cbar_ax = fig.add_axes([top_pos.x1 + 0.004, top_pos.y0, 0.005,
                                top_pos.y1 - top_pos.y0])
        cbar = fig.colorbar(grp_sm, cax=cbar_ax)
        cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], which="both", pad=0.8)
        cbar.ax.set_title(r"$\tilde{B}$", fontsize=plt.rcParams["axes.labelsize"], pad=1.5)

    # ---- Legend (bottom-right) ----
    handles, labels = bot_axes[0].get_legend_handles_labels()
    if handles:
        bot_axes[-1].legend(handles, labels, loc="center left",
                            bbox_to_anchor=(0.97, 0.5), frameon=True,
                            edgecolor="0.8", fancybox=False)

    # ---- Axis-group labels ----
    for axis_idx, axis_label in enumerate(axes_labels):
        pos = outer_gs[0, axis_idx].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, 0.96, f"{axis_label} error",
                 ha="center", va="bottom", fontsize=7, fontweight="bold")

    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_poisson_axis(poisson_data, axis_label, save_path, error_col=C.ERROR_COL):
    """Poisson layout: one figure per fidelity axis, panels grouped by
    dim (1D U-Net | 1D FNO | 2D U-Net | 2D FNO | 3D U-Net), shared colorbar."""
    navia_trunc = C.navia_truncated()
    dims = ["poisson1d", "poisson2d", "poisson3d"]

    fig = plt.figure(figsize=(C.COLUMN_WIDTH, C.COLUMN_WIDTH * 0.55))
    outer_gs = GridSpec(2, 3, figure=fig, width_ratios=[2, 2, 1],
                        height_ratios=[1, 1], wspace=0.42, hspace=0.35,
                        left=0.10, right=0.87, top=0.94, bottom=0.12)

    all_bt = sorted(set(np.concatenate(
        [poisson_data[p][axis_label][~poisson_data[p][axis_label].hf_only].b_tilde.unique()
         for p in dims])))
    norm = mcolors.LogNorm(vmin=min(all_bt), vmax=max(all_bt))
    sm = cm.ScalarMappable(cmap=navia_trunc, norm=norm)
    sm.set_array([])

    bot_axes = []
    for dim_idx, prob in enumerate(dims):
        df_exp = poisson_data[prob][axis_label]
        archs = C.POISSON_MODELS[prob]
        inner_top = GridSpecFromSubplotSpec(1, len(archs),
                                            subplot_spec=outer_gs[0, dim_idx], wspace=0.1)
        inner_bot = GridSpecFromSubplotSpec(1, len(archs),
                                            subplot_spec=outer_gs[1, dim_idx], wspace=0.1)

        for arch_idx, model_name in enumerate(archs):
            subset = df_exp[df_exp["model_name"] == model_name]
            subset_mf = subset[~subset["hf_only"]]
            subset_hf = subset[subset["hf_only"]]

            if arch_idx == 0:
                ax_top = fig.add_subplot(inner_top[0, 0])
                ax_top_left = ax_top
            else:
                ax_top = fig.add_subplot(inner_top[0, arch_idx], sharey=ax_top_left)
            top_last = ax_top

            for btilde in sorted(subset_mf["b_tilde"].unique()):
                color = navia_trunc(norm(btilde))
                bsub = subset_mf[subset_mf["b_tilde"] == btilde]
                stats = (bsub.groupby("lambd")[error_col].agg(["mean", "std"])
                         .reset_index().sort_values("lambd"))
                ax_top.plot(stats["lambd"], stats["mean"], color=color, linewidth=0.8)
                ax_top.fill_between(stats["lambd"], stats["mean"] - stats["std"],
                                    stats["mean"] + stats["std"], color=color, alpha=0.12)
                best_row = stats.loc[stats["mean"].idxmin()]
                ax_top.plot(best_row["lambd"], best_row["mean"], marker="*", markersize=5,
                            color=color, markeredgecolor="white", markeredgewidth=0.2,
                            zorder=10)
            ax_top.set_yscale("log")
            ax_top.set_title(f"{prob[-2].upper()}D {model_name}")
            ax_top.grid(True, alpha=0.3, ls="--")
            ax_top.set_xlabel(r"$\lambda$")

            if arch_idx == 0:
                ax_bot = fig.add_subplot(inner_bot[0, 0])
                ax_bot_left = ax_bot
            else:
                ax_bot = fig.add_subplot(inner_bot[0, arch_idx], sharey=ax_bot_left)
            bot_axes.append(ax_bot)

            opt_bt, opt_mean, opt_std = C.mf_optimal_curve(subset_mf, error_col)
            hf_bt, hf_mean, hf_std = C.hf_curve(subset_hf, error_col)
            is_first = (dim_idx == 0 and arch_idx == 0)
            ax_bot.plot(hf_bt, hf_mean, color=C.color_baseline, linestyle="--", marker="o",
                        markersize=2, linewidth=0.8, label="HF only" if is_first else None)
            ax_bot.fill_between(hf_bt, hf_mean - hf_std, hf_mean + hf_std,
                                color=C.color_baseline, alpha=0.12)
            ax_bot.plot(opt_bt, opt_mean, color=C.color_ours, linestyle="-", marker="*",
                        markersize=5, linewidth=0.8, markeredgecolor="white",
                        markeredgewidth=0.2,
                        label=r"MF (opt. $\lambda$)" if is_first else None)
            ax_bot.fill_between(opt_bt, opt_mean - opt_std, opt_mean + opt_std,
                                color=C.color_ours, alpha=0.12)
            ax_bot.set_xscale("log")
            ax_bot.set_yscale("log")
            ax_bot.set_xlabel(r"$\tilde{B}$")
            ax_bot.grid(True, alpha=0.3, ls="--")

            for ax in (ax_top, ax_bot):
                if arch_idx == 0 and dim_idx == 0:
                    ax.set_ylabel("Rel. $L_2$ error")
                elif arch_idx == 0:
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=True)
                else:
                    ax.tick_params(axis="y", labelleft=False, left=False, which="both")

    for ax in [a for a in fig.axes if a.get_yscale() == "log"]:
        C.decade_extreme_ticks(ax)
    for ax in bot_axes:
        C.decade_xticks(ax)

    pos = top_last.get_position(fig)
    cbar_ax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.006, pos.y1 - pos.y0])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], which="both", pad=0.8)
    cbar.ax.set_title(r"$\tilde{B}$", fontsize=plt.rcParams["axes.labelsize"], pad=1.5)

    handles, labels = bot_axes[0].get_legend_handles_labels()
    if handles:
        bot_axes[-1].legend(handles, labels, loc="center left",
                            bbox_to_anchor=(0.98, 0.5), frameon=True,
                            edgecolor="0.8", fancybox=False)
    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"wrote {save_path}")


def main():
    args = parse_args()
    C.apply_style()
    data = C.load_experiments(args.results_dir)
    for prob, dfs in data.items():
        plot_overview(dfs, f"{args.out_dir}/{prob}_overview.pdf")
    poisson = C.load_poisson()
    plot_poisson_axis(poisson, "Discretization error",
                      f"{args.out_dir}/poisson_discretization_results.pdf")
    plot_poisson_axis(poisson, "Solver error",
                      f"{args.out_dir}/poisson_truncation_results.pdf")


if __name__ == "__main__":
    main()
