"""Airfoil result figures.

The airfoil experiment has one fidelity axis but four metrics per run, so this pairs metrics
instead of fidelity axes, reusing the statistics and style helpers of common.py:

  airfoil_volume_surface_overview.pdf   volume field error | surface field error
  airfoil_cl_cd_overview.pdf            lift C_l error | drag C_d error
  airfoil_condensed_<arch>.pdf          iso-budget sweep + Pareto front: volume fields | C_l

Metrics (columns of paper/results/airfoil.csv, written by evaluate.py):
  volume  = val/volume_rel_l2_loss_fields_avg
  surface = val/surface_rel_l2_loss_fields_avg
  C_l     = val/rel_err_cl_avg
  C_d     = val/rel_err_cd_avg

Usage:  python airfoil.py
"""
import argparse
import os.path as osp

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

import common as C

HERE = osp.dirname(osp.abspath(__file__))
AIRFOIL_CSV = f"{C.RESULTS_DIR}/airfoil.csv"

# (column, header, y-axis label)
METRICS = {
    "volume": ("val/volume_rel_l2_loss_fields_avg", "Volume fields", r"Rel. $L_2$ error"),
    "surface": ("val/surface_rel_l2_loss_fields_avg", "Surface fields", r"Rel. $L_2$ error"),
    "cl": ("val/rel_err_cl_avg", r"Lift coefficient ($C_l$)", "Rel. error"),
    "cd": ("val/rel_err_cd_avg", r"Drag coefficient ($C_d$)", "Rel. error"),
}


def load_airfoil(csv_path=AIRFOIL_CSV):
    df = pd.read_csv(csv_path)
    df.loc[df.model_name == "ABUPT", "model_name"] = "AB-UPT"
    return df


def _sweep_and_pareto(ax_top, ax_bot, df, model_name, metric_col, cmap, norm,
                      ylabel, show_ylabel, first_legend, show_yticks=None):
    """Top: error vs lambda per budget (MF). Bottom: HF-only vs MF(lambda*) Pareto."""
    sub = df[df.model_name == model_name]
    mf, hf = sub[~sub.hf_only], sub[sub.hf_only]

    for bt, stats in C.iso_budget_stats(mf, metric_col).items():
        color = cmap(norm(bt))
        ax_top.plot(stats.lambd, stats["mean"], color=color, linewidth=0.8)
        ax_top.fill_between(stats.lambd, stats["mean"] - stats["std"],
                            stats["mean"] + stats["std"], color=color, alpha=0.12)
        best = stats.loc[stats["mean"].idxmin()]
        ax_top.plot(best.lambd, best["mean"], marker="*", markersize=5, color=color,
                    markeredgecolor="white", markeredgewidth=0.2, zorder=10)
    ax_top.set_yscale("log")
    ax_top.set_title(model_name, pad=2)
    ax_top.grid(True, alpha=0.3, ls="--")
    ax_top.set_xlabel(r"$\lambda$", labelpad=0)

    hb, hm, hs = C.hf_curve(hf, metric_col)
    ob, om, os_ = C.mf_optimal_curve(mf, metric_col)
    if len(hb):
        ax_bot.plot(hb, hm, color=C.color_baseline, linestyle="--", marker="o",
                    markersize=2, linewidth=0.8,
                    label="HF only" if first_legend else None)
        ax_bot.fill_between(hb, hm - hs, hm + hs, color=C.color_baseline, alpha=0.12)
    if len(ob):
        ax_bot.plot(ob, om, color=C.color_ours, linestyle="-", marker="*",
                    markersize=5, linewidth=0.8, markeredgecolor="white",
                    markeredgewidth=0.2,
                    label=r"MF ($\lambda^{*}$)" if first_legend else None)
        ax_bot.fill_between(ob, om - os_, om + os_, color=C.color_ours, alpha=0.12)
    ax_bot.set_xscale("log")
    ax_bot.set_yscale("log")
    ax_bot.set_xlabel(r"$\tilde{B}$", labelpad=0)
    ax_bot.grid(True, alpha=0.3, ls="--")
    if show_yticks is None:
        show_yticks = show_ylabel
    if show_ylabel:
        ax_top.set_ylabel(ylabel)
        ax_bot.set_ylabel(ylabel)
    if not show_yticks:  # inner (shared-y) archs: drop tick labels
        for ax in (ax_top, ax_bot):
            ax.tick_params(axis="y", labelleft=False, left=False, which="both")


def plot_overview(df, metric_left, metric_right, save_path, models=None):
    """2 metrics (columns) x 2 archs, mirroring full_appendix.plot_overview."""
    models = models or C.MODELS
    cmap = C.navia_truncated()
    groups = [metric_left, metric_right]

    fig = plt.figure(figsize=(C.COLUMN_WIDTH, C.COLUMN_WIDTH * 0.45))
    outer = GridSpec(2, 2, figure=fig, width_ratios=[1, 1], height_ratios=[1, 1],
                     wspace=0.3, hspace=0.35, left=0.10, right=0.90, top=0.87,
                     bottom=0.10)
    bot_axes, group_sms = [], []

    for gi, metric in enumerate(groups):
        col, header, ylabel = METRICS[metric]
        bts = sorted(df[~df.hf_only].b_tilde.dropna().unique())
        norm = mcolors.LogNorm(vmin=min(bts), vmax=max(bts))
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        group_sms.append(sm)

        inner_top = GridSpecFromSubplotSpec(1, len(models), subplot_spec=outer[0, gi],
                                            wspace=0.1)
        inner_bot = GridSpecFromSubplotSpec(1, len(models), subplot_spec=outer[1, gi],
                                            wspace=0.1)
        top_left = bot_left = None
        for ai, model in enumerate(models):
            ax_top = (fig.add_subplot(inner_top[0, 0]) if ai == 0 else
                      fig.add_subplot(inner_top[0, ai], sharey=top_left))
            ax_bot = (fig.add_subplot(inner_bot[0, 0]) if ai == 0 else
                      fig.add_subplot(inner_bot[0, ai], sharey=bot_left))
            if ai == 0:
                top_left, bot_left = ax_top, ax_bot
            bot_axes.append(ax_bot)
            _sweep_and_pareto(ax_top, ax_bot, df, model, col, cmap, norm, ylabel,
                              show_ylabel=(ai == 0 and gi == 0),
                              show_yticks=(ai == 0),  # first arch of each group
                              first_legend=(ai == 0 and gi == 0))

    for ax in [a for a in fig.axes if a.get_yscale() == "log"]:
        C.decade_extreme_ticks(ax)
    for ax in bot_axes:
        C.decade_xticks(ax)

    for gi in range(2):
        top_pos = outer[0, gi].get_position(fig)
        cax = fig.add_axes([top_pos.x1 + 0.004, top_pos.y0, 0.005,
                            top_pos.y1 - top_pos.y0])
        cbar = fig.colorbar(group_sms[gi], cax=cax)
        cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], which="both",
                            pad=0.8)
        cbar.ax.set_title(r"$\tilde{B}$", fontsize=plt.rcParams["axes.labelsize"],
                          pad=1.5)

    handles, labels = bot_axes[0].get_legend_handles_labels()
    if handles:
        bot_axes[-1].legend(handles, labels, loc="center left",
                            bbox_to_anchor=(0.97, 0.5), frameon=True, edgecolor="0.8",
                            fancybox=False)

    for gi, metric in enumerate(groups):
        pos = outer[0, gi].get_position(fig)
        fig.text((pos.x0 + pos.x1) / 2, 0.93, METRICS[metric][1], ha="center",
                 va="bottom", fontsize=7, fontweight="bold")

    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"wrote {save_path}")


def plot_condensed(df, model_name, metric_left, metric_right, save_path):
    """One arch, two metric panel-pairs (iso-budget sweep + Pareto), mirroring
    condensed.plot_condensed."""
    groups = [metric_left, metric_right]
    all_bt = sorted(df[~df.hf_only].b_tilde.dropna().unique())
    cmap, norm, sm = C.btilde_colormap(all_bt)

    fig = plt.figure(figsize=(C.COLUMN_WIDTH, C.COLUMN_WIDTH * 0.31))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1, 1], wspace=0.28,
                     left=0.07, right=0.995, top=0.78, bottom=0.18)

    for gi, metric in enumerate(groups):
        col, header, ylabel = METRICS[metric]
        inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, gi],
                                        width_ratios=[1, 1], wspace=0.12)
        sub = df[df.model_name == model_name]
        mf, hf = sub[~sub.hf_only], sub[sub.hf_only]

        ax_l = fig.add_subplot(inner[0, 0])
        for bt, stats in C.iso_budget_stats(mf, col).items():
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
        ax_l.set_ylabel(ylabel, labelpad=2)

        ax_r = fig.add_subplot(inner[0, 1], sharey=ax_l)
        hb, hm, hs = C.hf_curve(hf, col)
        ax_r.plot(hb, hm, color=C.color_baseline, linestyle="--", marker="o",
                  markersize=2, linewidth=0.8, label="HF only")
        ax_r.fill_between(hb, hm - hs, hm + hs, color=C.color_baseline, alpha=0.12)
        ob, om, os_ = C.mf_optimal_curve(mf, col)
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

        pl, pr = ax_l.get_position(fig), ax_r.get_position(fig)
        span = pr.x1 - pl.x0
        cb_w = span * 0.62
        cax = fig.add_axes([pl.x0 + (span - cb_w) / 2, pl.y1 + 0.035, cb_w, 0.032])
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.ax.xaxis.set_ticks_position("top")
        cbar.ax.tick_params(labelsize=plt.rcParams["ytick.labelsize"], which="both",
                            pad=1)
        cax.text(1.04, 0.4, r"$\tilde{B}$", transform=cax.transAxes, va="center",
                 ha="left", fontsize=plt.rcParams["axes.labelsize"])
        fig.text((pl.x0 + pr.x1) / 2, 0.995, header, ha="center", va="top",
                 fontsize=7, fontweight="bold")

    plt.savefig(save_path, format="pdf")
    plt.close(fig)
    print(f"wrote {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-o", "--out-dir", default=HERE)
    args = p.parse_args()
    C.apply_style()
    df = load_airfoil()

    plot_overview(df, "volume", "surface",
                  f"{args.out_dir}/airfoil_volume_surface_overview.pdf")
    plot_overview(df, "cl", "cd", f"{args.out_dir}/airfoil_cl_cd_overview.pdf")
    for model in C.MODELS:
        plot_condensed(df, model, "volume", "cl",
                       f"{args.out_dir}/airfoil_condensed_{model}.pdf")


if __name__ == "__main__":
    main()
