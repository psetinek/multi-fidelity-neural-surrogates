"""Multi-fidelity (continuous menu) vs bi-fidelity comparison.

One panel per (fidelity axis, B~): error of the continuous menu vs lambda (bottom axis) and of
the cost-matched bi-fidelity runs vs their HF ratio (top axis), mean +- std over seeds, plus
the HF-only reference band.

Inputs (paper/results):
  bi-fidelity: hyperelasticity_{solver,discretization}_binary.csv
  continuous:  hyperelasticity_{solver_truncation,discretization}.csv
Outputs: mf_vs_bf_<axis>_b<B~>.pdf (this directory)
Usage:   python plot_mf_vs_bf.py [--model Transolver] [--log-y]
"""
import argparse
import os.path as osp
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import color_baseline, color_ours  # noqa: E402

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "results"))
import common as C  # noqa: E402

RESULTS = osp.abspath(osp.join(HERE, "..", "..", "results"))
ERROR_COLUMN = "val/rel_l2_loss_fields_avg"
GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

SETTINGS = [
    dict(axis="solver", b_tilde=200,
         binary_csv=f"{RESULTS}/hyperelasticity_solver_binary.csv",
         continuous_csv=f"{RESULTS}/hyperelasticity_solver_truncation.csv"),
    dict(axis="solver", b_tilde=1600,
         binary_csv=f"{RESULTS}/hyperelasticity_solver_binary.csv",
         continuous_csv=f"{RESULTS}/hyperelasticity_solver_truncation.csv"),
    dict(axis="discretization", b_tilde=200,
         binary_csv=f"{RESULTS}/hyperelasticity_discretization_binary.csv",
         continuous_csv=f"{RESULTS}/hyperelasticity_discretization.csv",
         figsize=(1.8, 1.8)),  # square panel for the discretization axis
]


def parse_args():
    p = argparse.ArgumentParser(description="MF vs binary-fidelity figures.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", default="Transolver")
    p.add_argument("--log-y", action="store_true", help="log-scale y axis")
    p.add_argument("-o", "--out-dir", default=HERE)
    return p.parse_args()


def plot_binary_vs_continuous(cont, binr, hf_only, save_path, log_y=False,
                              figsize=(C.COLUMN_WIDTH * 0.48,
                                       C.COLUMN_WIDTH * 0.48 / GOLDEN_RATIO)):
    """cont: aggregated over lambd; binr: aggregated over hf_ratio; hf_only: raw rows."""
    fig, ax_bot = plt.subplots(figsize=figsize, constrained_layout=True)
    ax_top = ax_bot.twiny()

    ax_bot.plot(cont.lambd, cont["mean"], color=color_ours, marker="o",
                label="Spectrum", zorder=4)
    ax_bot.fill_between(cont.lambd, cont["mean"] - cont["std"],
                        cont["mean"] + cont["std"], color=color_ours, alpha=0.18,
                        linewidth=0, zorder=3)

    ax_top.plot(binr.hf_ratio, binr["mean"], color="black", marker="s",
                linestyle="--", label="Binary", zorder=4)
    ax_top.fill_between(binr.hf_ratio, binr["mean"] - binr["std"],
                        binr["mean"] + binr["std"], color="black", alpha=0.18,
                        linewidth=0, zorder=3)

    hf_mean, hf_std = hf_only[ERROR_COLUMN].mean(), hf_only[ERROR_COLUMN].std()
    ax_bot.axhline(hf_mean, color=color_baseline, linestyle="--", linewidth=1,
                   label="HF only", zorder=5)
    ax_bot.axhspan(hf_mean - hf_std, hf_mean + hf_std, color=color_baseline,
                   alpha=0.12, linewidth=0, zorder=2)

    if log_y:
        ax_bot.set_yscale("log")

    ax_bot.set_xlabel(r"$\lambda$ (spectrum)", color=color_ours)
    ax_bot.tick_params(axis="x", colors=color_ours)
    ax_bot.spines["bottom"].set_color(color_ours)
    ax_top.set_xlabel(r"HF ratio (binary)", color="black")
    ax_top.tick_params(axis="x", colors="black")
    ax_top.spines["top"].set_color("black")
    ax_bot.set_ylabel(r"Relative $L_2$ error")
    ax_bot.grid(True)
    ax_bot.spines["top"].set_visible(False)
    for spine in ("bottom", "left", "right"):
        ax_top.spines[spine].set_visible(False)

    h1, l1 = ax_bot.get_legend_handles_labels()
    h2, l2 = ax_top.get_legend_handles_labels()
    ax_bot.legend(h1 + h2, l1 + l2, loc="lower right", borderpad=0.3,
                  handlelength=1.5)

    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {save_path}")


def main():
    args = parse_args()
    C.apply_style()
    # slightly heavier lines/markers than the overview figures
    plt.rcParams.update({"lines.linewidth": 1.0, "lines.markersize": 1.0})

    for s in SETTINGS:
        bdf = pd.read_csv(s["binary_csv"])
        bdf = bdf[(bdf.model_name == args.model) & (bdf.b_tilde == s["b_tilde"])]
        bdf["hf_ratio"] = bdf.n_hf / bdf.n_samples_realized
        binr = (bdf.groupby("hf_ratio")[ERROR_COLUMN].agg(["mean", "std", "count"])
                .reset_index().sort_values("hf_ratio").reset_index(drop=True))

        cdf = pd.read_csv(s["continuous_csv"])
        cdf = cdf[(cdf.model_name == args.model) & (cdf.b_tilde == s["b_tilde"])]
        hf_only, cdf = cdf[cdf.hf_only], cdf[~cdf.hf_only]
        cont = (cdf.groupby("lambd")[ERROR_COLUMN].agg(["mean", "std", "count"])
                .reset_index().sort_values("lambd").reset_index(drop=True))

        tag = f"{s['axis']}_b{s['b_tilde']}"
        n_min = int(min(binr["count"].min(), cont["count"].min()))
        print(f"{tag}: spectrum {len(cont)} cells, binary {len(binr)} cells, "
              f"min seeds/cell {n_min}"
              + (" (INCOMPLETE cells present)" if n_min < 3 else ""))
        print(f"  min spectrum {cont['mean'].min():.4f} | min binary "
              f"{binr['mean'].min():.4f} | HF-only {hf_only[ERROR_COLUMN].mean():.4f}")
        kwargs = {"figsize": s["figsize"]} if "figsize" in s else {}
        plot_binary_vs_continuous(cont, binr, hf_only, log_y=args.log_y,
                                  save_path=f"{args.out_dir}/mf_vs_bf_{tag}.pdf",
                                  **kwargs)


if __name__ == "__main__":
    main()
