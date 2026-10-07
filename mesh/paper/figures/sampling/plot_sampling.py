"""Dataset-mixing figure.

Left: the exponential sampling distribution p(xi-hat; lambda) on a schematic uniform 11-level
menu for a range of lambda. Right: realized number of samples vs lambda on the linear-elasticity
solver dataset for the paper's budgets B~ = 100 ... 6400 (mean +- std over sampling repeats),
with a top axis mapping lambda to the expected fidelity E[xi-hat] on the real menu.

Outputs (this directory): combined_sampling.pdf, pmf_standalone.pdf (transparent, used in the
Figure 1 schematic). Needs the linear-elasticity solver dataset under data/.
Usage:  python plot_sampling.py
"""
import os.path as osp
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from mf_surrogates.data.plate import PlateData
from mf_surrogates.data.utils import get_exp_pmf

HERE = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(HERE, "..", "results"))
import common as C  # noqa: E402

REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
DATA = f"{REPO}/data/linear_elasticity/solver_truncation/processed"
C_HF = 262144.0
B_TILDES = [100, 200, 400, 800, 1600, 3200, 6400]  # paper budget ladder
LAM_BOUNDS = (-20.0, 20.0)
LAMBDAS_PMF = [-20, -4, 0, 4, 20]
N_REPEATS = 5
GOLDEN_RATIO = (1 + 5 ** 0.5) / 2


def navia_trunc():
    return mcolors.LinearSegmentedColormap.from_list(
        "navia_trunc", C.cmap_sequential(np.linspace(0, 0.9, 256)))


def plot_pmf_panel(ax, fig):
    x = np.linspace(0, 1, 11)
    cmap = navia_trunc()
    norm = Normalize(vmin=min(LAMBDAS_PMF), vmax=max(LAMBDAS_PMF))
    for lam in LAMBDAS_PMF:
        ax.plot(x, get_exp_pmf(x, lam), color=cmap(norm(lam)), alpha=0.8,
                linewidth=1, marker="o", markersize=1)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.9)
    cbar.set_label(r"$\lambda$", rotation=0, labelpad=6)
    ax.set_xlabel(r"$\hat{\xi}$")
    ax.set_ylabel(r"$p(\hat{\xi};\lambda, \Xi)$")
    ax.grid(True, which="major", linestyle="--", alpha=0.3, color="gray")
    ax.set_xlim(0, 1)


def main():
    C.apply_style()
    plt.rcParams.update({"lines.linewidth": 1.0, "lines.markersize": 1.0})

    dataset = PlateData(
        data_path=DATA, split="train", val_fraction=0.05, test_fraction=0.05,
        budget=10 * C_HF, linearization_col="solver_max_its",
        cost_column="solver_max_its", lambd=None, lambda_bounds=LAM_BOUNDS,
        n_samples_target=10, hf_only=False, q=0.99993,
        composition_rng=np.random.default_rng(seed=42))

    fig_w = C.COLUMN_WIDTH
    fig_h = C.COLUMN_WIDTH * 0.48 / GOLDEN_RATIO * 1.1
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(fig_w, fig_h),
                                            constrained_layout=True)
    plot_pmf_panel(ax_left, fig)

    # right panel: sample count vs lambda per ladder budget (B~ colorbar convention)
    cmap, norm_bt, sm_bt = C.btilde_colormap(B_TILDES)
    lam_range = np.linspace(*LAM_BOUNDS, 100)
    ax_top = ax_right.twiny()
    for bt in B_TILDES:
        dataset.budget = bt * C_HF
        color = cmap(norm_bt(bt))
        samples = {lam: [] for lam in lam_range}
        for _ in range(N_REPEATS):
            for lam in lam_range:
                try:
                    _, n_s, _ = dataset._create_sampling_plan(
                        lambd=lam, greedy_fill=True)
                    samples[lam].append(n_s)
                except Exception:
                    continue
        lams = np.array([l for l in lam_range if samples[l]])
        means = np.array([np.mean(samples[l]) for l in lams])
        stds = np.array([np.std(samples[l]) for l in lams])
        ax_right.plot(lams, means, color=color)
        ax_right.fill_between(lams, means - stds, means + stds, color=color,
                              alpha=0.3)
        print(f"B~={bt}: n ranges {means.min():.0f}..{means.max():.0f}")

    ax_right.set_xlabel(r"$\lambda$")
    ax_right.set_ylabel("Number of samples")
    ax_right.grid(True, alpha=0.3)
    ax_right.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax_right.ticklabel_format(axis="y", style="scientific", scilimits=(0, 0))
    ax_right.yaxis.get_offset_text().set_position((-0.15, 0))
    for spine in ("top", "right"):
        ax_right.spines[spine].set_visible(False)
    cbar = fig.colorbar(sm_bt, ax=ax_right, shrink=0.9)
    cbar.ax.set_ylabel(r"$\tilde{B}$", rotation=0)

    # top axis: lambda positions of target expected fidelities on the REAL menu
    fids = sorted(dataset.metadata_df["fidelity_linearized"].unique())
    # this menu's E[xi] transitions steeply around lambda=0, so 0.2/0.7 would collide
    # with their neighbors
    targets = [0.0, 0.1, 0.5, 0.9, 1.0]
    lam_search = np.linspace(*LAM_BOUNDS, 10000)
    expected = np.array([np.dot(get_exp_pmf(fids, l), fids) for l in lam_search])
    tick_pos = [lam_search[np.argmin(np.abs(expected - t))] for t in targets]
    ax_top.set_xlim(ax_right.get_xlim())
    ax_top.set_xticks(tick_pos)
    ax_top.set_xticklabels([f"{t:.1f}" for t in targets])
    ax_top.set_xlabel(r"$\mathbb{E}[\hat{\xi}]$")

    out = osp.join(HERE, "combined_sampling.pdf")
    fig.savefig(out, format="pdf")
    print(f"wrote {out}")

    # standalone transparent PMF panel (Fig. 1 schematic ingredient)
    fig2, ax2 = plt.subplots(
        figsize=(C.COLUMN_WIDTH * 0.48, C.COLUMN_WIDTH * 0.48 / GOLDEN_RATIO),
        constrained_layout=True)
    plot_pmf_panel(ax2, fig2)
    out2 = osp.join(HERE, "pmf_standalone.pdf")
    fig2.savefig(out2, format="pdf", transparent=True)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
