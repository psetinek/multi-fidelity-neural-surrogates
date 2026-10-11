"""Convergence of the proper multigrid V-cycle at the operating
resolution N = 64 per dimension.

The full `truncation.py` study sweeps all variants × all resolutions ×
all RHS modes; this script extracts the slice that matters for the
paper experiments (target_resolution=64 per dim, "proper" variant
only) and plots a representative subset of `num_modes`. Reads the CSV
already produced by `truncation.py`; does not re-run the experiments.

Layout: 1 × 3 panels (one per dim). Y-axis = relative L2 error vs the
direct DST solver at the same grid (log). X-axis = K, the number of
V-cycles. Curves: per-num_modes median across seeds with shaded IQR
band.

Styled like the convergence figures of the other experiments (see
`convergence_studies/style.py`); sized for a full `\\linewidth` include.

Output: convergence_studies/figures/mg_at_N64.{png,pdf}
"""

# ruff: noqa: E402
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from convergence_studies.style import (
    BAND_ALPHA,
    COLUMN_WIDTH,
    MULTI_PANEL_LINE,
    SERIES_MARKERS,
    apply_style,
    sequential_colors,
    style_grid,
)

# Per-dim representative `num_modes`. Match the same set used in the
# upstream sweep so the rows we want exist in the CSV. Plotting all 4
# is fine — they sit on a single colormap and give the reader a sense
# of how RHS smoothness affects convergence.
MODES_PER_DIM = {1: [5, 15, 25, 35], 2: [5, 15, 25, 35], 3: [2, 4, 8, 12]}

PACKAGE_DIR = Path(__file__).resolve().parent
CSV_PATH = PACKAGE_DIR / "results" / "truncation.csv"
OUT_DIR = PACKAGE_DIR / "figures"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df = df[(df["N"] == 64) & (df["mg_variant"] == "proper")].copy()
    if df.empty:
        raise SystemExit(
            f"no rows at N=64 / mg_variant=proper in {CSV_PATH}; "
            "re-run `python -m convergence_studies.truncation` first."
        )

    apply_style()

    fig, axs = plt.subplots(
        1,
        3,
        figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.3),
        sharey=True,
        constrained_layout=True,
    )

    for ax_idx, dim in enumerate([1, 2, 3]):
        ax = axs[ax_idx]
        modes = sorted(MODES_PER_DIM[dim])
        colors = sequential_colors(len(modes))
        for m_idx, m in enumerate(modes):
            sub = df[(df["dim"] == dim) & (df["num_modes"] == m)]
            if sub.empty:
                continue
            agg = (
                sub.groupby("K")["L_2_rel"]
                .agg(
                    med="median",
                    q25=lambda s: s.quantile(0.25),
                    q75=lambda s: s.quantile(0.75),
                )
                .reset_index()
                .sort_values("K")
            )
            color = colors[m_idx]
            ax.fill_between(
                agg["K"],
                agg["q25"],
                agg["q75"],
                alpha=BAND_ALPHA,
                color=color,
                linewidth=0,
            )
            ax.plot(
                agg["K"],
                agg["med"],
                marker=SERIES_MARKERS[m_idx],
                color=color,
                label=f"$M = {m}$",
                **MULTI_PANEL_LINE,
            )

        # Linear x-axis × log y-axis: exponential MG convergence reads
        # as straight lines, the classical convergence-rate visualisation.
        # ylim matches truncation_l2.png so the proper-MG slope reads at
        # the same scale as the original 9-panel sweep.
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1e1)
        ax.set_xlabel(r"V-cycles ($K$)")
        if ax_idx == 0:
            ax.set_ylabel(r"Rel. $L_2$ error")
        ax.set_title(f"{dim}D")
        style_grid(ax)
        ax.legend(loc="upper right")

        ax.set_xlim(0, 20)
        # Integer x-ticks rather than the auto float labels (0.0, 2.5, …).
        # Minor ticks every 1 K so it's visually unambiguous that x is linear.
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_xticks(range(0, 21), minor=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "mg_at_N64.png"
    out_pdf = OUT_DIR / "mg_at_N64.pdf"
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    print(f"  → {out_png}")
    print(f"  → {out_pdf}")

    # Quick numeric summary: per-dim per-decade contraction at the highest
    # num_modes (worst case for the smoother).
    print()
    print("Per-cycle contraction (median across seeds, hardest num_modes):")
    for dim in [1, 2, 3]:
        m = max(MODES_PER_DIM[dim])
        sub = df[(df["dim"] == dim) & (df["num_modes"] == m)]
        if sub.empty:
            continue
        per_K = sub.groupby("K")["L_2_rel"].median().sort_index()
        if len(per_K) >= 2:
            ratio = (per_K.iloc[-1] / per_K.iloc[0]) ** (
                1 / (np.log2(per_K.index[-1]) - np.log2(per_K.index[0]))
            )
            print(
                f"  dim={dim}, num_modes={m}: "
                f"{per_K.iloc[0]:.2e} → {per_K.iloc[-1]:.2e} over "
                f"K={int(per_K.index[0])} → {int(per_K.index[-1])}, "
                f"per-doubling-of-K factor ≈ {ratio:.2e}"
            )


if __name__ == "__main__":
    main()
