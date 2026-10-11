"""Iterative truncation error vs. V-cycle count, resolution, and RHS.

Fix the discretization (solve_direct on the SAME grid is the truth),
vary K = number of V-cycles in the MG iterator. For a well-tuned V-cycle
the convergence rate should be h-independent: doubling N should not slow
convergence. `num_modes` stresses the coarse-grid correction differently
(smoother RHS → easier to represent on coarse grids).

Three MG variants are swept:
- "proper": full coarsening, nu1=nu2=2, coarse_sweeps=50. One cycle is
  essentially a direct solve at these problem sizes.
- "medium": num_levels=2, nu1=nu2=1, coarse_sweeps=15. Same 2-level
  structure as shallow but with a better coarse solve; lands around
  ~1e-3 relative L2 at K=32 in 2D. 1D undershoots (GS alone is slow on
  tridiagonal Poisson, so even a better CGC doesn't help much); 3D
  overshoots (small grids + low modes make the coarse problem trivial).
- "shallow": num_levels=2, nu1=nu2=1, coarse_sweeps=5. Deliberately
  under-resolves the coarse-grid correction so the per-cycle contraction
  is ~0.9 and 32 cycles only reach ~1e-1.

Plot is a 3×3 grid: rows are MG variant, columns are dim. Curves are
log(error) vs K, one per (N, M): colour is N, marker is M. All at fp64 so
only iterative truncation contributes.

`--plot-only` redraws the figures from the CSV of a previous run.
"""

# Imports are split by jax.config.update() so x64 is enabled before
# poisson_nd modules pull in jnp; ruff's E402 doesn't apply here.
# ruff: noqa: E402

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

from convergence_studies.style import (
    COLOR_BASELINE_2,
    COLUMN_WIDTH,
    MULTI_PANEL_LINE,
    SERIES_MARKERS,
    apply_style,
    sequential_colors,
    style_grid,
)
from convergence_studies.utils import (
    RESULTS_DIR,
    l2_rel,
    l_inf,
    l_inf_rel,
    residual_linf,
    save_csv,
    save_figure,
    seed_keys,
)
from poisson_nd.dst import solve_direct
from poisson_nd.multigrid import build_multigrid_iterator
from poisson_nd.rhs import generate_rhs

RESOLUTIONS = {1: [32, 64, 128], 2: [32, 64, 128], 3: [16, 32, 64]}
MODES = {1: [5, 15, 25, 35], 2: [5, 15, 25, 35], 3: [2, 4, 8, 12]}
MG_VARIANTS = {
    "proper": {},
    "medium": {"num_levels": 2, "nu1": 1, "nu2": 1, "coarse_sweeps": 15},
    "shallow": {"num_levels": 2, "nu1": 1, "nu2": 1, "coarse_sweeps": 5},
}
K_VALUES = [1, 2, 4, 8, 16, 32]
N_SEEDS = 3
DOMAIN_EXTENT = 1.0


def run():
    # Clear JIT caches after each (variant, dim, N) shape iteration. Each
    # (shape, num_modes) triggers a fresh generate_rhs compile, and each shape
    # triggers fresh solve_direct and iterator compiles; left uncleared, the
    # accumulated ~30+ XLA programs exhaust LLVM's section-allocator on this
    # machine (OOM at the ~(dim=2, N=64, M=35) mark). Clearing after each
    # shape keeps the live JIT working set small. Execution still dominates
    # runtime, so the recompile cost is minor.
    rows = []
    for variant, kwargs in MG_VARIANTS.items():
        for dim in RESOLUTIONS:
            for N in RESOLUTIONS[dim]:
                shape = (N,) * dim
                h = DOMAIN_EXTENT / (N + 1)
                iterator = build_multigrid_iterator(shape, **kwargs)
                for num_modes in MODES[dim]:
                    for seed_idx, key in enumerate(seed_keys(100, N_SEEDS)):
                        print(
                            f"  variant={variant} dim={dim} N={N} "
                            f"num_modes={num_modes} seed={seed_idx}"
                        )
                        rhs = generate_rhs(
                            key,
                            shape,
                            domain_extent=DOMAIN_EXTENT,
                            num_modes=num_modes,
                        )
                        u_h = solve_direct(rhs, h)
                        for K in K_VALUES:
                            u_k = iterator(rhs, h, K)
                            rows.append(
                                {
                                    "dim": dim,
                                    "N": N,
                                    "num_modes": num_modes,
                                    "seed": seed_idx,
                                    "mg_variant": variant,
                                    "K": K,
                                    "L_inf": l_inf(u_k, u_h),
                                    "L_inf_rel": l_inf_rel(u_k, u_h),
                                    "L_2_rel": l2_rel(u_k, u_h),
                                    "residual_L_inf": residual_linf(u_k, rhs, h),
                                }
                            )
                del iterator
                jax.clear_caches()

    df = pd.DataFrame(rows)
    save_csv(df, "truncation")
    plot_all(df)
    return df


def plot_all(df):
    plot(
        df,
        metric="L_inf_rel",
        ylabel=r"Rel. $L_\infty$ error",
        out_name="truncation",
    )
    plot(
        df,
        metric="L_2_rel",
        ylabel=r"Rel. $L_2$ error",
        out_name="truncation_l2",
    )


def plot(df, *, metric, ylabel, out_name):
    """Mean over seeds; error is measured against the direct solve on the
    same grid."""
    apply_style()
    variants = list(MG_VARIANTS.keys())
    fig, axes = plt.subplots(
        len(variants),
        3,
        figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.27 * len(variants)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, variant in enumerate(variants):
        for col_idx, dim in enumerate([1, 2, 3]):
            ax = axes[row_idx, col_idx]
            ax.set_title(f"{dim}D, {variant} V-cycle")
            ax.set_yscale("log")
            ax.set_ylim(1e-5, 1e1)
            ax.set_xticks([K for K in K_VALUES if K != 2])  # 2 would crowd 1
            style_grid(ax)
            if row_idx == len(variants) - 1:
                ax.set_xlabel(r"V-cycles ($K$)")
            if col_idx == 0:
                ax.set_ylabel(ylabel)

            sub = df[(df["dim"] == dim) & (df["mg_variant"] == variant)]
            if sub.empty:
                continue

            resolutions = sorted(sub["N"].unique())
            modes = sorted(sub["num_modes"].unique())
            colors = sequential_colors(len(resolutions))
            for N, color in zip(resolutions, colors):
                for num_modes, marker in zip(modes, SERIES_MARKERS):
                    curve = (
                        sub[(sub["N"] == N) & (sub["num_modes"] == num_modes)]
                        .groupby("K")[metric]
                        .mean()
                        .reset_index()
                        .sort_values("K")
                    )
                    ax.plot(
                        curve["K"],
                        curve[metric],
                        marker=marker,
                        color=color,
                        **MULTI_PANEL_LINE,
                    )

            # N and M are the same down a column, so one legend per column
            # is enough; the top row ("proper") leaves its upper right free.
            if row_idx == 0:
                handles = [
                    Line2D([], [], color=color, label=f"$N = {N}$", **MULTI_PANEL_LINE)
                    for N, color in zip(resolutions, colors)
                ]
                # Blank entries so that M starts in the second column.
                handles += [Line2D([], [], linestyle="none", label="")] * (
                    len(modes) - len(resolutions)
                )
                handles += [
                    Line2D(
                        [],
                        [],
                        linestyle="none",
                        marker=marker,
                        color=COLOR_BASELINE_2,
                        label=f"$M = {num_modes}$",
                        markersize=MULTI_PANEL_LINE["markersize"],
                    )
                    for num_modes, marker in zip(modes, SERIES_MARKERS)
                ]
                ax.legend(
                    handles=handles,
                    loc="upper right",
                    ncol=2,
                    columnspacing=1.0,
                    handlelength=1.5,
                )
    save_figure(fig, out_name)


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot_all(pd.read_csv(RESULTS_DIR / "truncation.csv"))
    else:
        run()
