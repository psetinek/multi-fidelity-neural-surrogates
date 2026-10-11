"""Spatial discretization error vs. resolution, RHS smoothness, and dim.

Fix the solver (DST direct, fp64) and compare the solution on a coarse grid
to an ultra-fine-grid reference (also fp64, refinement factor ~4), so the
only remaining error source is the discretization itself.

Plot is log-log |u_h - u_ref| vs h per dim, one curve per `num_modes`,
with an O(h^2) guideline. Higher `num_modes` increases high-frequency
content in the RHS and should inflate the constant in front of h^2, but
the slope must remain -2 until the highest mode approaches the Nyquist.

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
from matplotlib.ticker import NullLocator, ScalarFormatter

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
    l2,
    l2_rel,
    l_inf,
    l_inf_rel,
    make_problem,
    save_csv,
    save_figure,
    seed_keys,
)
from poisson_nd.dst import solve_direct

RESOLUTIONS = {
    1: [8, 16, 32, 64, 128, 256, 512],
    2: [8, 16, 32, 64, 128],
    3: [4, 8, 16, 32],
}
N_REF = {1: 2048, 2: 512, 3: 96}
MODES = {1: [1, 3, 8, 15, 25], 2: [1, 3, 8, 15, 25], 3: [1, 2, 4, 8]}
N_SEEDS = 3
DOMAIN_EXTENT = 1.0


def run():
    rows = []
    for dim in RESOLUTIONS:
        shape_ref = (N_REF[dim],) * dim
        for num_modes in MODES[dim]:
            for seed_idx, key in enumerate(seed_keys(0, N_SEEDS)):
                print(
                    f"  dim={dim} num_modes={num_modes} seed={seed_idx} "
                    f"(N_ref={N_REF[dim]})"
                )
                problem = make_problem(key, num_modes, dim, shape_ref, DOMAIN_EXTENT)
                for N in RESOLUTIONS[dim]:
                    shape = (N,) * dim
                    h = problem.h_at(shape)
                    rhs = problem.rhs_at(shape)
                    u_h = solve_direct(rhs, h)
                    u_true = problem.u_true_at(shape)

                    rows.append(
                        {
                            "dim": dim,
                            "N": N,
                            "h": h,
                            "num_modes": num_modes,
                            "seed": seed_idx,
                            "L_inf": l_inf(u_h, u_true),
                            "L_inf_rel": l_inf_rel(u_h, u_true),
                            "L_2": l2(u_h, u_true),
                            "L_2_rel": l2_rel(u_h, u_true),
                        }
                    )

    df = pd.DataFrame(rows)
    save_csv(df, "discretization")
    plot_all(df)
    return df


def plot_all(df):
    plot(
        df,
        metric="L_inf_rel",
        ylabel=r"Rel. $L_\infty$ error",
        out_name="discretization",
    )
    plot(
        df,
        metric="L_2_rel",
        ylabel=r"Rel. $L_2$ error",
        out_name="discretization_l2",
    )


def plot(df, *, metric, ylabel, out_name):
    """Mean over seeds, one curve per `num_modes`; error is measured
    against the fine-grid reference."""
    apply_style()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.36),
        sharey=True,
        constrained_layout=True,
    )
    for ax, dim in zip(axes, [1, 2, 3]):
        sub = df[df["dim"] == dim]
        agg = (
            sub.groupby(["num_modes", "N"])[metric]
            .mean()
            .reset_index()
            .sort_values(["num_modes", "N"])
        )
        modes = sorted(sub["num_modes"].unique())
        colors = sequential_colors(len(modes))
        for m_idx, num_modes in enumerate(modes):
            curve = agg[agg["num_modes"] == num_modes]
            ax.loglog(
                curve["N"],
                curve[metric],
                marker=SERIES_MARKERS[m_idx],
                color=colors[m_idx],
                label=f"$M = {num_modes}$",
                **MULTI_PANEL_LINE,
            )

        Ns = sorted(sub["N"].unique())
        N_min = Ns[0]
        e_anchor = float(
            sub[(sub["N"] == N_min) & (sub["num_modes"] == sub["num_modes"].min())][
                metric
            ].mean()
        )
        ref_curve = [e_anchor * (N_min / N) ** 2 for N in Ns]
        ax.loglog(
            Ns,
            ref_curve,
            linestyle="--",
            color=COLOR_BASELINE_2,
            linewidth=MULTI_PANEL_LINE["linewidth"],
            label=r"$\mathcal{O}(N^{-2})$",
        )

        ax.set_title(f"{dim}D")
        ax.set_xlabel(r"Grid points per dim. ($N$)")
        if dim == 1:
            ax.set_ylabel(ylabel)
        ax.set_ylim(1e-5, 1e1)
        # N doubles from point to point: label the grid sizes themselves
        # rather than powers of ten.
        ax.set_xticks(Ns)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_locator(NullLocator())
        # The curves fill the panel diagonally and leave no corner for six
        # entries, so the legend goes below the axis.
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.3),
            ncol=3,
            columnspacing=1.0,
            handlelength=1.8,
        )
        style_grid(ax)
    save_figure(fig, out_name)


if __name__ == "__main__":
    if "--plot-only" in sys.argv[1:]:
        plot_all(pd.read_csv(RESULTS_DIR / "discretization.csv"))
    else:
        run()
