"""Shared data loading, statistics and plot style for the result figures.

Reads paper/results/<experiment>.csv (written by evaluate.py: one row per run with b_tilde,
lambd, seed, hf_only and the test metrics).
"""
import os.path as osp
import sys

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), ".."))  # paper/figures (style.py)
from style import cmap_sequential, color_baseline, color_ours  # noqa: E402, F401 (colors re-exported as C.color_*)

RESULTS_DIR = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "results"))
ERROR_COL = "val/rel_l2_loss_fields_avg"
AIRFOIL_THRESHOLD = 0.02  # eps* for airfoil savings (volume field; 2% reads naturally)
COLUMN_WIDTH = 5.5
SEEDS = 3

EXPERIMENTS = {
    "linear_elasticity": {
        "Solver error": "linear_elasticity_solver_truncation",
        "Discretization error": "linear_elasticity_discretization",
    },
    "hyperelasticity": {
        "Solver error": "hyperelasticity_solver_truncation",
        "Discretization error": "hyperelasticity_discretization",
    },
}
PROBLEM_LABELS = {"linear_elasticity": "Lin. elast. 2D", "hyperelasticity": "Hyperelast. 2D",
                  "poisson1d": "Poisson 1D", "poisson2d": "Poisson 2D", "poisson3d": "Poisson 3D"}
MODELS = ["Transolver", "AB-UPT"]
POISSON_CSV = f"{RESULTS_DIR}/poisson.csv"
POISSON_MODELS = {"poisson1d": ["U-Net", "FNO"], "poisson2d": ["U-Net", "FNO"], "poisson3d": ["U-Net"]}

RC = {
    "text.usetex": False,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 6,
    "axes.labelsize": 7,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1,
    "lines.markersize": 1,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "savefig.dpi": 300,
}


def apply_style():
    matplotlib.rcParams.update(RC)


def navia_truncated():
    return mcolors.LinearSegmentedColormap.from_list(
        "navia_trunc", cmap_sequential(np.linspace(0, 0.9, 256)))


def btilde_colormap(btildes):
    norm = mcolors.LogNorm(vmin=min(btildes), vmax=max(btildes))
    cmap = navia_truncated()
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    return cmap, norm, sm


def decade_extreme_ticks(ax, min_sep_dec=0.25):
    """Y ticks: decade majors (10^k) always labeled, plus the actual axis EDGES when
    those are non-decade minors. An extreme minor is labeled only when it lies beyond
    the outermost decade (i.e. it is genuinely the lowest/highest visible tick); when a
    decade sits at the edge, that decade already labels the edge and no minor is added.
    Extreme labels within `min_sep_dec` decades of a labeled tick are skipped."""
    lo, hi = ax.get_ylim()
    decades = [10.0 ** k for k in range(int(np.ceil(np.log10(lo) - 1e-9)),
                                        int(np.floor(np.log10(hi) + 1e-9)) + 1)]
    # candidate minor ticks m*10^k inside the limits
    cands = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1,
                                          int(np.ceil(np.log10(hi))) + 1)
             for m in range(2, 10)]
    cands = [c for c in cands if lo <= c <= hi]
    ticks = list(decades)
    extremes = []
    if cands:
        # low edge: only if no decade lies below it (else the decade IS the edge)
        if not decades or min(cands) < min(decades):
            extremes.append(min(cands))
        # high edge: only if no decade lies above it
        if not decades or max(cands) > max(decades):
            extremes.append(max(cands))
    for extreme in extremes:
        if all(abs(np.log10(extreme) - np.log10(t)) >= min_sep_dec for t in ticks):
            ticks.append(extreme)
    ticks = sorted(set(ticks))

    def fmt(v):
        k = int(np.floor(np.log10(v) + 1e-9))
        m = v / 10.0 ** k
        if abs(m - 1) < 1e-6:
            return rf"$10^{{{k}}}$"
        return rf"${m:.0f}\times10^{{{k}}}$"

    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([fmt(t) for t in ticks]))
    ax.yaxis.set_minor_locator(mticker.FixedLocator([c for c in cands if c not in ticks]))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    # tick MARKS at every log position; labels only on the majors set above
    ax.tick_params(axis="y", which="major", length=2.5, pad=1)
    ax.tick_params(axis="y", which="minor", length=1.4)


def decade_xticks(ax, minor_marks=False):
    """X ticks/labels (and thus grid lines) at every decade inside the limits.
    minor_marks=True additionally draws unlabeled marks at the 2-9 positions."""
    lo, hi = ax.get_xlim()
    ks = range(int(np.ceil(np.log10(lo) - 1e-9)), int(np.floor(np.log10(hi) + 1e-9)) + 1)
    decades = [10.0 ** k for k in ks]
    ax.xaxis.set_major_locator(mticker.FixedLocator(decades))
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(
        [rf"$10^{{{k}}}$" for k in ks]))
    if minor_marks:
        cands = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1,
                                              int(np.ceil(np.log10(hi))) + 1)
                 for m in range(2, 10)]
        ax.xaxis.set_minor_locator(mticker.FixedLocator(
            [c for c in cands if lo <= c <= hi]))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="x", which="minor", length=1.8)
    else:
        ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", which="major", length=2.5)


def log_ticks(ax, axis="y"):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(mticker.LogLocator(base=10.0))
    a.set_major_formatter(mticker.LogFormatterSciNotation(base=10.0))
    a.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=10))
    a.set_minor_formatter(mticker.NullFormatter())


def load_experiments(results_dir=RESULTS_DIR):
    """-> {problem: {axis_label: df}} with model_name ABUPT renamed to AB-UPT."""
    out = {}
    for prob, axes in EXPERIMENTS.items():
        out[prob] = {}
        for axis_label, exp in axes.items():
            df = pd.read_csv(f"{results_dir}/{exp}.csv")
            df.loc[df.model_name == "ABUPT", "model_name"] = "AB-UPT"
            out[prob][axis_label] = df
    return out


def load_poisson(csv_path=POISSON_CSV):
    """Poisson (grid models) -> {problem: {axis_label: df}} in the plate schema. No
    separate HF-only runs exist: the HF baseline is the max-lambda cell per budget --
    rows are duplicated into an
    hf_only=True view for the shared curve/savings machinery."""
    p = pd.read_csv(csv_path)
    # two small 2D budgets with a different normalization are excluded, as in the paper
    p = p[~(np.isclose(p.B_tilde, 5.42534722) & (p.dim == 2))]
    p = p[~(np.isclose(p.B_tilde, 14.28571429) & (p.dim == 2))]
    p["model_name"] = p.architecture.map({"unet": "U-Net", "fno": "FNO"})
    p = p.rename(columns={"B_tilde": "b_tilde"})
    p[ERROR_COL] = p.mean_prediction_error
    out = {}
    for dim in (1, 2, 3):
        prob = f"poisson{dim}d"
        out[prob] = {}
        for ax, axis_label in [("discretization", "Discretization error"),
                               ("truncation", "Solver error")]:
            d = p[(p.dim == dim) & (p.axis == ax)
                  & p.model_name.isin(POISSON_MODELS[prob])].copy()
            d["hf_only"] = False
            mx = d.groupby(["model_name", "b_tilde"]).lambd.transform("max")
            hf = d[d.lambd == mx].copy()
            hf["hf_only"] = True
            out[prob][axis_label] = pd.concat([d, hf], ignore_index=True)
    return out


def iso_budget_stats(df_mf, error_col=ERROR_COL):
    """{b_tilde: DataFrame(lambd, mean, std)} for the MF cells of one model."""
    out = {}
    for bt in sorted(df_mf.b_tilde.unique()):
        stats = (df_mf[df_mf.b_tilde == bt].groupby("lambd")[error_col]
                 .agg(["mean", "std"]).reset_index().sort_values("lambd"))
        if not stats.empty and not stats["mean"].isna().all():
            out[bt] = stats
    return out


def mf_optimal_curve(df_mf, error_col=ERROR_COL, agg="mean"):
    """Per budget: min over lambda of the seed-aggregated error. -> (bt, mean, std)."""
    bts, means, stds = [], [], []
    for bt, stats in iso_budget_stats(df_mf, error_col).items():
        i = stats["mean"].idxmin() if agg == "mean" else stats["mean"].idxmin()
        bts.append(bt)
        means.append(stats.loc[i, "mean"])
        stds.append(stats.loc[i, "std"])
    return np.array(bts), np.array(means), np.array(stds)


def hf_curve(df_hf, error_col=ERROR_COL):
    g = df_hf.groupby("b_tilde")[error_col].agg(["mean", "std"]).reset_index().sort_values("b_tilde")
    return g.b_tilde.values, g["mean"].values, g["std"].values


def _btilde_at_error(btildes, errors, threshold):
    """Log-log interpolation of the budget where the curve crosses `threshold`."""
    order = np.argsort(btildes)
    bt, er = np.asarray(btildes)[order], np.asarray(errors)[order]
    for i in range(len(er) - 1):
        if er[i] >= threshold >= er[i + 1]:
            t = (np.log10(threshold) - np.log10(er[i])) / (np.log10(er[i + 1]) - np.log10(er[i]))
            return 10 ** (np.log10(bt[i]) + t * (np.log10(bt[i + 1]) - np.log10(bt[i])))
    return np.nan


def compute_savings(df, threshold, model_name, error_col=ERROR_COL):
    """S(eps*) = B~_HF / B~_MF at the error threshold (paper convention, seed means)."""
    sub = df[df.model_name == model_name]
    mf_bt, mf_err, _ = mf_optimal_curve(sub[~sub.hf_only], error_col)
    hf_bt, hf_err, _ = hf_curve(sub[sub.hf_only], error_col)
    b_mf = _btilde_at_error(mf_bt, mf_err, threshold)
    b_hf = _btilde_at_error(hf_bt, hf_err, threshold)
    return dict(savings_ratio=b_hf / b_mf if np.isfinite(b_hf) and np.isfinite(b_mf) else np.nan,
                b_tilde_hf=b_hf, b_tilde_mf=b_mf)


def bootstrap_savings(df, threshold, model_name, n_draws=2000, seed=0, error_col=ERROR_COL):
    """Seed-bootstrap of S(eps*): per draw, resample seeds within every cell,
    re-select lambda* per budget, re-interpolate both curves, recompute the ratio.
    -> (lo, median, hi): 2.5/50/97.5 percentiles over the finite draws (bar height =
    median, whiskers = lo/hi, all from one distribution)."""
    sub = df[df.model_name == model_name]
    mf = sub[~sub.hf_only]
    hf = sub[sub.hf_only]
    # cell -> per-seed error arrays
    mf_cells = {(bt, l): g[error_col].values
                for (bt, l), g in mf.groupby(["b_tilde", "lambd"])}
    hf_cells = {bt: g[error_col].values for bt, g in hf.groupby("b_tilde")}
    mf_bts = sorted({bt for bt, _ in mf_cells})
    hf_bts = sorted(hf_cells)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_draws):
        mf_err = []
        for bt in mf_bts:
            best = np.inf
            for (b, l), v in mf_cells.items():
                if b != bt:
                    continue
                m = rng.choice(v, size=len(v), replace=True).mean()
                best = min(best, m)
            mf_err.append(best)
        hf_err = [rng.choice(hf_cells[bt], size=len(hf_cells[bt]), replace=True).mean()
                  for bt in hf_bts]
        b_mf = _btilde_at_error(np.array(mf_bts), np.array(mf_err), threshold)
        b_hf = _btilde_at_error(np.array(hf_bts), np.array(hf_err), threshold)
        if np.isfinite(b_mf) and np.isfinite(b_hf):
            draws.append(b_hf / b_mf)
    if len(draws) < n_draws * 0.5:
        return np.nan, np.nan, np.nan
    return tuple(np.percentile(draws, [2.5, 50, 97.5]))
