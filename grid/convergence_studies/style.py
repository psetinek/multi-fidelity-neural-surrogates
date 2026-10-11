"""Figure style shared by the appendix figures of the Poisson experiments
(the convergence studies and `scripts/plot_fidelity_examples.py`).

Mirrors the style of the linear-elasticity figures of the paper (rcParams,
widths, colormaps and line colours), so the appendix figures of the
different experiments look alike.

Font sizes are in points at final size. A figure therefore has to be
created at `COLUMN_WIDTH` (or at the fraction of it that it is included
with) and saved without `bbox_inches="tight"`; cropping changes the width
and with it the font size after `\\includegraphics[width=...]`.
"""

import cmcrameri.cm as cmc
import matplotlib
import numpy as np

# NeurIPS dimensions, in inches.
COLUMN_WIDTH = 5.5  # single column
FULL_WIDTH = 11.0  # two-column span
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# Colours for plotting.
CMAP_SEQUENTIAL = cmc.navia
CMAP_DIVERGING = cmc.vik
COLOR_OURS = "#2D6A4F"
COLOR_BASELINE = "#CF4924"
COLOR_BASELINE_2 = "#7A7A7A"

RC_PARAMS = {
    "text.usetex": False,  # no external LaTeX needed
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Computer Modern math
    "font.size": 6,
    "axes.labelsize": 7,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.5,
    "grid.linewidth": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "grid.alpha": 0.3,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "legend.fancybox": False,
    "savefig.dpi": 300,
    # Not part of the look: powers of ten as 10^k rather than 1ek, and
    # TrueType instead of Type 3 fonts in the PDFs.
    "axes.formatter.use_mathtext": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Alpha of a spread band (IQR / std) around a line.
BAND_ALPHA = 0.3

# Line plots with three panels across the column: the panels are ~0.6x
# as wide as a single 0.48-column panel, so the marks shrink with them.
MULTI_PANEL_LINE = {"linewidth": 1.0, "markersize": 3}

# One marker per series: neighbouring colours of `sequential_colors` are
# close, so the marker carries the series identity as well.
SERIES_MARKERS = ("o", "s", "^", "D", "v")


def apply_style():
    matplotlib.rcParams.update(RC_PARAMS)


def style_grid(ax):
    """Dashed major + minor grid of the line plots."""
    ax.grid(True, which="both", ls="--", alpha=0.5)


def sequential_colors(n):
    """`n` line colours for an ordered series, dark to light.

    Stops short of the pale end of `CMAP_SEQUENTIAL`, which does not
    read as a line on a white background.
    """
    return [CMAP_SEQUENTIAL(v) for v in np.linspace(0.08, 0.75, n)]
