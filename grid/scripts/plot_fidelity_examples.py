"""Appendix figures: one Poisson sample at the lowest vs highest fidelity.

Mirrors the qualitative appendix figures of the other experiments: three
panels "Lowest fidelity | Highest fidelity | Error" with Crameri's `navia`
for the fields (shared colorbar) and `vik` for the error (symmetric
colorbar), colorbars underneath. Fonts, colours and the panel/colorbar
placement are those of the linear-elasticity figure (see
`convergence_studies/style.py`); the figures are `COLUMN_WIDTH` wide and
saved uncropped. One figure per (axis, dim):

- 1D: line plots, three panels with the same titles.
- 2D: heatmaps.
- 3D: vape4d volume renders (needs WebGPU), matplotlib colorbars. Colour
  ranges are clipped at the 95th percentile of |v| (colorbars extended),
  and opacity ramps up away from u = 0.

Samples come from the training generators themselves
(`generate_truncation` / `generate_discretization`, linear prolongation
to the target grid), so the panels show exactly the (f, u) pairs the
surrogate sees. The RHS key is the `--sample`-th test-set key, and the
lowest / highest fidelity are the ends of the axis menu in
`configs/axis/<axis>_<dim>d.yaml`. Error = u_lowest - u_highest.
On the discretization axis a second figure (`*_rhs`) shows the RHS the
network receives, which also changes with the native resolution.

Run from the repo root:
    uv run python scripts/plot_fidelity_examples.py
    uv run python scripts/plot_fidelity_examples.py --dim 2 --axis truncation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax.random as jr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import ScalarFormatter

from convergence_studies.style import (
    CMAP_DIVERGING,
    CMAP_SEQUENTIAL,
    COLOR_BASELINE,
    COLOR_OURS,
    COLUMN_WIDTH,
    apply_style,
    style_grid,
)
from multifidelity.data import generate_discretization, generate_truncation

CONFIG_DIR = Path("configs")
OUT_DIR = Path("results/plots/fidelity_examples")

FIELD_CMAP = CMAP_SEQUENTIAL
ERROR_CMAP = CMAP_DIVERGING
SOLUTION_LABEL = r"Solution $u$"
RHS_LABEL = r"Right-hand side $f$"

# Three-panel row of the linear-elasticity figure, which is COLUMN_WIDTH
# wide and LINE_FIG_H high. Horizontal placement is taken over as figure
# fractions. Vertical placement is converted to inches, because the panels
# here are square instead of 2:1 and the figure is correspondingly taller.
LINE_FIG_H = COLUMN_WIDTH * 0.32
PANEL_X = (0.02, 0.32, 0.66)
PANEL_W = 0.28
CBAR_FIELD = (0.05, 0.52)  # (left, width), spans the two field panels
CBAR_ERROR = (0.68, 0.24)
CBAR_Y = 0.25 * LINE_FIG_H
CBAR_H = 0.03 * LINE_FIG_H
PANEL_Y = 0.35 * LINE_FIG_H
TITLE_H = 0.13 * LINE_FIG_H

# vape4d
RENDER_PX = 600
CLIP_PERCENTILE = 95.0  # symmetric |v| clip; saturates the colormap ends
ALPHA_POWER = 1.5
CAMERA_ANGLE = (1.05, 0.55)
DISTANCE_SCALE = 4.0

apply_style()


def load_setup(axis, dim):
    """Menu ends and generator settings from the Hydra YAMLs."""
    base = yaml.safe_load((CONFIG_DIR / "config.yaml").read_text())
    menu = yaml.safe_load((CONFIG_DIR / "axis" / f"{axis}_{dim}d.yaml").read_text())
    values = menu["fidelity_values"]
    return {
        "lo": int(min(values)),
        "hi": int(max(values)),
        "shape": (int(base["target_resolution"]),) * dim,
        "num_modes": int(base["num_modes"]),
        "domain_extent": float(base["domain_extent"]),
        "test_seed": int(base["test_seed"]),
        "num_test_samples": int(base["num_test_samples"]),
    }


def generate_pair(axis, dim, sample):
    """Return (f_lo, u_lo, f_hi, u_hi, setup) for one test-set RHS."""
    s = load_setup(axis, dim)
    key = jr.split(jr.PRNGKey(s["test_seed"]), s["num_test_samples"])[sample]
    kw = {"num_modes": s["num_modes"], "domain_extent": s["domain_extent"]}
    if axis == "truncation":
        f_lo, u_lo = generate_truncation(key, s["lo"], s["shape"], **kw)
        f_hi, u_hi = generate_truncation(key, s["hi"], s["shape"], **kw)
    else:
        f_lo, u_lo = generate_discretization(key, s["lo"], s["shape"], **kw)
        f_hi, u_hi = generate_discretization(key, s["hi"], s["shape"], **kw)
    return tuple(np.asarray(a) for a in (f_lo, u_lo, f_hi, u_hi)) + (s,)


def _layout(fig_h):
    """Figure-fraction rects for the three panels and the two colorbars."""
    h = lambda y: y / fig_h  # noqa: E731
    panel_h = h(PANEL_W * COLUMN_WIDTH)
    panels = [[x, h(PANEL_Y), PANEL_W, panel_h] for x in PANEL_X]

    def cbar(left, width):
        return [left, h(CBAR_Y), width, h(CBAR_H)]

    return panels, cbar(*CBAR_FIELD), cbar(*CBAR_ERROR)


def _colorbar(fig, rect, cmap, norm, label, extend="neither"):
    cax = fig.add_axes(rect)
    cb = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        extend=extend,
        format=ScalarFormatter(useMathText=True),
    )
    # Powers of ten only where they shorten the labels (the RHS is O(1)).
    cb.ax.ticklabel_format(style="scientific", scilimits=(-2, 2))
    cb.ax.set_xlabel(label)
    return cb


def _norms(u_lo, u_hi, err, clip_percentile=None):
    """Field range shared by both fidelities; symmetric error range.

    With `clip_percentile`, both ranges are symmetric and clipped at that
    percentile of |v| (used for the 3D renders).
    """
    if clip_percentile is None:
        field = Normalize(min(u_lo.min(), u_hi.min()), max(u_lo.max(), u_hi.max()))
        e = float(np.abs(err).max())
        return field, Normalize(-e, e)
    m = float(np.percentile(np.abs(np.concatenate([u_lo, u_hi])), clip_percentile))
    e = float(np.percentile(np.abs(err), clip_percentile))
    return Normalize(-m, m), Normalize(-e, e)


def _three_panel_images(imgs, u_lo, u_hi, err, label, clip_percentile=None):
    """Shared figure scaffold for 2D heatmaps and 3D renders."""
    fig_h = PANEL_Y + PANEL_W * COLUMN_WIDTH + TITLE_H
    fig = plt.figure(figsize=(COLUMN_WIDTH, fig_h))
    rects, cbar_field, cbar_err = _layout(fig_h)
    field_norm, err_norm = _norms(u_lo, u_hi, err, clip_percentile)
    titles = ["Lowest fidelity", "Highest fidelity", "Error"]
    for rect, (img, kw), title in zip(rects, imgs(field_norm, err_norm), titles):
        ax = fig.add_axes(rect)
        ax.imshow(img, **kw)
        ax.set_title(title, pad=4)
        ax.axis("off")
    extend = "neither" if clip_percentile is None else "both"
    _colorbar(fig, cbar_field, FIELD_CMAP, field_norm, label, extend)
    _colorbar(fig, cbar_err, ERROR_CMAP, err_norm, label, extend)
    return fig


def figure_2d(u_lo, u_hi, err, label):
    def imgs(field_norm, err_norm):
        common = {"origin": "lower", "extent": (0, 1, 0, 1), "interpolation": "bilinear"}
        return [
            (u_lo.T, {**common, "cmap": FIELD_CMAP, "norm": field_norm}),
            (u_hi.T, {**common, "cmap": FIELD_CMAP, "norm": field_norm}),
            (err.T, {**common, "cmap": ERROR_CMAP, "norm": err_norm}),
        ]

    return _three_panel_images(imgs, u_lo, u_hi, err, label)


def ramp_alpha(cmap):
    """Opacity grows with distance from the colormap centre (u = 0), so the
    near-zero bulk of the volume stays transparent."""
    rgba = cmap(np.linspace(0.0, 1.0, 256))
    rgba[:, 3] = np.abs(np.linspace(-1.0, 1.0, 256)) ** ALPHA_POWER
    return ListedColormap(rgba, name=cmap.name + "_ramp")


def figure_3d(u_lo, u_hi, err, label):
    from vape4d import render

    def vol(v, norm, cmap):
        v01 = np.clip(norm(v), 0.0, 1.0).astype(np.float32)
        return render(
            v01[None],
            cmap=ramp_alpha(cmap),
            time=0.0,
            width=RENDER_PX,
            height=RENDER_PX,
            vmin=0.0,
            vmax=1.0,
            distance_scale=DISTANCE_SCALE,
            background=(1.0, 1.0, 1.0, 1.0),
            camera_angle=CAMERA_ANGLE,
        )

    def imgs(field_norm, err_norm):
        renders = [
            vol(u_lo, field_norm, FIELD_CMAP),
            vol(u_hi, field_norm, FIELD_CMAP),
            vol(err, err_norm, ERROR_CMAP),
        ]
        return [(img, {}) for img in _crop_common(renders)]

    return _three_panel_images(
        imgs, u_lo, u_hi, err, label, clip_percentile=CLIP_PERCENTILE
    )


def _crop_common(images, pad=6):
    """Crop RGBA renders on a white background to their joint content box."""
    content = np.zeros(images[0].shape[:2], dtype=bool)
    for img in images:
        content |= (img[..., :3] < 250).any(axis=-1)
    rows, cols = np.where(content)
    r0, r1 = max(rows.min() - pad, 0), rows.max() + pad + 1
    c0, c1 = max(cols.min() - pad, 0), cols.max() + pad + 1
    return [img[r0:r1, c0:c1] for img in images]


def _pow10(values):
    """Exponent p such that values / 10**p has O(1) magnitude; 0 in the
    range where the colorbars print plain numbers as well."""
    p = int(np.floor(np.log10(float(np.abs(values).max()))))
    return p if abs(p) >= 2 else 0


def figure_1d(u_lo, u_hi, err, label):
    fig, axs = plt.subplots(
        1, 3, figsize=(COLUMN_WIDTH, LINE_FIG_H), constrained_layout=True
    )
    axs[1].sharey(axs[0])
    x = np.arange(1, u_lo.shape[0] + 1) / (u_lo.shape[0] + 1)
    field_color = COLOR_OURS
    err_color = COLOR_BASELINE

    # Put the power of ten into the y-label: an offset text would collide
    # with the panel title.
    p_field, p_err = _pow10(np.concatenate([u_lo, u_hi])), _pow10(err)
    u_lo, u_hi, err = u_lo / 10.0**p_field, u_hi / 10.0**p_field, err / 10.0**p_err
    lo = min(u_lo.min(), u_hi.min())
    hi = max(u_lo.max(), u_hi.max())
    pad = 0.08 * (hi - lo)
    e = 1.1 * float(np.abs(err).max())

    specs = [
        (u_lo, field_color, (lo - pad, hi + pad), "Lowest fidelity", p_field),
        (u_hi, field_color, (lo - pad, hi + pad), "Highest fidelity", None),
        (err, err_color, (-e, e), "Error", p_err),
    ]
    for ax, (y, c, ylim, title, p) in zip(axs, specs):
        ax.plot(x, y, color=c)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=4)
        ax.set_xlabel(r"$x$")
        style_grid(ax)
        if p is None:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(label if p == 0 else rf"{label} $[\times 10^{{{p}}}]$")
    return fig


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dim", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--axis", nargs="+", default=["truncation", "discretization"]
    )
    parser.add_argument("--sample", type=int, default=0, help="test-set index")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    builders = {1: figure_1d, 2: figure_2d, 3: figure_3d}
    symbol = {"truncation": "K", "discretization": "N"}
    for axis in args.axis:
        for dim in args.dim:
            f_lo, u_lo, f_hi, u_hi, s = generate_pair(axis, dim, args.sample)
            k = symbol[axis]
            print(
                f"{axis:15s} {dim}D  {k}={s['lo']} vs {k}={s['hi']}  "
                f"rel-L2(u)={rel_l2(u_lo, u_hi):.3f}  "
                f"rel-L2(f)={rel_l2(f_lo, f_hi):.3f}"
            )
            figs = {"": (u_lo, u_hi, SOLUTION_LABEL)}
            # On the truncation axis f is identical across fidelities; on the
            # discretization axis it is sampled natively and resampled, so the
            # network input differs too.
            if axis == "discretization":
                figs["_rhs"] = (f_lo, f_hi, RHS_LABEL)
            for suffix, (lo, hi, label) in figs.items():
                fig = builders[dim](lo, hi, lo - hi, label)
                out = OUT_DIR / f"{axis}_{dim}d{suffix}.pdf"
                fig.savefig(out)
                fig.savefig(out.with_suffix(".png"))
                plt.close(fig)
                print(f"  -> {out}")

if __name__ == "__main__":
    main()
