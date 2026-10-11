"""Shared helpers for the convergence studies.

Build a "problem": sample random Fourier coefficients once, solve at an
ultra-fine reference resolution in fp64. Downstream studies sample the
same analytical RHS on coarser grids and compare their solutions to the
reference (restricted to the target grid via `prolongate`).

`prolongate` is bidirectional — it uses `map_coordinates(order=1)`, so
sampling a fine-grid solution at a coarser target is just the same
operation with a different output shape.
"""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import pandas as pd

from poisson_nd.dst import solve_direct
from poisson_nd.multigrid import prolongate
from poisson_nd.operators import laplacian
from poisson_nd.rhs import evaluate_rhs, sample_coefficients

PACKAGE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PACKAGE_DIR / "results"
FIGURES_DIR = PACKAGE_DIR / "figures"


@dataclass
class Problem:
    """A single fixed random RHS plus its fp64 reference solution.

    `amplitudes` and `phases` define the analytical RHS continuously; any
    target grid samples the same field via `rhs_at`. `rhs_norm` is the max
    magnitude on the reference grid and is reused across resolutions so
    the field keeps a consistent scale.
    """

    amplitudes: jnp.ndarray
    phases: list
    rhs_norm: float
    domain_extent: float
    shape_ref: tuple
    u_ref: jnp.ndarray

    def rhs_at(self, shape, dtype=jnp.float64):
        """Sample the RHS on `shape`, cast to `dtype`, normalized by ref norm."""
        amps = self.amplitudes.astype(dtype)
        phs = [p.astype(dtype) for p in self.phases]
        vals = evaluate_rhs(amps, phs, shape, self.domain_extent)
        return vals / jnp.asarray(self.rhs_norm, dtype=dtype)

    def u_true_at(self, shape, dtype=jnp.float64):
        """Linearly interpolate the reference solution onto `shape`."""
        return prolongate(self.u_ref, shape).astype(dtype)

    def h_at(self, shape):
        return self.domain_extent / (shape[0] + 1)


def make_problem(key, num_modes, ndim, shape_ref, domain_extent=1.0):
    """Sample coefficients and compute the ultra-fine fp64 reference solve."""
    amplitudes, phases = sample_coefficients(key, num_modes, ndim)
    amplitudes = amplitudes.astype(jnp.float64)
    phases = [p.astype(jnp.float64) for p in phases]

    rhs_ref = evaluate_rhs(amplitudes, phases, shape_ref, domain_extent)
    rhs_norm = float(jnp.max(jnp.abs(rhs_ref)))
    rhs_ref = rhs_ref / rhs_norm

    h_ref = domain_extent / (shape_ref[0] + 1)
    u_ref = solve_direct(rhs_ref, h_ref)

    return Problem(
        amplitudes=amplitudes,
        phases=phases,
        rhs_norm=rhs_norm,
        domain_extent=domain_extent,
        shape_ref=shape_ref,
        u_ref=u_ref,
    )


def l_inf(u, u_ref):
    return float(jnp.max(jnp.abs(u.astype(jnp.float64) - u_ref.astype(jnp.float64))))


def l_inf_rel(u, u_ref):
    """max|u - u_ref| / max|u_ref| in fp64."""
    u64 = u.astype(jnp.float64)
    u_ref64 = u_ref.astype(jnp.float64)
    ref_norm = float(jnp.max(jnp.abs(u_ref64)))
    return float(jnp.max(jnp.abs(u64 - u_ref64))) / ref_norm


def l2(u, u_ref):
    diff = u.astype(jnp.float64) - u_ref.astype(jnp.float64)
    return float(jnp.sqrt(jnp.mean(diff**2)))


def l2_rel(u, u_ref):
    """||u - u_ref||_2 / ||u_ref||_2 (RMS relative error in fp64)."""
    u64 = u.astype(jnp.float64)
    u_ref64 = u_ref.astype(jnp.float64)
    ref_norm = float(jnp.sqrt(jnp.mean(u_ref64**2)))
    return float(jnp.sqrt(jnp.mean((u64 - u_ref64) ** 2))) / ref_norm


def residual_linf(u, rhs, h):
    """max |f - (-Lap u)| in fp64 regardless of the solver's own dtype."""
    u64 = u.astype(jnp.float64)
    rhs64 = rhs.astype(jnp.float64)
    r = rhs64 - (-laplacian(u64, float(h)))
    return float(jnp.max(jnp.abs(r)))


def save_csv(df: pd.DataFrame, name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def save_figure(fig, name: str) -> Path:
    """Save as PNG and PDF. No tight bbox: the figure keeps its declared
    width, which the font sizes in `style.py` rely on."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def seed_keys(seed0: int, n_seeds: int):
    return jr.split(jr.PRNGKey(seed0), n_seeds)
