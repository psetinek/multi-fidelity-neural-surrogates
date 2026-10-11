"""Geometric multigrid V-cycle for -Laplacian u = f in arbitrary dimension.

Components (all dimension-generic, inferred from array shapes):
- `restrict`: separable full-weighting, applied axis-by-axis with [1,2,1]/4.
- `prolongate`: linear interpolation via `jax.scipy.ndimage.map_coordinates`.
  Required because N = 2^k produces non-nested fine/coarse grids.
- `_make_smoother`: red-black Gauss-Seidel; red mask is
  (sum of indices) % 2 == 0.
- `_build_v_cycle`: Python-time unrolled recursion given a list of per-level
  shapes so the V-cycle compiles to a single JIT-able function.

`build_multigrid_solver` returns a single-V-cycle callable; the iterator
variant takes `num_v_cycles` as a runtime argument so one compiled function
serves an entire truncation-error sweep.
"""

import jax
import jax.numpy as jnp

from poisson_nd.operators import laplacian, neighbor_sum


def _restrict_axis(u, axis):
    """Apply 1D full-weighting [1,2,1]/4 with stride 2 along `axis`.

    Requires u.shape[axis] to be even. Output has half that length along
    `axis`.
    """
    N = u.shape[axis]
    Nc = N // 2
    pad_widths = [(0, 0)] * u.ndim
    pad_widths[axis] = (1, 1)
    padded = jnp.pad(u, pad_widths, mode="constant", constant_values=0.0)

    minus_slc = [slice(None)] * u.ndim
    center_slc = [slice(None)] * u.ndim
    plus_slc = [slice(None)] * u.ndim
    minus_slc[axis] = slice(1, 2 * Nc, 2)
    center_slc[axis] = slice(2, 2 * Nc + 1, 2)
    plus_slc[axis] = slice(3, 2 * Nc + 2, 2)

    return (
        padded[tuple(minus_slc)]
        + 2.0 * padded[tuple(center_slc)]
        + padded[tuple(plus_slc)]
    ) / 4.0


def restrict(u):
    """Full-weighting restriction from fine to coarse grid.

    Applies 1D full-weighting along each axis in turn; the combined
    d-dimensional stencil is the outer product of d copies of [1,2,1]/4.
    Requires every axis of u to have even length.
    """
    for axis in range(u.ndim):
        u = _restrict_axis(u, axis)
    return u


def prolongate(u_coarse, shape_fine):
    """Linear interpolation from coarse grid to `shape_fine`.

    The coarse grid is padded with zeros on each side to encode the
    Dirichlet BC. Fine-grid point i along axis a is located at fractional
    coarse-padded coordinate (i + 1) * (Nc_a + 1) / (Nf_a + 1).
    """
    d = u_coarse.ndim
    shape_coarse = u_coarse.shape
    padded = jnp.pad(u_coarse, 1, mode="constant", constant_values=0.0)

    axes_1d = []
    for a in range(d):
        Nc = shape_coarse[a]
        Nf = shape_fine[a]
        frac = (jnp.arange(Nf) + 1) * (Nc + 1) / (Nf + 1)
        axes_1d.append(frac)
    grids = jnp.meshgrid(*axes_1d, indexing="ij")
    coordinates = jnp.stack(grids, axis=0)

    return jax.scipy.ndimage.map_coordinates(
        padded, coordinates, order=1, mode="constant"
    )


def _make_smoother(shape):
    """Build a red-black Gauss-Seidel sweep function for a fixed shape.

    The red/black masks are precomputed as trace-time constants. One sweep
    performs a red update followed by a black update.
    """
    d = len(shape)
    indices = jnp.indices(shape)
    red_mask = indices.sum(axis=0) % 2 == 0
    black_mask = ~red_mask

    def sweep(u, rhs, h):
        h2 = h * h
        new_vals = (neighbor_sum(u) + h2 * rhs) / (2 * d)
        u = jnp.where(red_mask, new_vals, u)
        new_vals = (neighbor_sum(u) + h2 * rhs) / (2 * d)
        u = jnp.where(black_mask, new_vals, u)
        return u

    return sweep


def _default_levels(shape):
    """Halve every axis while each length stays even and >= 8 post-halving.

    Matches the reference's [64, 32, 16, 8, 4] convention for shape=(64,...).
    """
    shapes = [tuple(shape)]
    while all(N >= 8 and N % 2 == 0 for N in shapes[-1]):
        shapes.append(tuple(N // 2 for N in shapes[-1]))
    return shapes


def _explicit_levels(shape, num_levels):
    shapes = [tuple(shape)]
    for _ in range(num_levels - 1):
        current = shapes[-1]
        if not all(N % 2 == 0 and N >= 2 for N in current):
            raise ValueError(
                f"Cannot coarsen shape {current} further for num_levels={num_levels}"
            )
        shapes.append(tuple(N // 2 for N in current))
    return shapes


def _build_v_cycle(shapes, nu1, nu2, coarse_sweeps):
    smoothers = {shape: _make_smoother(shape) for shape in shapes}

    def _v_cycle_level(u, rhs, h, level_idx):
        shape = shapes[level_idx]
        smoother = smoothers[shape]

        if level_idx == len(shapes) - 1:

            def coarse_step(_, u_):
                return smoother(u_, rhs, h)

            return jax.lax.fori_loop(0, coarse_sweeps, coarse_step, u)

        def pre_smooth(_, u_):
            return smoother(u_, rhs, h)

        u = jax.lax.fori_loop(0, nu1, pre_smooth, u)

        residual = rhs - (-laplacian(u, h))
        r_coarse = restrict(residual)

        shape_c = shapes[level_idx + 1]
        h_coarse = 2.0 * h

        e_coarse = jnp.zeros(shape_c, dtype=u.dtype)
        e_coarse = _v_cycle_level(e_coarse, r_coarse, h_coarse, level_idx + 1)

        correction = prolongate(e_coarse, shape)
        u = u + correction

        def post_smooth(_, u_):
            return smoother(u_, rhs, h)

        u = jax.lax.fori_loop(0, nu2, post_smooth, u)

        return u

    def v_cycle(u, rhs, h):
        return _v_cycle_level(u, rhs, h, 0)

    return v_cycle


def _resolve_levels(shape, num_levels):
    if num_levels is None:
        return _default_levels(shape)
    return _explicit_levels(shape, num_levels)


def build_multigrid_solver(shape, *, num_levels=None, nu1=2, nu2=2, coarse_sweeps=50):
    """Return a closure that performs one V-cycle starting from u=0.

    Arguments:
        shape: tuple of per-axis interior sizes; baked in as a trace-time
            constant (the level hierarchy depends on it).
        num_levels: if None, coarsen while each axis stays even and >= 8.
        nu1, nu2: pre- and post-smoothing sweep counts.
        coarse_sweeps: smoothing sweeps at the coarsest level.
    """
    shapes = _resolve_levels(shape, num_levels)
    v_cycle = _build_v_cycle(shapes, nu1, nu2, coarse_sweeps)

    def solver(rhs, h):
        u0 = jnp.zeros_like(rhs)
        return v_cycle(u0, rhs, h)

    return solver


def build_multigrid_iterator(shape, *, num_levels=None, nu1=2, nu2=2, coarse_sweeps=50):
    """Return a closure iterating `num_v_cycles` V-cycles starting from u=0.

    `num_v_cycles` is a runtime argument so a single compiled function can
    serve an entire truncation-error sweep K in {1, 2, 4, 8, ...}.
    """
    shapes = _resolve_levels(shape, num_levels)
    v_cycle = _build_v_cycle(shapes, nu1, nu2, coarse_sweeps)

    def iterator(rhs, h, num_v_cycles):
        def step(_, u):
            return v_cycle(u, rhs, h)

        u0 = jnp.zeros_like(rhs)
        return jax.lax.fori_loop(0, num_v_cycles, step, u0)

    return iterator
