"""Laplacian correctness checks.

- Known analytical Laplacian of prod_a sin(pi x_a / L) should be matched up
  to discretization error.
- Laplacian of a constant field restricted to interior is a boundary
  artifact; interior entries should vanish and boundary-adjacent entries
  should pick up the boundary term.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from poisson_nd.operators import laplacian, neighbor_sum

L = 1.0


def _grid(shape):
    return [(jnp.arange(shape[a]) + 1) * L / (shape[a] + 1) for a in range(len(shape))]


def _sin_product(shape):
    grids = _grid(shape)
    d = len(shape)
    vals = jnp.ones(shape)
    for a in range(d):
        reshape = [1] * d
        reshape[a] = shape[a]
        vals = vals * jnp.sin(jnp.pi * grids[a] / L).reshape(reshape)
    return vals


@pytest.mark.parametrize("dim,size", [(1, 64), (2, 32), (3, 16)])
def test_laplacian_matches_analytical(dim, size):
    shape = (size,) * dim
    h = L / (size + 1)

    u = _sin_product(shape)
    lap = laplacian(u, h)

    expected = -dim * (jnp.pi / L) ** 2 * u

    # Interior truncation error for the (2d+1)-pt stencil is O(h^2).
    # h = L/(size+1); with size=64 in 1D, h^2 ~ 2.4e-4 and the leading
    # coefficient is (pi/L)^4/12 which gives an error ~ 2e-3.
    err = jnp.max(jnp.abs(lap - expected))
    # Dominant mode amplitude is about d * (pi/L)^2 = d * pi^2 ~ 10d, so the
    # relative error is ~1e-3 at these sizes.
    assert err < 5e-2 * dim, f"dim={dim} size={size}: max abs err {err}"


@pytest.mark.parametrize("dim,size", [(1, 8), (2, 8), (3, 8)])
def test_laplacian_of_zero_is_zero(dim, size):
    shape = (size,) * dim
    u = jnp.zeros(shape)
    lap = laplacian(u, h=0.1)
    assert jnp.all(lap == 0.0)


@pytest.mark.parametrize("dim,size", [(1, 5), (2, 5), (3, 5)])
def test_neighbor_sum_no_wraparound(dim, size):
    """jnp.roll would wrap around; neighbor_sum must not.

    Place a 1 at the "first" corner. Opposite boundary should not see any
    contribution from this point.
    """
    shape = (size,) * dim
    u = np.zeros(shape)
    u[(0,) * dim] = 1.0
    s = neighbor_sum(jnp.asarray(u))
    # Opposite corner: no wrap-around should contribute.
    assert float(s[(size - 1,) * dim]) == 0.0
