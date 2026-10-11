"""Multigrid checks: transfer operators, convergence, and iterator monotonicity."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from poisson_nd.dst import solve_direct
from poisson_nd.multigrid import (
    build_multigrid_iterator,
    build_multigrid_solver,
    prolongate,
    restrict,
)

L = 1.0


@pytest.mark.parametrize("dim,size", [(1, 32), (2, 16), (3, 8)])
def test_restrict_preserves_constant_interior(dim, size):
    """Full-weighting of a constant 1 field is 1 in the interior and 3/4
    (or the corresponding tensor-product fraction) near the boundary."""
    shape = (size,) * dim
    u = jnp.ones(shape)
    uc = restrict(u)

    assert uc.shape == (size // 2,) * dim
    # Interior of the coarse grid: enough distance from every boundary.
    if size // 2 >= 3:
        interior = (slice(1, -1),) * dim
        assert jnp.allclose(uc[interior], 1.0, atol=1e-12)


@pytest.mark.parametrize("dim,size", [(1, 32), (2, 16), (3, 8)])
def test_prolongate_of_constant_is_constant_away_from_boundary(dim, size):
    """Linear interpolation of a constant-1 coarse field fills the fine
    grid with values close to 1 away from the boundary."""
    shape_coarse = (size // 2,) * dim
    shape_fine = (size,) * dim
    uc = jnp.ones(shape_coarse)
    u = prolongate(uc, shape_fine)

    assert u.shape == shape_fine
    # Central slab far from boundaries.
    margin = size // 4
    interior = (slice(margin, -margin),) * dim
    assert jnp.all(u[interior] > 0.5)
    assert jnp.max(u) <= 1.0 + 1e-12


@pytest.mark.parametrize("dim,size", [(1, 32), (2, 16), (3, 8)])
def test_multigrid_converges_to_direct(dim, size):
    shape = (size,) * dim
    h = L / (size + 1)

    key = jr.PRNGKey(3)
    rhs = jr.normal(key, shape)

    iterator = build_multigrid_iterator(shape)
    u_mg = iterator(rhs, h, 30)
    u_direct = solve_direct(rhs, h)

    err = float(jnp.max(jnp.abs(u_mg - u_direct)))
    scale = float(jnp.max(jnp.abs(u_direct)))
    assert err / scale < 1e-6, f"dim={dim} size={size}: rel err {err / scale}"


@pytest.mark.parametrize("dim,size", [(1, 32), (2, 16), (3, 8)])
def test_multigrid_iterator_monotonic(dim, size):
    shape = (size,) * dim
    h = L / (size + 1)

    key = jr.PRNGKey(4)
    rhs = jr.normal(key, shape)

    iterator = build_multigrid_iterator(shape)
    u_direct = solve_direct(rhs, h)

    errs = [
        float(jnp.linalg.norm(iterator(rhs, h, k) - u_direct)) for k in [1, 2, 4, 8, 16]
    ]
    for prev, curr in zip(errs, errs[1:]):
        assert curr <= prev * 1.01, f"dim={dim} errs {errs}"


@pytest.mark.parametrize("dim,size", [(1, 16), (2, 16), (3, 8)])
def test_single_v_cycle_solver_matches_iterator_at_one(dim, size):
    shape = (size,) * dim
    h = L / (size + 1)

    key = jr.PRNGKey(5)
    rhs = jr.normal(key, shape)

    solver = build_multigrid_solver(shape)
    iterator = build_multigrid_iterator(shape)

    assert jnp.allclose(solver(rhs, h), iterator(rhs, h, 1), atol=1e-12)
