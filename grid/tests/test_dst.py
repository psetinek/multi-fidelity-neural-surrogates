"""DST-I round-trip and direct solver convergence checks.

The DST-I is self-inverse up to a factor of (2(N+1))^d (one factor 2(N+1)
per axis), so `dstn(dstn(x)) / norm` should recover x.

For the direct solver:
- The discrete-exact test feeds a discrete RHS f = -laplacian(u*, h) and
  checks we recover u* to machine precision.
- The MMS test uses the continuous Laplacian of u* as the RHS and checks
  O(h^2) convergence as h shrinks.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from poisson_nd.dst import dstn_type1, solve_direct
from poisson_nd.operators import laplacian

L = 1.0


def _sin_product_grid(shape):
    d = len(shape)
    vals = jnp.ones(shape)
    for a in range(d):
        x = (jnp.arange(shape[a]) + 1) * L / (shape[a] + 1)
        reshape = [1] * d
        reshape[a] = shape[a]
        vals = vals * jnp.sin(jnp.pi * x / L).reshape(reshape)
    return vals


@pytest.mark.parametrize("dim,size", [(1, 8), (2, 8), (3, 8)])
def test_dst_round_trip(dim, size):
    shape = (size,) * dim
    key = jr.PRNGKey(0)
    x = jr.normal(key, shape)
    norm = 1.0
    for N in shape:
        norm *= 2 * (N + 1)
    x_back = dstn_type1(dstn_type1(x)) / norm
    assert jnp.allclose(x, x_back, atol=1e-10)


@pytest.mark.parametrize("dim,size", [(1, 16), (2, 16), (3, 16)])
def test_solve_direct_discrete_consistency(dim, size):
    """f = -laplacian(u*, h) should give solve_direct(f, h) == u* exactly."""
    shape = (size,) * dim
    h = L / (size + 1)

    key = jr.PRNGKey(1)
    u_star = jr.normal(key, shape)
    rhs = -laplacian(u_star, h)

    u = solve_direct(rhs, h)
    assert jnp.allclose(u, u_star, atol=1e-10)


@pytest.mark.parametrize("dim,size", [(1, 64), (2, 32), (3, 16)])
def test_solve_direct_matches_analytical(dim, size):
    """Solver on a continuous-Laplacian RHS recovers the analytical solution.

    u*(x) = prod_a sin(pi x_a / L) satisfies -Laplacian u* = d (pi/L)^2 u*.
    Feeding that continuous RHS to the solver should give back u* up to
    leading-order discretization error. Expanding the 3-pt eigenvalue gives

        u_dst - u* ~ u* * (pi/L)^2 h^2 / 12

    which is dimension-independent for this basis function. Asserting this
    directly (rather than a convergence ratio) catches sign errors and
    constant offsets that a ratio-only test would miss.
    """
    shape = (size,) * dim
    h = L / (size + 1)

    u_star = _sin_product_grid(shape)
    rhs = dim * (jnp.pi / L) ** 2 * u_star

    u = solve_direct(rhs, h)
    err = float(jnp.max(jnp.abs(u - u_star)))
    predicted = (jnp.pi / L) ** 2 * h**2 / 12
    assert (
        err < 3 * predicted
    ), f"dim={dim} size={size}: err={err:.3e}, predicted<{3 * predicted:.3e}"


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_solve_direct_mms_second_order(dim):
    """Use the continuous Laplacian as RHS; error halves by ~4 when h halves."""
    sizes = [16, 32] if dim == 3 else [32, 64]

    errors = []
    for size in sizes:
        shape = (size,) * dim
        h = L / (size + 1)
        u_star = _sin_product_grid(shape)
        rhs = dim * (jnp.pi / L) ** 2 * u_star

        u = solve_direct(rhs, h)
        errors.append(float(jnp.max(jnp.abs(u - u_star))))

    ratio = errors[0] / errors[1]
    assert 3.5 < ratio < 4.5, f"dim={dim}: error ratio {ratio}, errors {errors}"
