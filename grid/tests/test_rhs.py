"""RHS generation: shape, normalization, and Hann boundary decay."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from poisson_nd.rhs import generate_rhs


@pytest.mark.parametrize("shape", [(32,), (16, 16), (8, 8, 8)])
def test_rhs_shape(shape):
    f = generate_rhs(jr.PRNGKey(0), shape)
    assert f.shape == shape


@pytest.mark.parametrize("shape", [(32,), (16, 16), (8, 8, 8)])
def test_rhs_max_abs_is_one(shape):
    f = generate_rhs(jr.PRNGKey(1), shape)
    assert jnp.isclose(jnp.max(jnp.abs(f)), 1.0)


@pytest.mark.parametrize("shape", [(32,), (16, 16), (8, 8, 8)])
def test_rhs_decays_near_boundary(shape):
    """The Hann window vanishes at the domain edges, so the first interior
    row should be much smaller than the peak."""
    f = generate_rhs(jr.PRNGKey(2), shape)

    d = len(shape)
    peak = float(jnp.max(jnp.abs(f)))
    # Take the first and last slab along each axis and compare to the peak.
    for a in range(d):
        first = jnp.take(f, jnp.array([0]), axis=a)
        last = jnp.take(f, jnp.array([shape[a] - 1]), axis=a)
        assert float(jnp.max(jnp.abs(first))) < 0.3 * peak
        assert float(jnp.max(jnp.abs(last))) < 0.3 * peak


def test_rhs_reproducible_with_same_key():
    shape = (8, 8)
    f1 = generate_rhs(jr.PRNGKey(42), shape)
    f2 = generate_rhs(jr.PRNGKey(42), shape)
    assert jnp.allclose(f1, f2)
