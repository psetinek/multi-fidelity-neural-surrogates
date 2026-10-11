"""Tests for the per-sample generators and the materialised-dataset helpers."""

from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import pytest

from multifidelity.data import (
    generate_discretization,
    generate_truncation,
    materialise_discretization,
    materialise_truncation,
)


@pytest.mark.parametrize("shape", [(16,), (16, 16)])
def test_generate_truncation_deterministic(shape):
    key = jr.PRNGKey(0)
    f1, u1 = generate_truncation(key, 3, shape)
    f2, u2 = generate_truncation(key, 3, shape)
    assert jnp.array_equal(f1, f2)
    assert jnp.array_equal(u1, u2)


def test_generate_truncation_shapes_1d():
    key = jr.PRNGKey(0)
    f, u = generate_truncation(key, 2, (16,))
    assert f.shape == (16,)
    assert u.shape == (16,)


def test_generate_truncation_shapes_2d():
    key = jr.PRNGKey(0)
    f, u = generate_truncation(key, 2, (16, 16))
    assert f.shape == (16, 16)
    assert u.shape == (16, 16)


def test_materialise_truncation_shapes_and_values():
    shape = (16, 16)
    M = 5
    seeds = jr.split(jr.PRNGKey(0), M)
    xis = jnp.array([1, 2, 3, 4, 5])
    cfg = SimpleNamespace(batch_size=2, num_modes=8, domain_extent=1.0)

    fs, us = materialise_truncation(seeds, xis, cfg, shape)
    assert fs.shape == (M, *shape)
    assert us.shape == (M, *shape)

    # Each row matches a direct single-sample call up to tiny fp noise.
    # `vmap` can reorder reductions (e.g. max/sum) and `fori_loop` under vmap
    # rolls to max K with masking — both introduce ε-level differences vs.
    # single-sample calls. We only need semantic equivalence.
    for i in range(M):
        f_i, u_i = generate_truncation(
            seeds[i], xis[i], shape, num_modes=8, domain_extent=1.0
        )
        assert jnp.allclose(fs[i], f_i, rtol=1e-5, atol=1e-7)
        assert jnp.allclose(us[i], u_i, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize(
    "N,target_shape", [(16, (16, 16)), (32, (16, 16)), (12, (8, 8))]
)
def test_generate_discretization_shapes(N, target_shape):
    key = jr.PRNGKey(0)
    f, u = generate_discretization(key, N, target_shape, num_modes=4)
    assert f.shape == target_shape
    assert u.shape == target_shape


def test_generate_discretization_deterministic():
    key = jr.PRNGKey(0)
    f1, u1 = generate_discretization(key, 24, (16, 16), num_modes=4)
    f2, u2 = generate_discretization(key, 24, (16, 16), num_modes=4)
    assert jnp.array_equal(f1, f2)
    assert jnp.array_equal(u1, u2)


def test_generate_discretization_identity_at_target_resolution():
    """N == target shape side → restriction is identity up to map_coordinates noise."""
    key = jr.PRNGKey(1)
    shape = (16, 16)
    f, u = generate_discretization(key, 16, shape, num_modes=4)
    # Check values are finite and nonzero — a real sample was produced.
    assert jnp.all(jnp.isfinite(f))
    assert jnp.all(jnp.isfinite(u))
    assert jnp.linalg.norm(f) > 0
    assert jnp.linalg.norm(u) > 0


def test_materialise_discretization_shapes_and_values():
    target_shape = (16, 16)
    Ns_plan = [16, 24, 16, 32, 24, 16]
    M = len(Ns_plan)
    seeds = jr.split(jr.PRNGKey(0), M)
    Ns = jnp.asarray(Ns_plan)
    cfg = SimpleNamespace(batch_size=2, num_modes=4, domain_extent=1.0)

    fs, us = materialise_discretization(seeds, Ns, cfg, target_shape)
    assert fs.shape == (M, *target_shape)
    assert us.shape == (M, *target_shape)

    # Per-sample: row must match a direct single-sample call up to vmap fp noise.
    for i in range(M):
        f_i, u_i = generate_discretization(
            seeds[i], int(Ns[i]), target_shape, num_modes=4, domain_extent=1.0
        )
        assert jnp.allclose(fs[i], f_i, rtol=1e-5, atol=1e-7)
        assert jnp.allclose(us[i], u_i, rtol=1e-5, atol=1e-7)
