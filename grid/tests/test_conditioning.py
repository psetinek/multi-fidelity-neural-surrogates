"""Conditioning network sanity checks."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from architectures import ConditioningNetwork


def test_output_shape():
    net = ConditioningNetwork(t_scale=60.0, key=jr.PRNGKey(0))
    out = net(30.0)
    assert out.shape == (256,)


def test_different_t_produces_different_outputs():
    net = ConditioningNetwork(t_scale=60.0, key=jr.PRNGKey(1))
    a = net(10.0)
    b = net(50.0)
    assert jnp.linalg.norm(a - b) > 1e-6


def test_vmap_over_batched_t():
    net = ConditioningNetwork(t_scale=60.0, key=jr.PRNGKey(2))
    ts = jnp.array([1.0, 10.0, 20.0, 50.0])
    out = jax.vmap(net)(ts)
    assert out.shape == (4, 256)


def test_t_scale_rescaling_changes_embedding():
    """Different t_scale values produce different embeddings for the same t."""
    net_a = ConditioningNetwork(t_scale=60.0, key=jr.PRNGKey(3))
    net_b = ConditioningNetwork(t_scale=1.0, key=jr.PRNGKey(3))
    # Same key, same t, different scale -> different embedding
    assert jnp.linalg.norm(net_a(10.0) - net_b(10.0)) > 1e-6


def test_custom_cond_dim():
    net = ConditioningNetwork(t_scale=10.0, cond_dim=64, key=jr.PRNGKey(4))
    assert net(5.0).shape == (64,)


def test_t_scale_is_required():
    with pytest.raises(TypeError):
        ConditioningNetwork(key=jr.PRNGKey(5))  # type: ignore[call-arg]
