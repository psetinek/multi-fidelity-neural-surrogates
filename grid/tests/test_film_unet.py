"""Dim-agnostic FiLM-UNet smoke tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from architectures import FiLMUNet


@pytest.mark.parametrize("ndim,size", [(1, 32), (2, 32), (3, 16)])
def test_forward_shape(ndim, size):
    model = FiLMUNet(
        num_spatial_dims=ndim,
        t_scale=60.0,
        num_levels=4,
        base_channels=8,
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((1,) + (size,) * ndim)
    out = model(x, 10.0)
    assert out.shape == (1,) + (size,) * ndim


def test_output_varies_with_t():
    model = FiLMUNet(
        num_spatial_dims=2,
        t_scale=60.0,
        num_levels=3,
        base_channels=8,
        key=jr.PRNGKey(1),
    )
    x = jnp.ones((1, 16, 16))
    a = model(x, 1.0)
    b = model(x, 30.0)
    assert jnp.linalg.norm(a - b) > 1e-6


def test_vmap_over_batch():
    model = FiLMUNet(
        num_spatial_dims=2,
        t_scale=60.0,
        num_levels=3,
        base_channels=8,
        key=jr.PRNGKey(2),
    )
    xs = jnp.ones((4, 1, 16, 16))
    ts = jnp.array([1.0, 10.0, 20.0, 40.0])
    out = jax.vmap(model)(xs, ts)
    assert out.shape == (4, 1, 16, 16)


def test_param_count_grows_with_num_levels():
    def params(num_levels):
        m = FiLMUNet(
            num_spatial_dims=2,
            t_scale=60.0,
            num_levels=num_levels,
            base_channels=8,
            key=jr.PRNGKey(3),
        )
        return sum(p.size for p in jax.tree.leaves(eqx.filter(m, eqx.is_array)))

    assert params(4) > params(3) > params(2)


def test_custom_activation_changes_output():
    x = jnp.ones((1, 16, 16))

    m_silu = FiLMUNet(
        num_spatial_dims=2,
        t_scale=60.0,
        num_levels=3,
        base_channels=8,
        activation=jax.nn.silu,
        key=jr.PRNGKey(4),
    )
    m_gelu = FiLMUNet(
        num_spatial_dims=2,
        t_scale=60.0,
        num_levels=3,
        base_channels=8,
        activation=jax.nn.gelu,
        key=jr.PRNGKey(4),
    )
    assert jnp.linalg.norm(m_silu(x, 10.0) - m_gelu(x, 10.0)) > 1e-6


def test_num_blocks_per_level_increases_params():
    def params(nb):
        m = FiLMUNet(
            num_spatial_dims=2,
            t_scale=60.0,
            num_levels=3,
            base_channels=8,
            num_blocks_per_level=nb,
            key=jr.PRNGKey(5),
        )
        return sum(p.size for p in jax.tree.leaves(eqx.filter(m, eqx.is_array)))

    assert params(3) > params(2) > params(1)


def test_t_scale_is_required():
    with pytest.raises(TypeError):
        FiLMUNet(num_spatial_dims=2, key=jr.PRNGKey(6))  # type: ignore[call-arg]
