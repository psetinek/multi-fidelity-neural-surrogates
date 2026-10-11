"""Dim-agnostic FiLM-FNO smoke tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from architectures import FiLMFNO


@pytest.mark.parametrize("ndim,size", [(1, 32), (2, 32), (3, 16)])
def test_forward_shape(ndim, size):
    model = FiLMFNO(
        num_spatial_dims=ndim,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        key=jr.PRNGKey(0),
    )
    x = jnp.ones((1,) + (size,) * ndim)
    out = model(x, 10.0)
    assert out.shape == (1,) + (size,) * ndim


def test_output_varies_with_t():
    model = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        key=jr.PRNGKey(1),
    )
    x = jnp.ones((1, 16, 16))
    a = model(x, 1.0)
    b = model(x, 30.0)
    assert jnp.linalg.norm(a - b) > 1e-6


def test_vmap_over_batch():
    model = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        key=jr.PRNGKey(2),
    )
    xs = jnp.ones((4, 1, 16, 16))
    ts = jnp.array([1.0, 10.0, 20.0, 40.0])
    out = jax.vmap(model)(xs, ts)
    assert out.shape == (4, 1, 16, 16)


def test_param_count_grows_with_num_blocks():
    def params(nb):
        m = FiLMFNO(
            num_spatial_dims=2,
            t_scale=60.0,
            num_blocks=nb,
            hidden_channels=16,
            num_modes=4,
            key=jr.PRNGKey(3),
        )
        return sum(p.size for p in jax.tree.leaves(eqx.filter(m, eqx.is_array)))

    assert params(6) > params(4) > params(2)


def test_custom_activation_changes_output():
    x = jnp.ones((1, 16, 16))

    m_gelu = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        activation=jax.nn.gelu,
        key=jr.PRNGKey(4),
    )
    m_silu = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        activation=jax.nn.silu,
        key=jr.PRNGKey(4),
    )
    assert jnp.linalg.norm(m_gelu(x, 10.0) - m_silu(x, 10.0)) > 1e-6


def test_pad_ratio_changes_output():
    """Different pad_ratio produces different outputs (since padded size differs)."""
    x = jnp.ones((1, 32, 32))
    m1 = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        pad_ratio=1.5,
        key=jr.PRNGKey(5),
    )
    m2 = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=4,
        pad_ratio=2.0,
        key=jr.PRNGKey(5),
    )
    assert jnp.linalg.norm(m1(x, 10.0) - m2(x, 10.0)) > 1e-6


def test_num_modes_as_tuple():
    """SpectralConv accepts num_modes as per-axis tuple."""
    model = FiLMFNO(
        num_spatial_dims=2,
        t_scale=60.0,
        num_blocks=2,
        hidden_channels=16,
        num_modes=(3, 5),
        key=jr.PRNGKey(6),
    )
    x = jnp.ones((1, 16, 16))
    assert model(x, 10.0).shape == (1, 16, 16)


def test_t_scale_is_required():
    with pytest.raises(TypeError):
        FiLMFNO(num_spatial_dims=2, key=jr.PRNGKey(7))  # type: ignore[call-arg]
