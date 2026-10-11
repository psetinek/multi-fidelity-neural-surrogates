"""Dimension-agnostic FiLM-conditioned FNO.

FNOs assume periodic domains. To operate on Dirichlet fields the model
zero-pads to a larger domain, runs the spectral convolutions there, and
crops back.
"""

from typing import Callable, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from pdequinox.conv import PointwiseLinearConv, SpectralConv

from architectures.conditioning import ConditioningNetwork
from architectures.film import apply_film, split_film_params


class FiLMSpectralBlock(eqx.Module):
    """SpectralConv + pointwise bypass + FiLM + activation."""

    spectral_conv: SpectralConv
    bypass_conv: PointwiseLinearConv
    film_proj: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims,
        in_channels,
        out_channels,
        cond_dim,
        *,
        num_modes,
        activation=jax.nn.gelu,
        key,
    ):
        k_spec, k_bypass, k_film = jax.random.split(key, 3)
        self.spectral_conv = SpectralConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            num_modes=num_modes,
            key=k_spec,
        )
        self.bypass_conv = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            key=k_bypass,
        )
        self.film_proj = eqx.nn.Linear(cond_dim, 2 * out_channels, key=k_film)
        self.activation = activation

    def __call__(self, x, cond):
        h = self.spectral_conv(x) + self.bypass_conv(x)
        scale, shift = split_film_params(self.film_proj(cond))
        h = apply_film(h, scale, shift)
        return self.activation(h)


class FiLMFNO(eqx.Module):
    """FiLM-conditioned Fourier Neural Operator for any spatial dimension.

    Topology:
        lifting:   1 → hidden_channels (pointwise)
        N blocks:  SpectralConv + bypass + FiLM + activation
        projection: hidden_channels → 1 (pointwise)

    Input is zero-padded by `pad_ratio` before the lifting and cropped
    after the projection, so the operator's "periodic" interior sees the
    original field separated from its periodic copies.
    """

    cond_net: ConditioningNetwork
    lifting: PointwiseLinearConv
    blocks: list
    projection: PointwiseLinearConv
    num_spatial_dims: int = eqx.field(static=True)
    pad_ratio: float = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims,
        *,
        t_scale,
        num_blocks=4,
        hidden_channels=64,
        num_modes: Union[int, tuple] = 12,
        pad_ratio=1.5,
        cond_dim=256,
        activation=jax.nn.gelu,
        key,
    ):
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be >= 1, got {num_blocks}")

        total_keys = 1 + 1 + num_blocks + 1
        keys = iter(jax.random.split(key, total_keys))

        self.cond_net = ConditioningNetwork(
            t_scale=t_scale, cond_dim=cond_dim, key=next(keys)
        )

        self.lifting = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=1,
            out_channels=hidden_channels,
            key=next(keys),
        )

        self.blocks = [
            FiLMSpectralBlock(
                num_spatial_dims,
                hidden_channels,
                hidden_channels,
                cond_dim,
                num_modes=num_modes,
                activation=activation,
                key=next(keys),
            )
            for _ in range(num_blocks)
        ]

        self.projection = PointwiseLinearConv(
            num_spatial_dims=num_spatial_dims,
            in_channels=hidden_channels,
            out_channels=1,
            key=next(keys),
        )

        self.num_spatial_dims = num_spatial_dims
        self.pad_ratio = float(pad_ratio)

    def __call__(self, x, t):
        cond = self.cond_net(t)

        spatial = x.shape[-self.num_spatial_dims :]
        pads = tuple(int(round((self.pad_ratio - 1.0) * n / 2)) for n in spatial)

        pad_widths = ((0, 0),) + tuple((p, p) for p in pads)
        x = jnp.pad(x, pad_widths)

        x = self.lifting(x)
        for block in self.blocks:
            x = block(x, cond)
        x = self.projection(x)

        crop_slices = (slice(None),) + tuple(
            slice(p, -p) if p > 0 else slice(None) for p in pads
        )
        return x[crop_slices]
