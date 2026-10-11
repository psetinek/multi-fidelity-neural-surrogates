"""Dimension-agnostic FiLM-conditioned UNet."""

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

from architectures.conditioning import ConditioningNetwork
from architectures.film import apply_film, split_film_params


class FiLMConvBlock(eqx.Module):
    """Conv → GroupNorm → FiLM → activation."""

    conv: eqx.nn.Conv
    norm: eqx.nn.GroupNorm
    film_proj: eqx.nn.Linear
    activation: Callable = eqx.field(static=True)

    def __init__(
        self,
        num_spatial_dims,
        in_channels,
        out_channels,
        cond_dim,
        *,
        activation=jax.nn.silu,
        key,
    ):
        k_conv, k_film = jax.random.split(key)
        self.conv = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=k_conv,
        )
        self.norm = eqx.nn.GroupNorm(min(8, out_channels), out_channels)
        self.film_proj = eqx.nn.Linear(cond_dim, 2 * out_channels, key=k_film)
        self.activation = activation

    def __call__(self, x, cond):
        h = self.norm(self.conv(x))
        scale, shift = split_film_params(self.film_proj(cond))
        h = apply_film(h, scale, shift)
        return self.activation(h)


class FiLMBlockStack(eqx.Module):
    """Stack of N FiLMConvBlocks; first one changes channels, rest preserve."""

    blocks: list

    def __init__(
        self,
        num_spatial_dims,
        in_channels,
        out_channels,
        cond_dim,
        *,
        num_blocks=2,
        activation=jax.nn.silu,
        key,
    ):
        keys = jax.random.split(key, num_blocks)
        self.blocks = [
            FiLMConvBlock(
                num_spatial_dims,
                in_channels if i == 0 else out_channels,
                out_channels,
                cond_dim,
                activation=activation,
                key=keys[i],
            )
            for i in range(num_blocks)
        ]

    def __call__(self, x, cond):
        for block in self.blocks:
            x = block(x, cond)
        return x


class Downsample(eqx.Module):
    """Strided 3-kernel conv halving every spatial dimension."""

    conv: eqx.nn.Conv

    def __init__(self, num_spatial_dims, channels, *, key):
        self.conv = eqx.nn.Conv(
            num_spatial_dims,
            channels,
            channels,
            kernel_size=3,
            stride=2,
            padding=1,
            key=key,
        )

    def __call__(self, x):
        return self.conv(x)


class Upsample(eqx.Module):
    """Nearest-neighbor ×2 along each spatial axis, then a 3-kernel conv."""

    conv: eqx.nn.Conv
    num_spatial_dims: int = eqx.field(static=True)

    def __init__(self, num_spatial_dims, in_channels, out_channels, *, key):
        self.conv = eqx.nn.Conv(
            num_spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            key=key,
        )
        self.num_spatial_dims = num_spatial_dims

    def __call__(self, x):
        for axis_offset in range(self.num_spatial_dims):
            x = jnp.repeat(x, 2, axis=-(axis_offset + 1))
        return self.conv(x)


class FiLMUNet(eqx.Module):
    """FiLM-conditioned UNet for any spatial dimension.

    Channel progression across levels:
        c_i = base_channels * channel_multiplier ** i   for i in 0..num_levels-1
    Spatial resolution halves from one level to the next, so the input
    must be divisible by 2**(num_levels - 1) along every spatial axis.

    Conditioning is a scalar `t`. See `ConditioningNetwork`; the meaning
    of `t` is up to the caller.
    """

    cond_net: ConditioningNetwork
    lift: FiLMBlockStack
    down_blocks: list
    enc_blocks: list
    up_blocks: list
    dec_blocks: list
    proj: eqx.nn.Conv

    def __init__(
        self,
        num_spatial_dims,
        *,
        t_scale,
        num_levels=4,
        base_channels=64,
        channel_multiplier=2,
        num_blocks_per_level=2,
        cond_dim=256,
        activation=jax.nn.silu,
        key,
    ):
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")

        channels = [base_channels * channel_multiplier**i for i in range(num_levels)]
        num_transitions = num_levels - 1

        total_keys = 1 + 1 + 2 * num_transitions + 2 * num_transitions + 1
        keys = iter(jax.random.split(key, total_keys))

        self.cond_net = ConditioningNetwork(
            t_scale=t_scale, cond_dim=cond_dim, key=next(keys)
        )

        self.lift = FiLMBlockStack(
            num_spatial_dims,
            1,
            channels[0],
            cond_dim,
            num_blocks=num_blocks_per_level,
            activation=activation,
            key=next(keys),
        )

        self.down_blocks = []
        self.enc_blocks = []
        for i in range(num_transitions):
            self.down_blocks.append(
                Downsample(num_spatial_dims, channels[i], key=next(keys))
            )
            self.enc_blocks.append(
                FiLMBlockStack(
                    num_spatial_dims,
                    channels[i],
                    channels[i + 1],
                    cond_dim,
                    num_blocks=num_blocks_per_level,
                    activation=activation,
                    key=next(keys),
                )
            )

        self.up_blocks = []
        self.dec_blocks = []
        for i in reversed(range(num_transitions)):
            self.up_blocks.append(
                Upsample(
                    num_spatial_dims,
                    channels[i + 1],
                    channels[i],
                    key=next(keys),
                )
            )
            self.dec_blocks.append(
                FiLMBlockStack(
                    num_spatial_dims,
                    2 * channels[i],
                    channels[i],
                    cond_dim,
                    num_blocks=num_blocks_per_level,
                    activation=activation,
                    key=next(keys),
                )
            )

        self.proj = eqx.nn.Conv(
            num_spatial_dims,
            channels[0],
            1,
            kernel_size=1,
            key=next(keys),
        )

    def __call__(self, x, t):
        cond = self.cond_net(t)

        h = self.lift(x, cond)
        skips = []

        for down, enc in zip(self.down_blocks, self.enc_blocks):
            skips.append(h)
            h = enc(down(h), cond)

        for up, dec, skip in zip(self.up_blocks, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = jnp.concatenate([h, skip], axis=0)
            h = dec(h, cond)

        return self.proj(h)
