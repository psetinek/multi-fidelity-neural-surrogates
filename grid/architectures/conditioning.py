"""Scalar fidelity signal → conditioning vector.

The sinusoidal embedding plus two-layer MLP is the standard pattern used
by diffusion-style conditioning. Nothing in the mechanism ties the
signal to a particular semantic meaning — the interpretation is the
caller's responsibility.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class SinusoidalEmbedding(eqx.Module):
    """Scalar t → sinusoidal features → learned linear projection.

    Frequencies are log-spaced in [0.1, 10] so the embedding spans slow
    features (coarse fidelity distinctions) and fast features (fine
    distinctions near the upper end of the signal range). The scalar
    is normalized by `t_scale`; the embedding is most discriminative
    when `t / t_scale` is in roughly [0, 1].

    Note: `freqs` is a regular float leaf, so it is trained along with
    the other weights (the training loop optimizes every
    `eqx.is_inexact_array` leaf). The log-spacing only describes its
    initialization.
    """

    freqs: jax.Array
    linear: eqx.nn.Linear
    t_scale: float = eqx.field(static=True)

    def __init__(self, embed_dim, t_scale, num_freqs=16, *, key):
        self.freqs = jnp.exp(jnp.linspace(jnp.log(0.1), jnp.log(10.0), num_freqs))
        self.linear = eqx.nn.Linear(2 * num_freqs, embed_dim, key=key)
        self.t_scale = float(t_scale)

    def __call__(self, t):
        t_norm = t / self.t_scale
        phases = t_norm * self.freqs
        raw = jnp.concatenate([jnp.sin(phases), jnp.cos(phases)])
        return self.linear(raw)


class ConditioningNetwork(eqx.Module):
    """Sinusoidal embedding → 2-layer MLP → conditioning vector."""

    embed: SinusoidalEmbedding
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, *, t_scale, cond_dim=256, num_freqs=16, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.embed = SinusoidalEmbedding(
            cond_dim, t_scale=t_scale, num_freqs=num_freqs, key=k1
        )
        self.linear1 = eqx.nn.Linear(cond_dim, cond_dim, key=k2)
        self.linear2 = eqx.nn.Linear(cond_dim, cond_dim, key=k3)

    def __call__(self, t):
        h = jax.nn.silu(self.embed(t))
        h = jax.nn.silu(self.linear1(h))
        return self.linear2(h)
