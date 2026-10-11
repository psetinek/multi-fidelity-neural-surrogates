"""Tests for whitening stats, disk IO, and whiten/dewhiten."""

from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr

from multifidelity.whitening import (
    WhiteningStats,
    compute_stats,
    dewhiten,
    load_stats,
    save_stats,
    warmup_stats_discretization,
    whiten,
    whiten_target,
)


def _random_dataset(shape=(16, 16), n=8, key=jr.PRNGKey(0)):
    k1, k2 = jr.split(key)
    fs = jr.normal(k1, (n, *shape))
    us = jr.normal(k2, (n, *shape))
    return fs, us


def test_stats_shape():
    fs, us = _random_dataset(shape=(16, 16), n=8)
    stats = compute_stats(fs, us)
    assert stats.mean_f.shape == (16, 16)
    assert stats.std_f.shape == (16, 16)
    assert stats.mean_u.shape == (16, 16)
    assert stats.std_u.shape == (16, 16)


def test_save_load_roundtrip(tmp_path):
    fs, us = _random_dataset(shape=(8, 8), n=4)
    stats = compute_stats(fs, us)
    path = tmp_path / "w.npz"
    save_stats(path, stats)
    loaded = load_stats(path)
    for name in WhiteningStats._fields:
        assert jnp.allclose(getattr(stats, name), getattr(loaded, name))


def test_whiten_dewhiten_inverse():
    fs, us = _random_dataset(shape=(8, 8), n=4)
    stats = compute_stats(fs, us)
    # Whitened-then-dewhitened should match the original up to eps.
    roundtrip = jnp.stack([dewhiten(whiten_target(u, stats), stats) for u in us])
    assert jnp.allclose(roundtrip, us, atol=1e-5)


def test_whiten_input_zero_mean_unit_std():
    fs, us = _random_dataset(shape=(16, 16), n=32)
    stats = compute_stats(fs, us)
    # Whitened samples on training data should be roughly zero-mean unit-std.
    whitened = jnp.stack([whiten(f, stats) for f in fs])
    assert jnp.abs(jnp.mean(whitened)) < 1e-4
    # Std should be close to 1 (not exactly, because of per-pixel stats on finite samples).
    assert jnp.abs(jnp.std(whitened) - 1.0) < 0.2


def test_warmup_stats_discretization_mixed_ns():
    """Warmup over samples with mixed native N values produces finite per-pixel stats."""
    target_shape = (16, 16)
    Ns_plan = [16, 24, 16, 32, 24, 16, 32, 16]
    seeds = jr.split(jr.PRNGKey(0), len(Ns_plan))
    Ns = jnp.asarray(Ns_plan)
    cfg = SimpleNamespace(
        batch_size=2, num_modes=4, domain_extent=1.0, num_whitening_samples=6
    )
    stats = warmup_stats_discretization(seeds, Ns, cfg, target_shape, key=jr.PRNGKey(1))
    assert stats.mean_f.shape == target_shape
    assert stats.std_u.shape == target_shape
    assert jnp.all(jnp.isfinite(stats.mean_f))
    assert jnp.all(jnp.isfinite(stats.std_f))
    assert jnp.all(jnp.isfinite(stats.mean_u))
    assert jnp.all(jnp.isfinite(stats.std_u))
