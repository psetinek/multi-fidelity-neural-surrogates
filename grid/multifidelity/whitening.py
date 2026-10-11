"""Per-pixel whitening statistics with procedural warmup subsampling.

Streaming per-step data gen means the full enumerated list is too large to
traverse for statistics (3D × millions of samples). The warmup pass draws
`num_whitening_samples` indices, regenerates those samples procedurally,
and computes per-pixel mean/std on the stacked arrays. Stats are persisted
to `whitening.npz` next to the model so post-hoc evaluation can reuse them.

Two axis-specific entry points mirror the two axes' generators:
- `warmup_stats_truncation` — all samples at a single `shape`.
- `warmup_stats_discretization` — samples have varying native N; bucket-
  by-N, vmap each bucket at target_shape, concatenate before computing.
"""

from collections import namedtuple
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from multifidelity.data import generate_discretization, generate_truncation

EPS = 1e-8

WhiteningStats = namedtuple("WhiteningStats", "mean_f std_f mean_u std_u")


def compute_stats(fs, us) -> WhiteningStats:
    return WhiteningStats(
        mean_f=jnp.mean(fs, axis=0),
        std_f=jnp.std(fs, axis=0),
        mean_u=jnp.mean(us, axis=0),
        std_u=jnp.std(us, axis=0),
    )


def warmup_stats_truncation(seeds, xis, cfg, shape, *, key) -> WhiteningStats:
    """Subsample the enumerated list, procedurally generate, compute stats."""
    n = min(cfg.num_whitening_samples, len(seeds))
    idx = jr.choice(key, len(seeds), shape=(n,), replace=False)
    seeds_sub = seeds[idx]
    xis_sub = xis[idx]

    def body(pair):
        s, k = pair
        return generate_truncation(
            s, k, shape, num_modes=cfg.num_modes, domain_extent=cfg.domain_extent
        )

    chunk = min(cfg.batch_size, n)
    fs, us = jax.lax.map(body, (seeds_sub, xis_sub), batch_size=chunk)
    return compute_stats(fs, us)


def warmup_stats_discretization(seeds, Ns, cfg, target_shape, *, key) -> WhiteningStats:
    """Subsample, bucket by N, vmap per bucket, concatenate, compute stats."""
    target_shape = tuple(target_shape)
    n = min(cfg.num_whitening_samples, len(seeds))
    idx = jr.choice(key, len(seeds), shape=(n,), replace=False)
    seeds_sub = seeds[idx]
    Ns_sub_np = np.asarray(Ns[idx])
    unique_Ns = sorted(int(x) for x in np.unique(Ns_sub_np))

    chunks_f = []
    chunks_u = []
    for N in unique_Ns:
        bucket_indices = np.where(Ns_sub_np == N)[0]
        bucket_seeds = seeds_sub[bucket_indices]
        chunk = max(1, min(cfg.batch_size, len(bucket_indices)))

        def body(seed, N=N):
            return generate_discretization(
                seed,
                N,
                target_shape,
                num_modes=cfg.num_modes,
                domain_extent=cfg.domain_extent,
                dst_interpolation=getattr(cfg, "dst_interpolation", False),
            )

        fs_bucket, us_bucket = jax.lax.map(body, bucket_seeds, batch_size=chunk)
        chunks_f.append(fs_bucket)
        chunks_u.append(us_bucket)

    fs = jnp.concatenate(chunks_f, axis=0)
    us = jnp.concatenate(chunks_u, axis=0)
    return compute_stats(fs, us)


def save_stats(path, stats: WhiteningStats) -> None:
    path = Path(path)
    np.savez(
        path,
        mean_f=np.asarray(stats.mean_f),
        std_f=np.asarray(stats.std_f),
        mean_u=np.asarray(stats.mean_u),
        std_u=np.asarray(stats.std_u),
    )


def load_stats(path) -> WhiteningStats:
    data = np.load(path)
    return WhiteningStats(
        mean_f=jnp.asarray(data["mean_f"]),
        std_f=jnp.asarray(data["std_f"]),
        mean_u=jnp.asarray(data["mean_u"]),
        std_u=jnp.asarray(data["std_u"]),
    )


def whiten(rhs, stats: WhiteningStats):
    return (rhs - stats.mean_f) / (stats.std_f + EPS)


def whiten_target(u, stats: WhiteningStats):
    return (u - stats.mean_u) / (stats.std_u + EPS)


def dewhiten(u_norm, stats: WhiteningStats):
    return u_norm * (stats.std_u + EPS) + stats.mean_u
