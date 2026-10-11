"""Per-axis sample generators and dataset helpers.

Truncation axis: `generate_truncation` + `materialise_truncation`.
Discretization axis: `generate_discretization` + `make_gen_at_N` (for a
per-N vmap closure) + `materialise_discretization` (buckets by N, vmaps
per bucket, scatters back into the original enumeration order).

All generators are pure and bitwise deterministic for a fixed `seed`
The batching logic lives in `training.py` — one jitted
step per axis that either regenerates per step or indexes into cached
arrays.
"""

import jax
import jax.numpy as jnp
import numpy as np

from poisson_nd.dst import solve_direct
from poisson_nd.dst import dst_resample
from poisson_nd.multigrid import build_multigrid_iterator, prolongate
from poisson_nd.rhs import generate_rhs


def generate_truncation(seed, K, shape, *, num_modes=15, domain_extent=1.0):
    """Single-sample generator for the truncation axis.

    Returns (rhs, u) at `shape`, where `u` is the MG iterator's solution
    after `K` V-cycles starting from zero.
    """
    rhs = generate_rhs(seed, shape, num_modes=num_modes, domain_extent=domain_extent)
    h = domain_extent / (shape[0] + 1)
    iterator = build_multigrid_iterator(shape)
    u = iterator(rhs, h, K)
    return rhs, u


def materialise_truncation(seeds, xis, cfg, shape, *, chunk_size=None, cpu_device=None):
    """Pre-generate the full truncation dataset (fs, us).

    Two code paths:

    - **Default / 1D / 2D**: `jax.lax.map(body, ..., batch_size=chunk)`
      assembles the full `(M, *shape)` output on the default device
      (GPU). Fast but peaks at `M * prod(shape) * 4 B` of GPU memory,
      which OOMs at 3D / large `M`.
    - **3D with `cpu_device` set**: Python-level chunking, transferring
      each chunk's output to CPU immediately before accumulating, so
      the full `(M, *shape)` cache only lives on CPU. Avoids the OOM.

    The 3D chunk size defaults to 100 (see `chunk_size` parameter), which
    fits GPU comfortably and cuts Python-loop overhead to negligible.
    """
    M = int(len(seeds))

    def body(pair):
        s, k = pair
        return generate_truncation(
            s,
            k,
            shape,
            num_modes=cfg.num_modes,
            domain_extent=cfg.domain_extent,
        )

    use_cpu_loop = cpu_device is not None and cfg.dim == 3
    if use_cpu_loop:
        # 3D path: chunk in Python, transfer to CPU per chunk.
        default_chunk = chunk_size if chunk_size is not None else 100
        chunk = max(1, min(default_chunk, M))
        vmapped = jax.vmap(body)
        fs_chunks, us_chunks = [], []
        for start in range(0, M, chunk):
            end = min(start + chunk, M)
            fs_c, us_c = vmapped((seeds[start:end], xis[start:end]))
            fs_chunks.append(jax.device_put(fs_c, cpu_device))
            us_chunks.append(jax.device_put(us_c, cpu_device))
        fs = jnp.concatenate(fs_chunks, axis=0)
        us = jnp.concatenate(us_chunks, axis=0)
        return fs, us

    # Default (1D / 2D / cpu_device=None): existing fast path.
    chunk = chunk_size if chunk_size is not None else cfg.batch_size
    chunk = max(1, min(chunk, M))
    fs, us = jax.lax.map(body, (seeds, xis), batch_size=chunk)
    return fs, us


def generate_discretization(
    seed, N, target_shape, *, num_modes=15, domain_extent=1.0, dst_interpolation=False
):
    """Single-sample generator for the discretization axis.

    Solves -Δu = f at the native resolution `N` (DST direct solver), then
    resamples both the RHS and the solution to `target_shape`. Default is the
    linear `prolongate` (bidirectional via `map_coordinates`); with
    `dst_interpolation=True` the resampling is done in the DST-I sine basis,
    which is spectrally exact and introduces no interpolation error.

    `N` must be a Python int — it's a trace-time constant used to build
    the native shape tuple.
    """
    target_shape = tuple(target_shape)
    native_shape = (N,) * len(target_shape)
    rhs_native = generate_rhs(
        seed, native_shape, num_modes=num_modes, domain_extent=domain_extent
    )
    h = domain_extent / (N + 1)
    u_native = solve_direct(rhs_native, h)
    resample = dst_resample if dst_interpolation else prolongate
    f_star = resample(rhs_native, target_shape)
    u_star = resample(u_native, target_shape)
    return f_star, u_star


def make_gen_at_N(N, target_shape, num_modes, domain_extent, dst_interpolation=False):
    """Return a vmap'd single-seed generator with `N` fixed at trace time."""

    def _fn(seed):
        return generate_discretization(
            seed,
            N,
            target_shape,
            num_modes=num_modes,
            domain_extent=domain_extent,
            dst_interpolation=dst_interpolation,
        )

    return jax.vmap(_fn)


def materialise_discretization(
    seeds, Ns, cfg, target_shape, *, chunk_size=None, cpu_device=None
):
    """Pre-generate the full discretization dataset at `target_shape`.

    Buckets samples by N (Python level — buckets are computed once at
    startup), vmaps each bucket, then scatters the per-bucket tensors
    back into the original enumeration order.

    Two code paths (mirroring `materialise_truncation`):

    - **Default / 1D / 2D**: `jax.lax.map` per bucket, then scatter into
      pre-allocated `(M, *target_shape)` GPU arrays. Fast but peaks at
      `M * prod(target_shape) * 4 B` of GPU memory.
    - **3D with `cpu_device` set**: Python-level chunking per bucket,
      transferring each chunk's output to CPU immediately; scatter
      happens on CPU arrays (`.at[].set()` is device-transparent).

    Returns `(fs, us)` with shapes `(M, *target_shape)` each.
    """
    target_shape = tuple(target_shape)
    M = int(len(seeds))
    Ns_np = np.asarray(Ns)
    unique_Ns = sorted(int(x) for x in np.unique(Ns_np))

    use_cpu_loop = cpu_device is not None and cfg.dim == 3
    default_chunk_fallback = 100 if use_cpu_loop else cfg.batch_size
    default_chunk = chunk_size if chunk_size is not None else default_chunk_fallback

    # Pre-allocate the big output arrays on the right device up front.
    if use_cpu_loop:
        fs = jax.device_put(jnp.zeros((M, *target_shape)), cpu_device)
        us = jax.device_put(jnp.zeros((M, *target_shape)), cpu_device)
    else:
        fs = jnp.zeros((M, *target_shape))
        us = jnp.zeros((M, *target_shape))

    for N in unique_Ns:
        bucket_indices = np.where(Ns_np == N)[0]
        bucket_seeds = seeds[bucket_indices]
        chunk = max(1, min(default_chunk, len(bucket_indices)))

        def body(seed, N=N):
            return generate_discretization(
                seed,
                N,
                target_shape,
                num_modes=cfg.num_modes,
                domain_extent=cfg.domain_extent,
                dst_interpolation=getattr(cfg, "dst_interpolation", False),
            )

        if use_cpu_loop:
            # 3D path: per-bucket Python chunk loop, transfer each chunk to CPU.
            B = len(bucket_indices)
            vmapped = jax.vmap(body)
            bucket_fs_chunks, bucket_us_chunks = [], []
            for start in range(0, B, chunk):
                end = min(start + chunk, B)
                ch_fs, ch_us = vmapped(bucket_seeds[start:end])
                bucket_fs_chunks.append(jax.device_put(ch_fs, cpu_device))
                bucket_us_chunks.append(jax.device_put(ch_us, cpu_device))
            fs_bucket = jnp.concatenate(bucket_fs_chunks, axis=0)
            us_bucket = jnp.concatenate(bucket_us_chunks, axis=0)
        else:
            fs_bucket, us_bucket = jax.lax.map(body, bucket_seeds, batch_size=chunk)

        idx_j = jnp.asarray(bucket_indices)
        fs = fs.at[idx_j].set(fs_bucket)
        us = us.at[idx_j].set(us_bucket)

    return fs, us
