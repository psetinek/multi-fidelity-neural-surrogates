"""Generic training loops for the scalar-conditioned surrogate.

Three entry points share one loss function and optimizer recipe
(`optax.adam` + warmup-cosine + `eqx.filter_jit` update):

- `train_streaming_truncation` — per step regenerates its batch via
  `generate_truncation` under `jax.vmap`; single jitted `train_step`.
- `train_streaming_discretization` — per step picks a bucket (∝ bucket
  size), batches from inside, and dispatches to the N-specific jitted
  `train_step`. One compiled kernel per unique N in the enumerated list.
- `train_cached` — indexes into a pre-materialised `(fs, us)` tensor;
  axis-agnostic.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from multifidelity.conditioning import transform_log
from multifidelity.data import generate_truncation, make_gen_at_N
from multifidelity.whitening import WhiteningStats, whiten, whiten_target


def _build_optimizer(cfg):
    schedule = optax.warmup_cosine_decay_schedule(
        0.0,
        cfg.learning_rate,
        cfg.num_warmup_steps,
        cfg.num_training_steps,
    )
    return optax.adam(schedule)


def _make_core_loss(stats: WhiteningStats, xi_max: float):
    """Core loss: whiten + log-condition + forward + MSE. Axis-agnostic."""

    def loss_fn(model, fs_b, us_b, xis_b):
        whitened_fs = jax.vmap(lambda f: whiten(f, stats))(fs_b)
        whitened_us = jax.vmap(lambda u: whiten_target(u, stats))(us_b)
        inputs = whitened_fs[:, None]  # (B, 1, *spatial)
        targets = whitened_us[:, None]  # (B, 1, *spatial)
        t = transform_log(xis_b, xi_max)  # (B,)
        pred = jax.vmap(model)(inputs, t)  # (B, 1, *spatial)
        return jnp.mean((pred - targets) ** 2)

    return loss_fn


def _run_loop(
    train_step, model, opt_state, draw_batch, cfg, *, times_out=None, loss_out=None
):
    """Training loop driver.

    If `times_out` is a list, append one per-step wall-clock measurement
    (seconds) to it. Adds a sync via `float(loss)` after each step so the
    timing reflects true GPU completion. Intended for diagnostic scripts
    only — the normal training path leaves it `None` so the extra sync
    doesn't slow a production run.

    If `loss_out` is a list, append one `(step, loss)` pair per logged step.
    Free: the `float(loss)` sync it needs already happens for the progress
    print, so nothing extra is forced onto the device.
    """
    import time as _time

    step = 0
    while step < cfg.num_training_steps:
        args = draw_batch(step)
        t0 = _time.perf_counter() if times_out is not None else None
        model, opt_state, loss = train_step(model, opt_state, *args)
        if times_out is not None:
            float(loss)  # force GPU completion
            times_out.append(_time.perf_counter() - t0)
        step += 1
        if step % cfg.log_every == 0 or step == cfg.num_training_steps:
            loss_value = float(loss)
            print(f"  step {step}/{cfg.num_training_steps}, loss {loss_value:.3e}")
            if loss_out is not None:
                loss_out.append((step, loss_value))
    return model


def train_streaming_truncation(
    model, seeds, xis, cfg, spec, stats, shape, *, key, times_out=None, loss_out=None
):
    """Procedural per-step generation for the truncation axis."""
    core_loss = _make_core_loss(stats, spec.xi_max)

    def _gen(seed, K):
        return generate_truncation(
            seed,
            K,
            shape,
            num_modes=cfg.num_modes,
            domain_extent=cfg.domain_extent,
        )

    gen_batched = jax.vmap(_gen)

    def loss_fn(model, seeds_b, xis_b):
        fs_b, us_b = gen_batched(seeds_b, xis_b)
        return core_loss(model, fs_b, us_b, xis_b)

    optimizer = _build_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(model, opt_state, seeds_b, xis_b):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, seeds_b, xis_b)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    M = len(seeds)
    batch_size = cfg.batch_size
    perm_state = {"key": key, "perm": None, "cursor": 0}

    def draw_batch(_step):
        if perm_state["perm"] is None or perm_state["cursor"] + batch_size > M:
            perm_state["key"], sub = jr.split(perm_state["key"])
            perm_state["perm"] = jr.permutation(sub, M)
            perm_state["cursor"] = 0
        start = perm_state["cursor"]
        idx = perm_state["perm"][start : start + batch_size]
        perm_state["cursor"] = start + batch_size
        return seeds[idx], xis[idx]

    return _run_loop(
        train_step,
        model,
        opt_state,
        draw_batch,
        cfg,
        times_out=times_out,
        loss_out=loss_out,
    )


def train_streaming_discretization(
    model, seeds, Ns, cfg, spec, stats, target_shape, *, key, loss_out=None
):
    """Per-step bucket-then-batch dispatch across N-specific jitted steps."""
    target_shape = tuple(target_shape)
    core_loss = _make_core_loss(stats, spec.xi_max)

    Ns_np = np.asarray(Ns)
    unique_Ns = sorted(int(x) for x in np.unique(Ns_np))
    buckets = {N: jnp.asarray(np.where(Ns_np == N)[0]) for N in unique_Ns}
    bucket_sizes = jnp.asarray(
        [int(buckets[N].shape[0]) for N in unique_Ns], dtype=jnp.float32
    )
    bucket_probs = bucket_sizes / jnp.sum(bucket_sizes)

    optimizer = _build_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def _make_train_step(N: int):
        gen_N = make_gen_at_N(
            N,
            target_shape,
            cfg.num_modes,
            cfg.domain_extent,
            dst_interpolation=getattr(cfg, "dst_interpolation", False),
        )

        def loss_fn(model, seeds_b, xis_b):
            fs_b, us_b = gen_N(seeds_b)
            return core_loss(model, fs_b, us_b, xis_b)

        @eqx.filter_jit
        def train_step(model, opt_state, seeds_b, xis_b):
            loss, grads = eqx.filter_value_and_grad(loss_fn)(model, seeds_b, xis_b)
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_model = eqx.apply_updates(model, updates)
            return new_model, new_opt_state, loss

        return train_step

    train_steps = {N: _make_train_step(N) for N in unique_Ns}

    batch_size = cfg.batch_size
    xi_dtype = jnp.asarray(Ns).dtype
    state = {"key": key}

    step = 0
    while step < cfg.num_training_steps:
        state["key"], k_bucket, k_sample = jr.split(state["key"], 3)
        bucket_idx = int(jr.choice(k_bucket, len(unique_Ns), p=bucket_probs))
        N = unique_Ns[bucket_idx]
        bucket = buckets[N]
        sample_idx = jr.choice(k_sample, bucket.shape[0], shape=(batch_size,))
        seeds_b = seeds[bucket[sample_idx]]
        xis_b = jnp.full((batch_size,), N, dtype=xi_dtype)

        model, opt_state, loss = train_steps[N](model, opt_state, seeds_b, xis_b)
        step += 1
        if step % cfg.log_every == 0 or step == cfg.num_training_steps:
            loss_value = float(loss)
            print(f"  step {step}/{cfg.num_training_steps}, loss {loss_value:.3e}")
            if loss_out is not None:
                loss_out.append((step, loss_value))

    return model


def train_cached(
    model, fs, us, xis, cfg, spec, stats, *, key, times_out=None, loss_out=None
):
    """Pre-materialised dataset mode; per-step work is just forward/backward.

    Device handling: the full cache (`fs`, `us`, `xis`) and the permutation
    RNG live on the cache device (GPU for cached-gpu, CPU for cached-cpu).
    Each per-step batch slice gets an explicit `jax.device_put` onto the
    compute device (the default JAX device — the GPU on a cluster node / the
    workstation). Without the explicit transfer, mixed-device inputs to
    `eqx.filter_jit` caused ~60× slowdown at 3D, most likely by
    dispatching the compiled function to the host CPU.
    """
    core_loss = _make_core_loss(stats, spec.xi_max)

    optimizer = _build_optimizer(cfg)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    cache_device = list(fs.devices())[0]
    us = jax.device_put(us, cache_device)
    xis = jax.device_put(xis, cache_device)
    key = jax.device_put(key, cache_device)

    compute_device = jax.devices()[0]

    @eqx.filter_jit
    def train_step(model, opt_state, fs_b, us_b, xis_b):
        loss, grads = eqx.filter_value_and_grad(core_loss)(model, fs_b, us_b, xis_b)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_opt_state, loss

    M = len(xis)
    batch_size = cfg.batch_size
    perm_state = {"key": key, "perm": None, "cursor": 0}

    def draw_batch(_step):
        if perm_state["perm"] is None or perm_state["cursor"] + batch_size > M:
            perm_state["key"], sub = jr.split(perm_state["key"])
            perm_state["perm"] = jr.permutation(sub, M)
            perm_state["cursor"] = 0
        start = perm_state["cursor"]
        idx = perm_state["perm"][start : start + batch_size]
        perm_state["cursor"] = start + batch_size
        return (
            jax.device_put(fs[idx], compute_device),
            jax.device_put(us[idx], compute_device),
            jax.device_put(xis[idx], compute_device),
        )

    return _run_loop(
        train_step,
        model,
        opt_state,
        draw_batch,
        cfg,
        times_out=times_out,
        loss_out=loss_out,
    )
