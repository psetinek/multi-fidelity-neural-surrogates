"""Test-set construction and error metrics.

The test set is fully determined by `cfg.test_seed` and runs at
`spec.xi_max` (highest fidelity on the support). Two axis-specific
constructors mirror the two generators; `evaluate` is axis-agnostic.

Metrics:
- mean relative L2 prediction error against the reference solution;
- mean relative L2 residual under the (2d+1)-point Laplacian applied to
  the dewhitened prediction.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

from multifidelity.conditioning import transform_log
from multifidelity.data import generate_discretization, generate_truncation
from multifidelity.whitening import WhiteningStats, dewhiten, whiten
from poisson_nd.dst import dst_resample, solve_direct
from poisson_nd.multigrid import prolongate
from poisson_nd.operators import laplacian
from poisson_nd.rhs import evaluate_rhs, sample_coefficients

# Ultra-fine reference resolution per dim for the discretization super-reference
# (the same analytical field solved far finer than the menu top, restricted to
# the target grid). 1D can afford a very fine reference; 3D is memory-bound.
DISC_SUPER_NREF = {1: 1024, 2: 512, 3: 192}


def create_test_set_truncation(spec, cfg, shape, *, key_test):
    """Fixed-seed test dataset at `spec.xi_max` (= K_max).

    Returns (fs, us, xis, us_super). `us_super` is the DST-exact discrete solve
    at the target grid (the K→∞ limit the menu top approximates) when
    `eval_super_reference` is set, else None.
    """
    n = cfg.num_test_samples
    seeds = jr.split(key_test, n)
    xis = jnp.full((n,), spec.xi_max, dtype=jnp.asarray(spec.values).dtype)

    def body(pair):
        s, k = pair
        return generate_truncation(
            s,
            k,
            shape,
            num_modes=cfg.num_modes,
            domain_extent=cfg.domain_extent,
        )

    chunk = min(cfg.batch_size, n)
    fs, us = jax.lax.map(body, (seeds, xis), batch_size=chunk)

    us_super = None
    if getattr(cfg, "eval_super_reference", False):
        h = cfg.domain_extent / (cfg.target_resolution + 1)
        us_super = jax.lax.map(lambda f: solve_direct(f, h), fs, batch_size=chunk)
    return fs, us, xis, us_super


def create_test_set_discretization(spec, cfg, target_shape, *, key_test):
    """Fixed-seed test dataset at `spec.xi_max` (= N_max).

    Returns (fs, us, xis, us_super). `us_super` (when `eval_super_reference` is
    set) is the SAME analytical field solved at an ultra-fine reference
    resolution and restricted to `target_shape` with the same operator the
    pipeline uses (linear or DST). The field is reconstructed from each test
    seed's coefficients and normalized by the N_max-grid max, so it matches the
    scale the surrogate predicts; the menu-top error and the super error then
    differ only by the native-N vs reference resolution.
    """
    target_shape = tuple(target_shape)
    d = len(target_shape)
    n = cfg.num_test_samples
    seeds = jr.split(key_test, n)
    N_max = int(spec.xi_max)

    def body(seed):
        return generate_discretization(
            seed,
            N_max,
            target_shape,
            num_modes=cfg.num_modes,
            domain_extent=cfg.domain_extent,
            dst_interpolation=getattr(cfg, "dst_interpolation", False),
        )

    chunk = min(cfg.batch_size, n)
    fs, us = jax.lax.map(body, seeds, batch_size=chunk)
    xis = jnp.full((n,), N_max, dtype=jnp.asarray(spec.values).dtype)

    us_super = None
    if getattr(cfg, "eval_super_reference", False):
        n_ref = DISC_SUPER_NREF[d]
        ref_shape = (n_ref,) * d
        nmax_shape = (N_max,) * d
        h_ref = cfg.domain_extent / (n_ref + 1)
        resample = (
            dst_resample if getattr(cfg, "dst_interpolation", False) else prolongate
        )

        def super_body(seed):
            amps, phs = sample_coefficients(seed, cfg.num_modes, d)
            norm = jnp.max(
                jnp.abs(evaluate_rhs(amps, phs, nmax_shape, cfg.domain_extent))
            )
            f_fine = evaluate_rhs(amps, phs, ref_shape, cfg.domain_extent) / norm
            return resample(solve_direct(f_fine, h_ref), target_shape)

        us_super = jax.lax.map(super_body, seeds, batch_size=chunk)
    return fs, us, xis, us_super


def evaluate(model, test_data, stats: WhiteningStats, cfg):
    """Return mean relative L2 prediction + residual errors.

    The model forward pass is chunked via `jax.lax.map(..., batch_size=cfg.batch_size)`
    so 3D configs don't OOM on 40GB GPUs — at 3D/N=64 with 64 base channels,
    a batch=100 UNet forward allocates multi-GB activation tensors that
    won't fit alongside the cached dataset. Chunking to `batch_size` (20
    by default) keeps eval memory in line with training memory.
    """
    fs, us_ref, xis, us_super = test_data

    whitened_fs = jax.vmap(lambda f: whiten(f, stats))(fs)
    inputs = whitened_fs[:, None]
    t = transform_log(xis, max(cfg.fidelity_values))

    chunk = min(cfg.batch_size, cfg.num_test_samples)
    pred_norm = jax.lax.map(
        lambda it: model(it[0], it[1]), (inputs, t), batch_size=chunk
    )
    u_pred = jax.vmap(lambda p: dewhiten(p[0], stats))(pred_norm)

    h = cfg.domain_extent / (cfg.target_resolution + 1)

    def _errs(pred, ref, rhs):
        pred_err = jnp.linalg.norm(pred - ref) / jnp.linalg.norm(ref)
        res_err = jnp.linalg.norm(-laplacian(pred, h) - rhs) / jnp.linalg.norm(rhs)
        return pred_err, res_err

    pred_errs, res_errs = jax.vmap(_errs)(u_pred, us_ref, fs)
    out = {
        "mean_prediction_error": float(jnp.mean(pred_errs)),
        "mean_residual_error": float(jnp.mean(res_errs)),
    }

    # Optional: error against a super-fidelity reference solution, if the test
    # set built one (truncation: DST-exact solve; discretization: ultra-fine
    # solve of the same field). Isolates how much reported error is the solver's
    # incomplete convergence / finite resolution rather than the surrogate.
    if us_super is not None:
        super_errs = jax.vmap(lambda p, r: jnp.linalg.norm(p - r) / jnp.linalg.norm(r))(
            u_pred, us_super
        )
        out["mean_prediction_error_super"] = float(jnp.mean(super_errs))

    return out
