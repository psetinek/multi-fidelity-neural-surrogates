"""Truncation-axis runner: implements `run_single_config` for one
`ExperimentConfig`. Invoked by `experiments.run` (the Hydra entry point)
which composes configs and handles grid sweeps via multirun.

Writes `result.csv` and `whitening.npz` into the run's output folder
(plus `loss_history.csv` when `cfg.save_loss_history`)
(Hydra-managed when called via `experiments.run`; under
`<results_root>/truncation/<cfg.folder_name()>/` otherwise). Skips
cleanly when `result.csv` already exists.

`model.eqx` (trained weights) is only persisted if
`cfg.save_model_weights == True`. The default is False — analysis
reads only `result.csv`, and weight files dominate disk usage in
multi-seed sweeps (e.g. ~80 GB across a 5-seed 1D+2D run), so we
discard them by default.
"""

import csv
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr

from experiments._common import CSV_COLUMNS, estimate_cache_gb, write_loss_history
from experiments.configs import ExperimentConfig, build_model, resolve_cond_dim
from multifidelity.data import materialise_truncation
from multifidelity.evaluation import create_test_set_truncation, evaluate
from multifidelity.fidelity import FidelitySpec, sample_budget
from multifidelity.training import train_cached, train_streaming_truncation
from multifidelity.whitening import save_stats, warmup_stats_truncation


def run_single_config(
    cfg: ExperimentConfig,
    results_root: str = "results",
    *,
    folder: Path | None = None,
) -> dict:
    spec = FidelitySpec(
        axis=cfg.axis,
        values=cfg.fidelity_values,
        cost_fn=lambda K: float(K),
        xi_max=float(max(cfg.fidelity_values)),
    )

    if folder is None:
        folder = Path(results_root) / cfg.axis / cfg.folder_name()
    result_csv = folder / "result.csv"
    if result_csv.exists():
        print(f"[skip] {result_csv} exists")
        return {"status": "skipped", "result_csv": str(result_csv)}

    folder.mkdir(parents=True, exist_ok=True)
    shape = (cfg.target_resolution,) * cfg.dim

    key = jr.PRNGKey(cfg.seed)
    sample_key, seeds_key, whitening_key, model_key, training_key = jr.split(key, 5)

    xis, realised_cost = sample_budget(sample_key, spec, cfg.lambd, cfg.budget)
    num_train_samples = int(xis.shape[0])
    print(
        f"Sampled {num_train_samples} training samples; realised cost "
        f"{realised_cost:.2f} (budget {cfg.budget})"
    )

    sample_seeds = jr.split(seeds_key, num_train_samples)

    print("Warmup: computing whitening stats...")
    stats = warmup_stats_truncation(sample_seeds, xis, cfg, shape, key=whitening_key)
    save_stats(folder / "whitening.npz", stats)

    model = build_model(cfg, key=model_key)
    loss_history = [] if cfg.save_loss_history else None

    if cfg.cache_dataset:
        gb = estimate_cache_gb(num_train_samples, shape)
        print(
            f"Caching dataset: {num_train_samples} samples x {shape} (~{gb:.2f} GB at fp32), "
            f"location={cfg.cache_location}"
        )
        cpu = jax.devices("cpu")[0] if cfg.cache_location == "cpu" else None
        fs, us = materialise_truncation(sample_seeds, xis, cfg, shape, cpu_device=cpu)
        # For 1D/2D cached-cpu the function returns GPU arrays; move them.
        # For 3D cached-cpu the function already returns CPU arrays and
        # this `device_put` is a no-op.
        if cpu is not None:
            fs = jax.device_put(fs, cpu)
            us = jax.device_put(us, cpu)
        print("Training (cached)...")
        trained = train_cached(
            model,
            fs,
            us,
            xis,
            cfg,
            spec,
            stats,
            key=training_key,
            loss_out=loss_history,
        )
    else:
        print("Training (streaming)...")
        trained = train_streaming_truncation(
            model,
            sample_seeds,
            xis,
            cfg,
            spec,
            stats,
            shape,
            key=training_key,
            loss_out=loss_history,
        )

    if cfg.save_loss_history:
        write_loss_history(folder / "loss_history.csv", loss_history)

    print("Evaluating...")
    test_data = create_test_set_truncation(
        spec, cfg, shape, key_test=jr.PRNGKey(cfg.test_seed)
    )
    metrics = evaluate(trained, test_data, stats, cfg)

    with open(result_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        writer.writerow(
            [
                cfg.axis,
                cfg.dim,
                cfg.architecture,
                cfg.lambd,
                cfg.budget,
                realised_cost,
                cfg.seed,
                num_train_samples,
                metrics["mean_prediction_error"],
                metrics["mean_residual_error"],
                cfg.base_channels,
                resolve_cond_dim(cfg),
                metrics.get("mean_prediction_error_super", ""),
            ]
        )

    if cfg.save_model_weights:
        eqx.tree_serialise_leaves(str(folder / "model.eqx"), trained)
    print(f"[done] {folder}")

    return {
        "status": "ok",
        "axis": cfg.axis,
        "dim": cfg.dim,
        "architecture": cfg.architecture,
        "lambd": cfg.lambd,
        "budget": cfg.budget,
        "seed": cfg.seed,
        "num_train_samples": num_train_samples,
        "mean_prediction_error": float(metrics["mean_prediction_error"]),
        "mean_residual_error": float(metrics["mean_residual_error"]),
    }
