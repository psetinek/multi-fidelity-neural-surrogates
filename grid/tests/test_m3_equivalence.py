"""Sanity check: streaming and cached training paths produce close results.

All three paths (streaming, cached-gpu, cached-cpu) run `run_single_config`
with the same seed, same data, same permutation schedule. Differences come
only from `vmap` + `fori_loop` masking and (for cached-cpu) per-batch
device transfers — all ε-level. Tolerance is generous enough to
accommodate how ε noise can amplify through training steps; tighten if
we find we can.
"""

import csv

import pytest

from experiments.configs import ExperimentConfig
from experiments.run_truncation import run_single_config


def _read_metrics(results_root, cfg):
    path = results_root / cfg.axis / cfg.folder_name() / "result.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader)
    return (
        float(row["mean_prediction_error"]),
        float(row["mean_residual_error"]),
    )


def _base_cfg(cache_dataset: bool, cache_location: str) -> ExperimentConfig:
    return ExperimentConfig(
        axis="truncation",
        dim=2,
        fidelity_values=(1, 2, 4, 8),
        lambd=0.1,
        budget=20.0,
        seed=0,
        architecture="unet",
        target_resolution=16,
        num_levels=3,
        base_channels=8,
        num_training_steps=20,
        num_warmup_steps=5,
        batch_size=2,
        num_modes=4,
        num_whitening_samples=4,
        num_test_samples=4,
        log_every=10,
        cache_dataset=cache_dataset,
        cache_location=cache_location,
    )


@pytest.mark.parametrize(
    "variant,cache_dataset,cache_location",
    [
        ("cached-gpu", True, "gpu"),
        ("cached-cpu", True, "cpu"),
    ],
)
def test_matches_streaming(tmp_path, variant, cache_dataset, cache_location):
    """Streaming (baseline) vs each cached variant, pairwise."""
    cfg_stream = _base_cfg(cache_dataset=False, cache_location="gpu")
    cfg_cache = _base_cfg(cache_dataset=cache_dataset, cache_location=cache_location)

    stream_root = tmp_path / "stream"
    cache_root = tmp_path / variant
    run_single_config(cfg_stream, results_root=str(stream_root))
    run_single_config(cfg_cache, results_root=str(cache_root))

    pred_s, res_s = _read_metrics(stream_root, cfg_stream)
    pred_c, res_c = _read_metrics(cache_root, cfg_cache)

    assert (
        abs(pred_s - pred_c) < 5e-2
    ), f"{variant}: pred err diverges: stream={pred_s} cache={pred_c}"
    assert (
        abs(res_s - res_c) < 5e-2
    ), f"{variant}: res err diverges: stream={res_s} cache={res_c}"
