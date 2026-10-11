"""Shared helpers for the per-axis runner scripts (truncation, discretization)."""

import csv
from functools import reduce
from operator import mul
from pathlib import Path

import pandas as pd


CSV_COLUMNS = [
    "axis",
    "dim",
    "architecture",
    "lambd",
    "budget",
    "realised_cost",
    "seed",
    "num_train_samples",
    "mean_prediction_error",
    "mean_residual_error",
    # Architecture-size columns, added for the capacity ablation. Both are
    # UNet-only knobs; FNO rows carry the resolved config values but ignore
    # them (FNO sizes itself via `fno_hidden_channels`).
    "base_channels",
    "cond_dim",
    # Prediction error vs the DST-exact reference solution; only populated when
    # `eval_super_reference` is set (truncation). Empty (→ NaN) otherwise.
    "mean_prediction_error_super",
]

# Every result.csv written before the architecture-size ablation has the
# 10-column schema and was trained at these values. Backfilled on read so
# the landed rows are correctly labelled rather than NaN.
LEGACY_COLUMN_DEFAULTS = {"base_channels": 64, "cond_dim": 256}


def write_loss_history(path, loss_history) -> None:
    """Write the `(step, loss)` pairs collected by `multifidelity.training`.

    Two properties worth remembering when reading these back:

    - The value is the *instantaneous single-batch* training loss at that
      step, not a running mean. At the default `batch_size=20` it is noisy;
      judge convergence by comparing the mean over the last ~1k steps
      against the preceding ~1k rather than by the final value.
    - It is MSE on *whitened* targets (see `multifidelity.training.
      _make_core_loss`), so it is not comparable to the relative-L2
      `mean_prediction_error` in `result.csv`. It *is* comparable across
      runs that share (axis, dim, lambd, budget, seed), since those draw
      identical samples and therefore identical whitening stats — which is
      what makes it usable for comparing architecture sizes.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        writer.writerows(loss_history)


def estimate_cache_gb(num_samples: int, shape: tuple) -> float:
    """(fs, us) both fp32 at `shape` — 4 bytes per element, 2 channels."""
    return num_samples * 2 * 4 * reduce(mul, shape, 1) / 1e9


def get_all_results(
    results_root: str | Path = "results",
    axis: str | None = None,
) -> pd.DataFrame:
    """Concatenate every `result.csv` under `<results_root>/<axis>/*/`.

    `axis` selects `"truncation"`, `"discretization"`, or both when `None`.
    Returns an empty DataFrame with `CSV_COLUMNS` if nothing is found.
    """
    root = Path(results_root)
    axes = [axis] if axis is not None else ["truncation", "discretization"]

    dfs = []
    for a in axes:
        axis_dir = root / a
        if not axis_dir.exists():
            continue
        for p in sorted(axis_dir.glob("*/result.csv")):
            dfs.append(pd.read_csv(p))

    if not dfs:
        return pd.DataFrame(columns=CSV_COLUMNS)

    df = pd.concat(dfs, ignore_index=True)
    for col, default in LEGACY_COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)
        df[col] = df[col].astype(int)
    return df
