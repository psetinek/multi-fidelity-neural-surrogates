"""End-to-end smoke test (truncation axis).

Runs the truncation-axis runner at tiny sizes on CPU/fp64 (enforced by
`tests/conftest.py`). Verifies artefacts exist on disk with the right
schema and that the model round-trips through eqx save/load.
"""

import csv

import equinox as eqx
import jax
import jax.random as jr
import pytest

from architectures import FiLMUNet
from experiments.configs import ExperimentConfig, resolve_cond_dim
from experiments.run_truncation import CSV_COLUMNS, run_single_config
from multifidelity.whitening import load_stats


@pytest.mark.parametrize("cache_dataset", [False, True])
def test_end_to_end_smoke(tmp_path, cache_dataset):
    cfg = ExperimentConfig(
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
        num_whitening_samples=4,
        num_test_samples=4,
        log_every=10,
        cache_dataset=cache_dataset,
        save_model_weights=True,  # the test reloads the weights below
    )

    run_single_config(cfg, results_root=str(tmp_path))

    folder = tmp_path / "truncation" / cfg.folder_name()
    csv_path = folder / "result.csv"
    model_path = folder / "model.eqx"
    stats_path = folder / "whitening.npz"

    assert csv_path.exists()
    assert model_path.exists()
    assert stats_path.exists()

    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        row = next(reader)
    assert header == CSV_COLUMNS
    assert row[0] == "truncation"
    assert row[1] == "2"
    assert row[2] == "unet"
    # Errors are finite floats.
    assert float(row[header.index("mean_prediction_error")]) >= 0.0
    assert float(row[header.index("mean_residual_error")]) >= 0.0

    # Whitening stats round-trip with the expected spatial shape.
    stats = load_stats(stats_path)
    assert stats.mean_f.shape == (cfg.target_resolution,) * cfg.dim
    assert stats.std_u.shape == (cfg.target_resolution,) * cfg.dim

    # Model deserialises into a template of the same architecture.
    template = FiLMUNet(
        num_spatial_dims=cfg.dim,
        t_scale=float(max(cfg.fidelity_values)),
        num_levels=cfg.num_levels,
        base_channels=cfg.base_channels,
        cond_dim=resolve_cond_dim(cfg),
        key=jr.PRNGKey(99),
    )
    reloaded = eqx.tree_deserialise_leaves(str(model_path), template)
    template_leaves = jax.tree.leaves(eqx.filter(template, eqx.is_inexact_array))
    reload_leaves = jax.tree.leaves(eqx.filter(reloaded, eqx.is_inexact_array))
    assert len(template_leaves) == len(reload_leaves)
    for t, r in zip(template_leaves, reload_leaves):
        assert t.shape == r.shape


def test_run_or_skip_idempotent(tmp_path):
    """Second invocation with the same config should no-op."""
    cfg = ExperimentConfig(
        axis="truncation",
        dim=2,
        fidelity_values=(1, 2, 4),
        lambd=0.0,
        budget=10.0,
        seed=0,
        architecture="unet",
        target_resolution=16,
        num_levels=3,
        base_channels=8,
        num_training_steps=10,
        num_warmup_steps=2,
        batch_size=2,
        num_whitening_samples=2,
        num_test_samples=2,
        log_every=5,
    )
    run_single_config(cfg, results_root=str(tmp_path))
    csv_path = tmp_path / "truncation" / cfg.folder_name() / "result.csv"
    first_mtime = csv_path.stat().st_mtime

    # Re-running must not rewrite the CSV.
    run_single_config(cfg, results_root=str(tmp_path))
    assert csv_path.stat().st_mtime == first_mtime
