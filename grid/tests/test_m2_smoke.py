"""End-to-end smoke test (discretization axis).

Runs the discretization-axis runner at tiny sizes on CPU/fp64 (enforced by
`tests/conftest.py`). Verifies artefacts exist on disk with the right
schema and that the model round-trips through eqx save/load. Parametrised
over `cache_dataset ∈ {False, True}` so both streaming and cached paths
are covered.
"""

import csv

import equinox as eqx
import jax
import jax.random as jr
import pytest

from architectures import FiLMUNet
from experiments._common import CSV_COLUMNS
from experiments.configs import ExperimentConfig, resolve_cond_dim
from experiments.run_discretization import run_single_config
from multifidelity.whitening import load_stats


@pytest.mark.parametrize("cache_dataset", [False, True])
def test_end_to_end_smoke(tmp_path, cache_dataset):
    cfg = ExperimentConfig(
        axis="discretization",
        dim=2,
        fidelity_values=(16, 24, 32),
        lambd=0.0,  # uniform — each N gets some samples
        budget=10_000.0,  # ~10 samples at average cost ~(24²)
        seed=0,
        architecture="unet",
        target_resolution=16,
        num_levels=3,
        base_channels=8,
        num_training_steps=20,
        num_warmup_steps=5,
        batch_size=2,
        num_modes=4,
        num_whitening_samples=6,
        num_test_samples=4,
        log_every=10,
        cache_dataset=cache_dataset,
        save_model_weights=True,  # the test reloads the weights below
    )

    run_single_config(cfg, results_root=str(tmp_path))

    folder = tmp_path / "discretization" / cfg.folder_name()
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
    assert row[0] == "discretization"
    assert row[1] == "2"
    assert row[2] == "unet"
    assert float(row[header.index("mean_prediction_error")]) >= 0.0
    assert float(row[header.index("mean_residual_error")]) >= 0.0

    stats = load_stats(stats_path)
    assert stats.mean_f.shape == (cfg.target_resolution,) * cfg.dim
    assert stats.std_u.shape == (cfg.target_resolution,) * cfg.dim

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
