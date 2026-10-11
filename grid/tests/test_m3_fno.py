"""Smoke test for the FiLM-FNO architecture via the truncation runner."""

import csv

import equinox as eqx
import jax
import jax.random as jr

from architectures import FiLMFNO
from experiments._common import CSV_COLUMNS
from experiments.configs import ExperimentConfig
from experiments.run_truncation import run_single_config
from multifidelity.whitening import load_stats


def test_fno_end_to_end_smoke(tmp_path):
    cfg = ExperimentConfig(
        axis="truncation",
        dim=2,
        fidelity_values=(1, 2, 4),
        lambd=0.1,
        budget=20.0,
        seed=0,
        architecture="fno",
        target_resolution=16,
        fno_num_blocks=2,
        fno_hidden_channels=8,
        num_modes_fno=4,
        fno_pad_ratio=1.5,
        num_training_steps=20,
        num_warmup_steps=5,
        batch_size=2,
        num_modes=4,
        num_whitening_samples=4,
        num_test_samples=4,
        save_model_weights=True,  # the test reloads the weights below
        log_every=10,
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
    assert row[2] == "fno"
    assert float(row[header.index("mean_prediction_error")]) >= 0.0
    assert float(row[header.index("mean_residual_error")]) >= 0.0

    stats = load_stats(stats_path)
    assert stats.mean_f.shape == (cfg.target_resolution,) * cfg.dim

    # Model deserialises into a FiLMFNO template with matching param shapes.
    template = FiLMFNO(
        num_spatial_dims=cfg.dim,
        t_scale=float(max(cfg.fidelity_values)),
        num_blocks=cfg.fno_num_blocks,
        hidden_channels=cfg.fno_hidden_channels,
        num_modes=cfg.num_modes_fno,
        pad_ratio=cfg.fno_pad_ratio,
        key=jr.PRNGKey(99),
    )
    reloaded = eqx.tree_deserialise_leaves(str(model_path), template)
    template_leaves = jax.tree.leaves(eqx.filter(template, eqx.is_inexact_array))
    reload_leaves = jax.tree.leaves(eqx.filter(reloaded, eqx.is_inexact_array))
    assert len(template_leaves) == len(reload_leaves)
    for t, r in zip(template_leaves, reload_leaves):
        assert t.shape == r.shape
