# AGENTS.md

Orientation for coding agents working in this folder. The human-facing documentation is
[`README.md`](README.md); read it for installation, data and usage.

## What this is

Multi-fidelity training of neural surrogates on unstructured meshes. Training sets are drawn
from simulations at several fidelities under a fixed compute budget, and the model is
conditioned on the fidelity of each sample; evaluation is always on high-fidelity (HF) data.
Problems: linear elasticity and hyperelasticity on a perforated plate (solver-truncation and
discretization fidelity axes) and 2D airfoil RANS (mesh refinement). Models: Transolver, AB-UPT.
This folder is the `mesh/` part of a larger repository whose `grid/` part holds the grid-based
experiments.

## Layout

| path | content |
|---|---|
| `main.py` | Hydra training entrypoint (`configs/main.yaml`); single GPU or DDP (`use_ddp=true`) |
| `evaluate.py` | lists finished wandb runs of an `experiment_id`, deduplicates, evaluates each checkpoint on the test split, writes one CSV row per run |
| `launcher.py` | local grid sweeps over comma-separated overrides with a GPU pool |
| `mf_surrogates/data/` | `base.py::BaseMFDataset` (fidelity linearization, sampling plan, normalization hooks), `plate.py`, `airfoil.py`, `get_data()` in `__init__.py`, registry |
| `mf_surrogates/models/` | `transolver.py`, `ab_upt.py` + `ab_upt_utils/`, `utils/condition.py` (sin-cos embedding, DiT modulation), registry; `get_model()` in `__init__.py` |
| `mf_surrogates/train/` | `trainer.py` (step-based loop, eval, checkpointing, early stopping), `loss.py`, `utils.py` (optimizer, scheduler, loss factories) |
| `mf_surrogates/eval/` | `plate_utils.eval_epoch`, `airfoil_utils.eval_epoch` (training validation) and `eval_epoch_with_coeffs` (test, incl. C_d/C_l), `evaluate.py` (per-checkpoint evaluation used by the root `evaluate.py`) |
| `mf_surrogates/utils.py` | `set_seed`, `load_ckpt`, `Logger`, DDP setup |
| `configs/` | `dataset/`, `model/`, `training/` (+ `optimizer/`, `scheduler/`, `loss/`), `logging/`; the configs of the paper runs |
| `data/` | dataset location (`data/<problem>/<axis>/processed`, not versioned) and `download_airfoil.py` |
| `data_generation/` | simulation pipelines: `linear_elasticity/`, `hyperelasticity/` (FEniCSx, own conda env), `airfoil/` (OpenFOAM v2506 + post-processing, vendored AirfRANS code) |
| `paper/` | `results/` (evaluated CSVs of the paper runs) and scripts for figures, tables and the λ* extrapolation recipe |

## Multi-fidelity mechanism (`BaseMFDataset`)

1. `metadata.csv` lists every (sample, fidelity level) with its cost and raw fidelity.
2. Raw fidelities are linearized onto ξ̂ ∈ [0, 1] with a convergence model: power law E ∝ N^-q for
   mesh size, geometric E ∝ q^k for solver iterations (`linearization_col`, `q` in the dataset
   config).
3. The training plan samples one fidelity per sample from p(ξ̂) ∝ exp(λ ξ̂) within `budget`
   (in units of `cost_column`). `hf_only=true` uses HF samples only; otherwise `lambd` is used if
   set, else λ is found by bisection within `lambda_bounds` to hit `n_samples_target`.
4. The plan is fixed by `composition_seed`; validation and test splits are HF only.
5. B~ = budget / c_HF (c_HF = mean HF sample cost) is the budget in HF-simulation units used
   throughout the paper.

The models multiply the conditioning input by `cond_scale` (1000 in the paper configs) before a
sin-cos embedding.

## Running

```bash
python main.py dataset=<dataset> model=<model> training=<training> [overrides]
python evaluate.py --experiment-id <id> --entity <entity> --project <project> [--ckpt-dir ...] [--data-path ...]
```

Paper combinations: solver datasets with `training=default`, discretization datasets with
`training=discretization`, airfoil with `model=transolver_airfoil training=airfoil` or
`model=ab_upt_airfoil training=airfoil_abupt`. Paths in the configs are relative to this folder
(`hydra.job.chdir: false`), so run from here.

## Conventions

- Checkpoints: `<output_path>/<run_id>/best.pt` (lowest validation error) and `last.pt`; they store
  the config, the training-set normalization statistics and the realized fidelity mix.
  `load_ckpt` rebuilds model and datasets from them and loads weights strictly.
- Model selection metric: `val/rel_l2_loss_fields_avg` (plate: mean per-field relative L2; airfoil:
  volume (u, p) + surface (p, wss) per-channel relative L2).
- Model initialization consumes the global RNG in a fixed order; changing or reordering
  `reset_parameters`/init calls changes the weights a given seed produces.
- Datasets, checkpoints, wandb files and generated figures/tables are not versioned (see
  `.gitignore`); `paper/results/*.csv` are.
