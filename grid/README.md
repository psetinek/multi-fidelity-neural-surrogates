# Multi-fidelity neural surrogates on grids

Code for the Poisson experiments.

We train neural surrogates (FiLM-conditioned UNet and FNO) for the Poisson
equation `Δu = -f` on the unit cube in 1D, 2D and 3D with homogeneous
Dirichlet boundary conditions. The training data comes from a numerical
solver whose fidelity ξ can be traded against cost along one of two axes:

| Axis | Fidelity ξ | Error of a sample | Cost per sample |
|---|---|---|---|
| **truncation** | number of multigrid V-cycles `K` on the 64^D target grid | `O(ρ^K)` | `K` |
| **discretization** | native grid resolution `N` (prolongated to 64^D) | `O(1/N²)` | `N^D` |

For a training-data budget `B`, samples are drawn i.i.d. from a softmax
distribution over the fidelity menu, `q(ξ; λ) ∝ exp(λ ξ̄)` with ξ̄ the
min-max-normalized fidelity, until the cumulative cost reaches `B`.
`λ → -∞` gives only the cheapest samples, `λ = 0` a uniform mixture, and
`λ → +∞` only the most accurate ones. The central question is **which `λ`
minimizes the surrogate's test error at a fixed budget `B`**, and how that
optimum `λ*` moves as `B` grows.

This repository contains everything needed to reproduce the results table
behind the paper's figures (`data/results.csv`), the solver convergence
studies in the appendix and the appendix's qualitative examples.

## Contents

| Path | What it is |
|---|---|
| `poisson_nd/` | JAX Poisson solvers: finite-difference Laplacian, DST direct solver, geometric multigrid (red-black Gauss-Seidel), random Fourier right-hand sides. |
| `multifidelity/` | Fidelity PMF and budget sampler, data generation, whitening, training loop, evaluation. |
| `architectures/` | Dimension-agnostic FiLM-conditioned UNet and FNO. |
| `experiments/` | Hydra entry point (`python -m experiments.run`), per-axis runners, config schema. |
| `configs/` | Hydra configs: defaults, one fidelity menu per (axis, dim), architectures, SLURM launchers. |
| `sweeps/` | `launch.py` reproduces every sweep of the paper; smoke test and two annotated example sweeps. |
| `scripts/build_results_csv.py` | Aggregates all per-run results into one CSV. |
| `scripts/plot_fidelity_examples.py` | Appendix figures of one sample at the lowest vs. highest fidelity of each axis. |
| `data/results.csv` | The results of all 9,317 training runs in the paper. |
| `convergence_studies/` | Multigrid and discretization convergence of the solvers (appendix figures). |
| `tests/` | Pytest suite (solvers, sampler, data generation, models, end-to-end runs). |

## Installation

We use [uv](https://docs.astral.sh/uv/):

```bash
uv sync            # installs the locked environment (JAX with CUDA 12)
uv run pytest      # optional; set JAX_PLATFORMS=cpu on machines without a GPU
```

Python ≥ 3.10. The dependencies are listed in `pyproject.toml` and pinned in
`uv.lock`. All commands below are run from the repository root.

## Running experiments

### A single run

```bash
uv run python -m experiments.run axis=truncation_1d arch=unet lambd=0 budget=3000 seed=0
```

`axis` selects one of `configs/axis/{truncation,discretization}_{1,2,3}d.yaml`
(this sets the dimension and fidelity menu), and `arch` is `unet` or `fno`.
Every field of `configs/config.yaml` can be overridden on the command line,
e.g. `num_training_steps=5000`. The run writes `result.csv` (errors, realized
cost, number of samples) and the whitening statistics to
`results/truncation/unet_dim_1_lambd_0.0_budget_3000.0_seed_0/`. Runs whose
`result.csv` already exists are skipped, so interrupted sweeps can simply be
relaunched.

### Sweeps and SLURM job arrays

Hydra's multirun mode (`-m`) sweeps over the Cartesian product of
comma-separated values. With a launcher from `configs/hydra/launcher/`, the
whole sweep is submitted as one SLURM job array via
[hydra-submitit-launcher](https://hydra.cc/docs/plugins/submitit_launcher/):

```bash
export SLURM_ACCOUNT=<your-account>     # if your cluster requires one
uv run python -m experiments.run -m axis=truncation_1d arch=unet \
    lambd=-5,0,5 budget=300,3000 seed=0,1 \
    hydra/launcher=slurm_generic
```

- **Launchers.** `slurm_generic.yaml` is the template to adapt: set your
  partition, GPU request and walltime. `leonardo`, `lrz`, `lrz_mig`,
  `lrz_mig_3d` and `alex` are the configurations we used on CINECA
  Leonardo, the LRZ AI Systems and NHR@FAU Alex, kept as worked examples.
  Any field can be overridden on the command line, e.g.
  `hydra.launcher.timeout_min=240` or `hydra.launcher.gres=gpu:2g.20gb`.
- **Per-user submission limits.** Hydra splits a multirun into arrays of
  at most `hydra.sweeper.max_batch_size` tasks (900 by default) and submits
  them one after another. Lower it if your QoS has a stricter cap, e.g.
  `hydra.sweeper.max_batch_size=40`.
- **Blocking.** Submitit waits for each array to finish, so run large
  sweeps inside `tmux` or `screen`. Task logs go to `slurm/<cluster>/logs/<jobid>/`.
- **Smoke test.** `bash sweeps/smoke_test.sh [launcher]` runs two tiny
  jobs, locally or through a launcher, and checks their outputs.

### Reproducing the paper's sweeps

The λ × B grids of the paper are irregular because the main sweeps were
refined around the interesting budgets. `sweeps/launch.py` therefore reads
the exact set of runs from `data/results.csv` and emits one Hydra multirun
(one job array) per (axis, dim, architecture, ablation setting, budget):

```bash
uv run python sweeps/launch.py --summary                    # runs per experiment
uv run python sweeps/launch.py --experiment dst             # print the commands
uv run python sweeps/launch.py --experiment dst --launcher slurm_generic --run

# Subsets and extra Hydra overrides (everything after `--`):
uv run python sweeps/launch.py --experiment main --dim 2 --arch fno --seeds 0 \
    --launcher lrz_mig --run -- hydra.sweeper.max_batch_size=40 hydra.launcher.gres=gpu:2g.20gb
```

| `--experiment` | Runs | Description |
|---|---:|---|
| `main` | 5,450 | Both axes, 1D/2D (UNet and FNO, 6 seeds) and 3D (UNet, 3 seeds). |
| `endpoint` | 1,050 | 2D truncation, UNet: top of the menu `K_max ∈ {3, 5, 6, 7, 9}`. |
| `arch` | 765 | 2D truncation, UNet: base channels ∈ {16, 32, 64, 128}, plus a 40k-step convergence check. |
| `duration` | 1,512 | 1D truncation, UNet: 1k to 80k training steps. |
| `dst` | 252 | 1D discretization, UNet: DST instead of linear prolongation of low-resolution samples. |
| `dual_reference` | 36 | Test error against the exact (DST) solution as well as the menu-top reference. |
| `bimodal` | 252 | 2D, UNet: menus reduced to the two endpoints. |

`sweeps/examples/` spells out two of these as plain shell scripts, to show
what a sweep looks like without the generator.

Every run uses 20,000 Adam steps (warmup-cosine schedule, 10% warmup) at
batch size 20 unless the ablation changes it. On A100-class hardware, a run
takes about 3–6 minutes in 1D and 20–40 minutes in 2D (both on a MIG
slice), and 1–6 hours in 3D on a full GPU. The main sweep took roughly
1,300–1,550 full-A100 hours. 3D runs need about 17 GB of GPU memory for the
whitening statistics.

### Building the results table

```bash
uv run python scripts/build_results_csv.py      # → results/results.csv
```

This collects every `result.csv` under `results/`, tags each row with its
experiment (from the output directory layout that `sweeps/launch.py` uses),
and adds derived columns. The output has the same schema as
`data/results.csv`:

| Column | Meaning |
|---|---|
| `experiment` | `main` or the ablation tag (see the table above). |
| `axis`, `dim`, `architecture` | Fidelity axis, spatial dimension, `unet`/`fno`. |
| `lambd`, `budget`, `seed` | Mixture parameter λ, training-data budget `B`, random seed. |
| `realised_cost`, `num_train_samples` | Cost actually spent (≥ `B` by at most one sample) and number of training samples drawn. |
| `num_training_steps`, `base_channels`, `cond_dim`, `dst_interpolation` | Training and model settings (varied only in ablations). |
| `xi_max`, `c_xi_max`, `B_tilde` | Top of the fidelity menu, cost of one such sample, and `B / c_xi_max`, i.e. the budget in units of highest-fidelity samples. |
| `mean_prediction_error` | Mean relative L2 test error against the menu-top solution `P(f; ξ_max)` (100 test samples). |
| `mean_residual_error` | Mean relative residual of the prediction under the discrete Laplacian. |
| `mean_prediction_error_super` | Test error against an exact DST reference solution; only for `endpoint` and `dual_reference`. |

## Solver convergence studies

```bash
uv run python -m convergence_studies.truncation       # multigrid: error vs. V-cycles K
uv run python -m convergence_studies.mg_at_N64        # the N = 64 slice (re-plots the CSV)
uv run python -m convergence_studies.discretization   # error vs. resolution N, O(h²)
```

These run on the CPU in fp64. They write CSVs to `convergence_studies/results/`
and PNG + PDF figures to `convergence_studies/figures/`. The shipped CSVs are
the ones used in the paper; the figures are not tracked. `truncation` and
`discretization` accept `--plot-only` to redraw the figures from the shipped
CSVs without re-solving, and `mg_at_N64` always re-plots from the CSV.

## Qualitative examples

```bash
uv run python scripts/plot_fidelity_examples.py                          # all axes, 1D–3D
uv run python scripts/plot_fidelity_examples.py --dim 2 --axis truncation
```

Plots one test-set sample at the lowest and highest fidelity of each axis
menu, plus their difference, using the same generators as training. On the
discretization axis an additional `*_rhs` figure shows the right-hand side,
which also changes with the native resolution. The 3D figures are volume
renders with [vape4d](https://github.com/KeKsBoTer/vape4d), which needs a
WebGPU-capable GPU. Figures go to `results/plots/fidelity_examples/`.
