# Multi-fidelity neural surrogates on meshes

This folder holds the mesh-based experiments: neural surrogates for PDEs on unstructured meshes,
trained on cost-budgeted mixtures of low- and high-fidelity simulations and conditioned on the
fidelity of each training sample. For the paper and the grid-based experiments, see the
[top-level README](../README.md).

Three problems, each with one or two fidelity axes:

| problem | fidelity axis | levels | dataset |
|---|---|---|---|
| Linear elasticity, perforated plate | solver truncation (Gauss-Seidel iterations) | 11 | 90,000 samples |
| | discretization (mesh refinement) | 21 | 10,000 samples |
| Hyperelasticity, perforated plate (Neo-Hookean) | solver truncation (Newton iterations) | 16 | 90,000 samples |
| | discretization (mesh refinement) | 21 | 10,000 samples |
| Airfoil, 2D RANS (k-ω SST) | discretization (mesh refinement) | 11 | 1,627 cases |

Two architectures are trained on all of them: [Transolver](mf_surrogates/models/transolver.py)
and [AB-UPT](mf_surrogates/models/ab_upt.py).

## Contents

| path | |
|---|---|
| [`mf_surrogates/`](mf_surrogates) | the package: datasets and fidelity sampling ([`data/`](mf_surrogates/data)), models ([`models/`](mf_surrogates/models)), training ([`train/`](mf_surrogates/train)), evaluation ([`eval/`](mf_surrogates/eval)) |
| [`configs/`](configs) | Hydra configs of the paper runs |
| [`main.py`](main.py) | training entrypoint |
| [`evaluate.py`](evaluate.py) | test-set evaluation of all runs of an experiment |
| [`launcher.py`](launcher.py) | small local sweep launcher (one run per GPU) |
| [`data_generation/`](data_generation) | simulation pipelines that produce the datasets |
| [`data/`](data) | dataset location, and the download script for the airfoil dataset |
| [`paper/`](paper) | evaluated results of the paper runs and the scripts for figures and tables |

## Installation

With conda (a CUDA-capable GPU is needed for training):

```bash
conda create -n mf_surrogates -c conda-forge python=3.12
conda activate mf_surrogates
pip install -e .
```

This covers training, evaluation, the airfoil post-processing tools and the paper scripts. The
solid-mechanics simulations need FEniCSx, which is installed separately with conda (see
[`data_generation/`](data_generation/README.md)).

## Data

The datasets are expected under `data/<problem>/<axis>/processed`:

```
data/linear_elasticity/{solver_truncation,discretization}/processed
data/hyperelasticity/{solver_truncation,discretization}/processed
data/airfoil/processed
```

The airfoil dataset is on the Hugging Face Hub
([psetinek/multi-fidelity-airfoil](https://huggingface.co/datasets/psetinek/multi-fidelity-airfoil), ~107 GB):

```bash
python data/download_airfoil.py
```

The solid-mechanics datasets are generated with the pipelines in
[`data_generation/`](data_generation/README.md).

## Training

Pick a dataset, a model and a training config:

```bash
python main.py dataset=linear_elasticity_solver model=transolver training=default
```

| dataset | model | training |
|---|---|---|
| `linear_elasticity_solver`, `hyperelasticity_solver` | `transolver`, `ab_upt` | `default` |
| `linear_elasticity_discretization`, `hyperelasticity_discretization` | `transolver`, `ab_upt` | `discretization` |
| `airfoil` | `transolver_airfoil` | `airfoil` |
| `airfoil` | `ab_upt_airfoil` | `airfoil_abupt` |

The training data is defined by a compute budget, given in units of high-fidelity simulations:
B~ = budget / c_HF, where c_HF is the cost of one high-fidelity sample (noted in each
[dataset config](configs/dataset)). The default is B~ = 200 of high-fidelity data only. A
multi-fidelity mixture draws samples from an exponential distribution over the fidelity levels,
p(ξ) ∝ exp(λξ), spending the same budget. It is set either by the target number of samples (λ is
then solved for) or by λ directly:

```bash
# B~ = 800, mixture with 1,500 training samples
python main.py dataset=linear_elasticity_solver model=transolver training=default \
    dataset.hparams.budget=209715200 dataset.hparams.hf_only=false dataset.hparams.n_samples_target=1500

# B~ = 800, fixed lambda
python main.py dataset=linear_elasticity_solver model=transolver training=default \
    dataset.hparams.budget=209715200 dataset.hparams.hf_only=false dataset.hparams.lambd=-2.0
```

Checkpoints (`best.pt`, `last.pt`) are written to `ckpts/<run_id>/`; they store the realized
training mixture. Logging goes to Weights & Biases (`logging=wandb`, default) or the console
(`logging=local`); `logging.experiment_id` groups runs for evaluation. Multi-GPU training:
`use_ddp=true`.

## Evaluation

[`evaluate.py`](evaluate.py) evaluates all finished runs of an experiment on the test split and
writes one CSV row per run (training mixture and all test metrics):

```bash
python evaluate.py --experiment-id my_experiment --entity <wandb entity> --project <wandb project>
```

## Paper results

[`paper/`](paper/README.md) contains the evaluated results of all mesh runs in the paper and the
scripts that produce the figures and tables from them, including the figures that combine mesh
and grid results.

## Acknowledgements

The AB-UPT implementation follows the AB-UPT paper ([Alkin et al., 2025](https://arxiv.org/abs/2502.09692))
and Emmi AI's [Noether](https://github.com/Emmi-AI/noether) reference implementation; Transolver
follows [Wu et al., 2024](https://arxiv.org/abs/2402.02366). The airfoil simulation pipeline
builds on the AirfRANS setup and code ([Bonnet et al., 2022](https://arxiv.org/abs/2212.07564);
see [`data_generation/airfoil/`](data_generation/airfoil/README.md#vendored-code)).
