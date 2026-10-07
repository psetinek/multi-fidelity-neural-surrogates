# Paper results, figures and tables

This folder contains the evaluated results of all training runs in the paper and the scripts
that turn them into the paper's figures and tables. Every script writes its outputs next to
itself and can be run from any directory.

## Results

[`results/`](results) holds one CSV per experiment, written by [`evaluate.py`](../evaluate.py)
(one row per training run: model, seed, budget B~, mixture λ, realized training set and all
test metrics). The runs were logged under the experiment id `camera_ready_<name>_0`.

| file | experiment |
|---|---|
| [`linear_elasticity_solver_truncation.csv`](results/linear_elasticity_solver_truncation.csv) | linear elasticity, solver error |
| [`linear_elasticity_discretization.csv`](results/linear_elasticity_discretization.csv) | linear elasticity, discretization error |
| [`hyperelasticity_solver_truncation.csv`](results/hyperelasticity_solver_truncation.csv) | hyperelasticity, solver error |
| [`hyperelasticity_discretization.csv`](results/hyperelasticity_discretization.csv) | hyperelasticity, discretization error |
| [`airfoil.csv`](results/airfoil.csv) | airfoil RANS, mesh refinement |
| [`hyperelasticity_{solver,discretization}_binary.csv`](results) | bi-fidelity comparison |
| [`xi_sweep_hyperelasticity_solver.csv`](results/xi_sweep_hyperelasticity_solver.csv) | conditioning experiments (fidelity-input sweep) |
| [`poisson.csv`](results/poisson.csv) | nonlinear Poisson (grid models) |

To recompute a CSV from the checkpoints, e.g.

```bash
python evaluate.py --experiment-id camera_ready_airfoil_0 --entity <entity> --project <project> \
    --ckpt-dir <checkpoints>/airfoil --data-path data/airfoil/processed --batch-size 2 \
    --out paper/results/airfoil.csv
```

and [`figures/conditioning/xi_sweep_eval.py`](figures/conditioning/xi_sweep_eval.py) for the
fidelity-input sweep (a few GPU hours).

## Figures and tables

Run the scripts in this order; the second and third steps read outputs of the first.

| script | produces | needs datasets |
|---|---|---|
| [`figures/results/condensed.py`](figures/results/condensed.py) | condensed result figures, savings summary (`savings_summary.{pdf,csv}`) | |
| [`figures/results/full_appendix.py`](figures/results/full_appendix.py) | full result overviews per problem, Poisson figures | |
| [`figures/results/airfoil.py`](figures/results/airfoil.py) | airfoil result figures (fields, C_l, C_d) | |
| [`figures/figure_1/plot_fig1.py`](figures/figure_1/plot_fig1.py) | Figure 1 (after `condensed.py`) | |
| [`figures/mf_vs_bf/plot_mf_vs_bf.py`](figures/mf_vs_bf/plot_mf_vs_bf.py) | multi- vs bi-fidelity comparison | |
| [`figures/conditioning/plot_conditioning.py`](figures/conditioning/plot_conditioning.py) | conditioning figures for B~ = 400, 1600, 6400 | hyperelasticity solver |
| [`figures/conditioning/plot_condscale_ab.py`](figures/conditioning/plot_condscale_ab.py) | conditioning-scale ablation (B~ = 1600) | hyperelasticity solver |
| [`figures/sampling/plot_sampling.py`](figures/sampling/plot_sampling.py) | dataset-mixing figure | linear elasticity solver |
| [`figures/dataset_descriptions/`](figures/dataset_descriptions) | dataset convergence and example figures | all |
| [`tables/make_detailed_tables.py`](tables/make_detailed_tables.py) | detailed result tables (LaTeX) | |
| [`extrapolation/recipe.py`](extrapolation/recipe.py) | λ* extrapolation recipe (`recipe_results.csv`) | |
| [`extrapolation/make_regret_table.py`](extrapolation/make_regret_table.py) | regret tables (after `recipe.py`) | |
| [`extrapolation/overhead.py`](extrapolation/overhead.py) | calibration overhead (after `recipe.py`) | linear elasticity solver |

Scripts marked with a dataset read its `metadata.csv` (or samples) from [`data/`](../data); see
[`data_generation/`](../data_generation) for generating the solid-mechanics datasets and
[`data/download_airfoil.py`](../data/download_airfoil.py) for the airfoil dataset.
[`figures/mf_vs_bf/make_binary_metadata.py`](figures/mf_vs_bf/make_binary_metadata.py) builds the
metadata file used to train the bi-fidelity runs.
