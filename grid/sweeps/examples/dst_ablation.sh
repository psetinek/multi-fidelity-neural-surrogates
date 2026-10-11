#!/usr/bin/env bash
# DST-interpolation ablation (1D discretization, UNet), written out as one
# plain Hydra multirun: 7 λ × 6 budgets × 6 seeds = 252 runs, submitted as
# a single SLURM job array.
#
# Equivalent to `python sweeps/launch.py --experiment dst --run`, but spelled
# out to show what a sweep looks like. `dst_interpolation=true` prolongates
# low-resolution solutions to the 64-point target grid with the DST sine
# basis instead of linear interpolation, removing interpolation error from
# the low-fidelity samples.
#
# Usage (from the repository root):
#     export SLURM_ACCOUNT=<your-account>     # if your cluster needs one
#     bash sweeps/examples/dst_ablation.sh [launcher]   # default: slurm_generic

set -euo pipefail
cd "$(dirname "$0")/../.."
launcher=${1:-slurm_generic}

uv run python -m experiments.run -m \
    axis=discretization_1d arch=unet \
    dst_interpolation=true \
    lambd=-20,-5,-1,0,1,5,20 \
    budget=480,4800,14400,48000,144000,480000 \
    seed=0,1,2,3,4,5 \
    'hydra.sweep.dir=results/dst_ablation/${axis}' \
    'hydra.sweep.subdir=${architecture}_dim_${dim}_lambd_${lambd}_budget_${budget}_seed_${seed}' \
    hydra/launcher="$launcher" \
    hydra.launcher.timeout_min=120
