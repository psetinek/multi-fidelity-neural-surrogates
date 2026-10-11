#!/usr/bin/env bash
# Dual-reference check: evaluate each surrogate against both the
# target-fidelity test solution and an exact (DST) reference. Two plain Hydra
# multiruns, each sweeping both architectures:
#   2D truncation:     3 λ × 3 budgets × {unet, fno} = 18 runs
#   1D discretization: 3 λ × 3 budgets × {unet, fno} = 18 runs
#
# Equivalent to `python sweeps/launch.py --experiment dual_reference --run`.
#
# Usage (from the repository root):
#     export SLURM_ACCOUNT=<your-account>     # if your cluster needs one
#     bash sweeps/examples/dual_reference.sh [launcher]  # default: slurm_generic

set -euo pipefail
cd "$(dirname "$0")/../.."
launcher=${1:-slurm_generic}

common=(
    seed=0
    eval_super_reference=true
    'hydra.sweep.dir=results/dual_reference/${axis}'
    'hydra.sweep.subdir=${architecture}_dim_${dim}_lambd_${lambd}_budget_${budget}_seed_${seed}'
    hydra/launcher="$launcher"
    hydra.launcher.timeout_min=120
)

uv run python -m experiments.run -m \
    axis=truncation_2d arch=unet,fno \
    lambd=-5,0,20 budget=700,7000,70000 \
    "${common[@]}"

uv run python -m experiments.run -m \
    axis=discretization_1d arch=unet,fno \
    lambd=-5,0,20 budget=4800,48000,480000 \
    "${common[@]}"
