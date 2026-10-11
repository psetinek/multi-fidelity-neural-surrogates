#!/usr/bin/env bash
# Tiny 2-run smoke test of the full pipeline: Hydra multirun -> (optionally)
# SLURM array via Submitit -> per-run result.csv.
#
# Usage (from the repository root):
#     bash sweeps/smoke_test.sh                 # run locally, one after another
#     bash sweeps/smoke_test.sh slurm_generic   # submit as a SLURM array
#
# The argument is any launcher in configs/hydra/launcher/. Submitit blocks
# until the array finishes, so on a cluster run this inside tmux/screen.
# On failure, check the task logs under slurm/**/logs/<jobid>/.
#
# Outputs: results/smoke/truncation/unet_dim_1_lambd_{0.0,5.0}_budget_300.0_seed_0/

set -euo pipefail
cd "$(dirname "$0")/.."

launcher=${1:-}
launcher_override=()
if [[ -n "$launcher" ]]; then
    launcher_override=("hydra/launcher=$launcher")
fi

echo "[$(date +%H:%M:%S)] smoke test (${launcher:-local}): 2 runs, 100 training steps"

uv run python -m experiments.run -m \
    axis=truncation_1d arch=unet \
    lambd=0,5 budget=300 seed=0 \
    num_training_steps=100 num_warmup_steps=10 num_whitening_samples=200 \
    num_test_samples=20 log_every=50 \
    'hydra.sweep.dir=results/smoke/${axis}' \
    "${launcher_override[@]}"

missing=0
for lambd in 0.0 5.0; do
    csv="results/smoke/truncation/unet_dim_1_lambd_${lambd}_budget_300.0_seed_0/result.csv"
    if [[ -f "$csv" ]]; then
        echo "  OK       $csv"
    else
        echo "  MISSING  $csv"
        missing=$((missing + 1))
    fi
done

if (( missing > 0 )); then
    echo "[$(date +%H:%M:%S)] $missing / 2 runs failed."
    exit 1
fi
echo "[$(date +%H:%M:%S)] smoke test passed."
