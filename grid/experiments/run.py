"""Unified Hydra entry point for compute-optimal experiments.

Single runs and multirun sweeps share this one entry point; the axis
(`truncation` / `discretization`) dispatches to the appropriate
`run_single_config` implementation. Hydra places each run into its own
output directory (see `configs/config.yaml` → `hydra.run.dir` /
`hydra.sweep.subdir`), which matches the on-disk layout the rest of the
pipeline (including `experiments._common.get_all_results`) expects.

Examples (one YAML per (axis, dim), so `dim` is pinned by the
axis group and is not a separate CLI field):

    # Single local run
    python -m experiments.run axis=truncation_1d lambd=0.3 budget=2500 seed=0

    # Local multirun (Cartesian product):
    python -m experiments.run -m axis=truncation_1d \\
        lambd=-0.3,0,0.3 budget=1000,2500 seed=0

    # SLURM array via Submitit (see configs/hydra/launcher/):
    python -m experiments.run -m axis=truncation_1d \\
        lambd=-20,-10,-5,-2,-1,-0.5,0,0.5,1,2,5,10,20 \\
        budget=1000,2500,5000,10000,25000,50000,100000,200000 \\
        seed=0 arch=unet hydra/launcher=slurm_generic
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from experiments.configs import ExperimentConfig


@hydra.main(
    config_path="../configs",
    config_name="config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> dict:
    exp_cfg: ExperimentConfig = OmegaConf.to_object(cfg)
    # The schema declares `fidelity_values: tuple`, but OmegaConf hands back a list.
    exp_cfg.fidelity_values = tuple(exp_cfg.fidelity_values)

    folder = Path.cwd()  # Hydra's `job.chdir=true` puts us in results/<axis>/<folder>/

    if exp_cfg.axis == "truncation":
        from experiments.run_truncation import run_single_config
    elif exp_cfg.axis == "discretization":
        from experiments.run_discretization import run_single_config
    else:
        raise ValueError(f"Unknown axis {exp_cfg.axis!r}")

    # Returning the metrics dict lets Hydra's LogJobReturnCallback + our
    # SweepSummaryCallback surface per-job results in sweep logs.
    return run_single_config(exp_cfg, folder=folder)


if __name__ == "__main__":
    main()
