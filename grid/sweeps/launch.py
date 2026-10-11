"""Reproduce the paper's sweeps as Hydra multiruns.

The λ × B grids in the paper are irregular: the main sweeps were densified
around the interesting budgets over time. Rather than hand-maintaining them
in shell scripts, the exact set of cells is read from the paper's results
CSV (`data/results.csv`), and this script turns it back into
`python -m experiments.run -m ...` calls. Each call covers one
(axis, dim, architecture, ablation setting, budget) group, with the λs and
seeds of that group as comma-separated Hydra sweep values. On SLURM, each
call becomes one job array.

Outputs go to the directory layout that `scripts/build_results_csv.py`
expects (see EXPERIMENTS below).

Usage (run from the repository root):

    # Print the commands for one experiment (dry run, the default).
    python sweeps/launch.py --experiment dst

    # Submit them as SLURM arrays via a Hydra launcher config.
    python sweeps/launch.py --experiment dst --launcher slurm_generic --run

    # Subset: only 1D UNet cells, seeds 0-2; forward extra Hydra overrides.
    python sweeps/launch.py --experiment main --dim 1 --arch unet \\
        --seeds 0 1 2 --launcher lrz_mig --run \\
        -- hydra.sweeper.max_batch_size=40 hydra.launcher.timeout_min=120

    # Summary of cells per experiment.
    python sweeps/launch.py --summary
"""

import argparse
import shlex
import subprocess
import sys
from itertools import groupby
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GRID = REPO / "data" / "results.csv"

DEFAULT_STEPS = 20000

# Bimodal ablation: only the two menu endpoints (the top matches the standard
# menu, so `xi_max` in the CSV does not distinguish these menus).
BIMODAL_MENU = {"truncation": "[1,7]", "discretization": "[20,96]"}


def _steps_overrides(r):
    steps = int(r.num_training_steps)
    if steps == DEFAULT_STEPS:
        return []
    # Warmup is 10% of training in every run of the paper.
    return [f"num_training_steps={steps}", f"num_warmup_steps={steps // 10}"]


# Per experiment: output dir, per-run subdir template, and a function that
# maps one grid row to the Hydra overrides beyond axis/arch/λ/B/seed.
# `${...}` are Hydra interpolations, resolved per run.
EXPERIMENTS = {
    "main": dict(
        sweep_dir="results/${axis}",
        subdir="${architecture}_dim_${dim}_lambd_${lambd}_budget_${budget}_seed_${seed}",
        overrides=lambda r: [],
    ),
    "endpoint": dict(
        sweep_dir="results/endpoint_ablation/${axis}/kmax_{xi_max}",
        subdir="${architecture}_lambd_${lambd}_budget_${budget}_seed_${seed}",
        overrides=lambda r: [
            f"fidelity_values=[{','.join(str(k) for k in range(1, int(r.xi_max) + 1))}]",
            "eval_super_reference=true",
        ],
    ),
    "arch": dict(
        sweep_dir="results/arch_ablation/${axis}",
        subdir="${architecture}_dim_${dim}_bc_${base_channels}_lambd_${lambd}"
        "_budget_${budget}_seed_${seed}_steps_${num_training_steps}",
        overrides=lambda r: [f"base_channels={int(r.base_channels)}"]
        + _steps_overrides(r),
    ),
    "duration": dict(
        sweep_dir="results/duration_ablation/${axis}",
        subdir="${architecture}_dim_${dim}_lambd_${lambd}_budget_${budget}"
        "_seed_${seed}_steps_${num_training_steps}",
        overrides=_steps_overrides,
    ),
    "dst": dict(
        sweep_dir="results/dst_ablation/${axis}",
        subdir="${architecture}_dim_${dim}_lambd_${lambd}_budget_${budget}_seed_${seed}",
        overrides=lambda r: ["dst_interpolation=true"],
    ),
    "dual_reference": dict(
        sweep_dir="results/dual_reference/${axis}",
        subdir="${architecture}_dim_${dim}_lambd_${lambd}_budget_${budget}_seed_${seed}",
        overrides=lambda r: ["eval_super_reference=true"],
    ),
    "bimodal": dict(
        sweep_dir="results/bimodal_ablation/${axis}",
        subdir="${architecture}_dim_${dim}_bimodal_lambd_${lambd}_budget_${budget}_seed_${seed}",
        overrides=lambda r: [f"fidelity_values={BIMODAL_MENU[r.axis]}"],
    ),
}


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def build_commands(cells: pd.DataFrame, experiment: str, launcher, extra):
    """One Hydra multirun per (axis, dim, arch, overrides, budget) group.

    Within a group, the (λ, seed) cells are emitted as a Cartesian product
    when they form one; otherwise the group is split per seed.
    """
    spec = EXPERIMENTS[experiment]
    rows = []
    for r in cells.itertuples(index=False):
        ov = tuple(spec["overrides"](r))
        sweep_dir = spec["sweep_dir"].replace("{xi_max}", str(int(r.xi_max)))
        key = (r.axis, int(r.dim), r.architecture, ov, sweep_dir, float(r.budget))
        rows.append((key, float(r.lambd), int(r.seed)))
    rows.sort(key=lambda t: t[0])

    commands = []
    for key, grp in groupby(rows, key=lambda t: t[0]):
        axis, dim, arch, ov, sweep_dir, budget = key
        pairs = {(lam, s) for _, lam, s in grp}
        lambdas = sorted({lam for lam, _ in pairs})
        seeds = sorted({s for _, s in pairs})
        if len(pairs) == len(lambdas) * len(seeds):
            chunks = [(lambdas, seeds)]
        else:
            chunks = [
                (sorted(lam for lam, s2 in pairs if s2 == s), [s]) for s in seeds
            ]
        for lams, ss in chunks:
            cmd = [
                sys.executable, "-m", "experiments.run", "-m",
                f"axis={axis}_{dim}d",
                f"arch={arch}",
                *ov,
                f"lambd={','.join(_fmt(x) for x in lams)}",
                f"budget={_fmt(budget)}",
                f"seed={','.join(str(s) for s in ss)}",
                f"hydra.sweep.dir={sweep_dir}",
                f"hydra.sweep.subdir={spec['subdir']}",
            ]
            if launcher:
                cmd.append(f"hydra/launcher={launcher}")
            cmd.extend(extra)
            commands.append((cmd, len(lams) * len(ss)))
    return commands


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce the paper's sweeps as Hydra multiruns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Anything after `--` is appended to every Hydra command.",
    )
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS))
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID,
                        help="CSV listing the cells to run (default: data/results.csv)")
    parser.add_argument("--axis", choices=["truncation", "discretization"])
    parser.add_argument("--dim", type=int, choices=[1, 2, 3])
    parser.add_argument("--arch", choices=["unet", "fno"])
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--launcher", help="Hydra launcher config, e.g. slurm_generic")
    parser.add_argument("--run", action="store_true",
                        help="execute the commands (default: print them)")
    parser.add_argument("--summary", action="store_true",
                        help="print cell counts per experiment and exit")
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = parser.parse_args(argv)

    grid = pd.read_csv(args.grid)
    if args.summary:
        print(grid.groupby(["experiment", "axis", "dim", "architecture"])
              .size().rename("cells").to_string())
        return
    if args.experiment is None:
        parser.error("--experiment is required (or use --summary)")

    cells = grid[grid["experiment"] == args.experiment]
    if args.axis:
        cells = cells[cells["axis"] == args.axis]
    if args.dim:
        cells = cells[cells["dim"] == args.dim]
    if args.arch:
        cells = cells[cells["architecture"] == args.arch]
    if args.seeds is not None:
        cells = cells[cells["seed"].isin(args.seeds)]
    if cells.empty:
        sys.exit("No cells match the selection.")

    commands = build_commands(cells, args.experiment, args.launcher, extra)
    total = sum(n for _, n in commands)
    print(f"# {args.experiment}: {total} runs in {len(commands)} Hydra multiruns",
          file=sys.stderr)

    for i, (cmd, n) in enumerate(commands, 1):
        printable = shlex.join(["python", *cmd[1:]])
        if not args.run:
            print(printable)
            continue
        print(f"[{i}/{len(commands)}] ({n} runs) {printable}", flush=True)
        rc = subprocess.run(cmd, cwd=REPO).returncode
        if rc != 0:
            print(f"  -> exited with {rc}; continuing", file=sys.stderr)


if __name__ == "__main__":
    main()
