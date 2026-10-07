"""Fidelity-input sweep for the conditioning figure (GPU evaluation).

Evaluates each selected checkpoint on the test split with the conditioning input forced to a
constant value v (eval_epoch(forced_conditioning=v)), for
  - the 16 fidelities of the solver menu (in_menu=True, seen in training),
  - the 15 midpoints between consecutive menu levels (interpolation),
  - values beyond 1.0: 1.05, 1.10, 1.15, 1.20 (extrapolation).

Runs (finished wandb runs, Transolver, at the matched cells in CELLS):
  - camera_ready_hyperelasticity_solver_truncation_0: the conditioned runs, swept (swept=True).
  - camera_ready_hyperelasticity_solver_no_conditioning_0: the unconditioned ablation, trained
    with cond = 0. Not swept, since any other input is out of distribution: one evaluation at
    cond = 0, repeated along the sweep axis (swept=False).
  - the conditioning-scale ablations condscale1 and condscale50, swept like the main runs.

Writes paper/results/xi_sweep_hyperelasticity_solver.csv (one row per run and value; resumes
where it stopped, --force rebuilds). Takes a few GPU hours. Needs the checkpoints under
ckpts/camera_ready/ and the hyperelasticity solver dataset under data/.

Usage:  python xi_sweep_eval.py [-b 32] [--device cuda] [--force]
"""
import argparse
import os.path as osp
import re
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from mf_surrogates.eval.evaluate import hf_cost
from mf_surrogates.eval.plate_utils import eval_epoch
from mf_surrogates.utils import load_ckpt

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
DATA_PATH = osp.join(REPO, "data", "hyperelasticity", "solver_truncation", "processed")

EXP_UNCOND = "camera_ready_hyperelasticity_solver_no_conditioning_0"
EXP_COND = "camera_ready_hyperelasticity_solver_truncation_0"
EXP_COND50 = "camera_ready_hyperelasticity_solver_condscale50_0"  # cond_scale=50 pilot
EXP_COND1 = "camera_ready_hyperelasticity_solver_condscale1_0"    # unscaled embedding
MODEL = "Transolver"
# matched cells = per-B~ argmin (budget, n_samples_target); both cond and uncond runs
# are admitted only at these cells
CELLS = {(76000.0, 3178),      # B~=400  (lambda -6.25)
         (304000.0, 10578),    # B~=1600 (lambda -3.59)
         (1216000.0, 25238)}   # B~=6400 (lambda -0.31)
# checkpoint folder per experiment (all trained on the hyperelasticity solver dataset)
CKPT_DIRS = {
    EXP_UNCOND: osp.join(REPO, "ckpts", "camera_ready", "hyperelasticity_solver_no_conditioning"),
    EXP_COND: osp.join(REPO, "ckpts", "camera_ready", "hyperelasticity_solver"),
    EXP_COND50: osp.join(REPO, "ckpts", "camera_ready", "hyperelasticity_solver_condscale50"),
    EXP_COND1: osp.join(REPO, "ckpts", "camera_ready", "hyperelasticity_solver_condscale1"),
}
EXTRAPOLATION = [1.05, 1.10, 1.15, 1.20]
OUT_CSV = osp.join(REPO, "paper", "results", "xi_sweep_hyperelasticity_solver.csv")


def parse_args():
    p = argparse.ArgumentParser(
        description="Forced-conditioning (xi-hat) sweep for the conditioning figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-b", "--batch-size", type=int, default=32,
                   help="eval batch size (memory only; solver meshes are small)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device")
    p.add_argument("-f", "--force", action="store_true",
                   help="re-evaluate (run_id, value) pairs already in the CSV")
    p.add_argument("-o", "--out-csv", default=OUT_CSV, help="output CSV path")
    p.add_argument("--entity", default="setinek", help="wandb entity of the paper runs")
    p.add_argument("--project", default="multi-fidelity CFD", help="wandb project of the paper runs")
    return p.parse_args()


def launch_time(run_name):
    # run names are '<petname>_<YYYYMMDD>_<HHMMSS>' (see main.py)
    m = re.search(r"_(\d{8})_(\d{6})$", run_name)
    return m.group(1) + m.group(2) if m else "0"


def run_cell(r):
    hp = r.config.get("dataset", {}).get("hparams", {})
    return float(hp.get("budget", 0)), int(hp.get("n_samples_target", -1))


def select_runs(api, entity, project):
    """-> list of (experiment, run_name) at the matched CELLS, wandb-enumerated,
    deduped to the latest run per (experiment, seed, cell)."""
    sel = []
    for exp in (EXP_UNCOND, EXP_COND, EXP_COND50, EXP_COND1):
        runs = api.runs(f"{entity}/{project}",
                        filters={"config.logging.experiment_id": exp})
        for r in runs:
            hp = r.config.get("dataset", {}).get("hparams", {})
            if (r.state == "finished" and not hp.get("hf_only", False)
                    and r.config.get("model", {}).get("name") == MODEL
                    and run_cell(r) in CELLS):
                sel.append((exp, r))

    best = {}
    for exp, r in sel:
        key = (exp, int(r.config.get("seed", -1)), run_cell(r))
        if key not in best or launch_time(r.name) > launch_time(best[key][1].name):
            best[key] = (exp, r)
    return sorted(best.values(), key=lambda t: (t[0], int(t[1].config.get("seed", -1))))


def sweep_values(fidelities_linearized):
    menu = np.unique(np.round(np.asarray(fidelities_linearized, dtype=float), 8))
    mids = (menu[:-1] + menu[1:]) / 2
    vals = ([(float(v), True, False) for v in menu]
            + [(float(v), False, False) for v in mids]
            + [(float(v), False, True) for v in EXTRAPOLATION])
    return sorted(vals)  # (value, in_menu, extrapolation)


def main():
    args = parse_args()
    api = wandb.Api(timeout=120)
    selected = select_runs(api, args.entity, args.project)
    c_hf = hf_cost({"data_path": DATA_PATH, "cost_column": "solver_max_its"})
    print(f"selected {len(selected)} runs: "
          + ", ".join(f"{r.name}[{e.split('_')[-2]}]" for e, r in selected))

    done, rows = set(), []
    if osp.exists(args.out_csv) and not args.force:
        prev = pd.read_csv(args.out_csv)
        rows = prev.to_dict("records")
        done = {(r["run_id"], round(r["conditioning_fidelity"], 6)) for r in rows}

    testset_raw, values = None, None
    for exp, wr in selected:
        rid = wr.name
        path = osp.join(CKPT_DIRS[exp], rid, "best.pt")
        if not osp.exists(path):
            print(f"  WARNING: no ckpt for {rid} ({path}) -- skipped (rsync gap?)")
            continue
        cfg_changes = {"dataset.hparams.data_path": DATA_PATH}
        if testset_raw is None:
            testset_raw = load_ckpt(ckpt_path=path, load_model=False, load_opt=False,
                                    load_trainset=False, load_valset=False, load_testset=True,
                                    normalize_datasets=False, cfg_changes=cfg_changes)["testset"]
        ck = load_ckpt(ckpt_path=path, load_model=True, load_opt=False, load_trainset=False,
                       load_valset=False, load_testset=False, normalize_datasets=False,
                       cfg_changes=cfg_changes)
        model, cfg = ck["model"], ck["cfg"]
        if values is None:
            values = sweep_values(ck["fidelities_linearized"])
            print(f"sweep: {len(values)} values "
                  f"({sum(v[1] for v in values)} menu, {sum(v[2] for v in values)} extrapolation)")
        testset = deepcopy(testset_raw)
        testset.normalize(normalization_stats=ck["trainset_normalization_stats"])
        loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=testset.collate)
        fm = ck.get("fidelity_mix", None)
        f = np.array([t[2] for t in fm], dtype=float) if fm else np.array([])
        ds_h = cfg.dataset.hparams

        is_uncond = bool(ds_h.get("no_conditioning", False))
        base = dict(
            experiment=exp, run_id=rid, model_name=cfg.model.name, seed=int(cfg.seed),
            no_conditioning=is_uncond,
            b_tilde=int(round(float(ds_h.budget) / c_hf)),
            n_samples_target=int(ds_h.get("n_samples_target", -1)),
            lambd=float(ck["lambd"]) if ck.get("lambd") is not None else np.nan,
            xi_bar=float(f.mean()) if len(f) else np.nan,
        )
        todo = [(v, m, x) for v, m, x in values if (rid, round(v, 6)) not in done]
        if not todo:
            continue
        if is_uncond:
            # no forced sweep: cond input was 0 throughout training, so nonzero values are
            # OOD probes. One natural eval, replicated across the axis (swept=False).
            _, loss_dict = eval_epoch(model=model, valset=testset, valloader=loader,
                                      device=args.device, forced_conditioning=0.0)
            for v, in_menu, extrap in todo:
                rows.append(dict(**base, conditioning_fidelity=v, in_menu=in_menu,
                                 extrapolation=extrap, swept=False, **loss_dict))
            pd.DataFrame(rows).to_csv(args.out_csv, index=False)
            print(f"  {rid}: uncond -> 1 natural eval replicated over {len(todo)} values")
        else:
            for v, in_menu, extrap in tqdm(todo, desc=rid):
                _, loss_dict = eval_epoch(model=model, valset=testset, valloader=loader,
                                          device=args.device, forced_conditioning=v)
                rows.append(dict(**base, conditioning_fidelity=v, in_menu=in_menu,
                                 extrapolation=extrap, swept=True, **loss_dict))
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
