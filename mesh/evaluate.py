import argparse
import os
import os.path as osp
import re

import pandas as pd
import torch
import wandb
import yaml
from tqdm import tqdm

from mf_surrogates.eval.evaluate import evaluate_checkpoint, hf_cost

HERE = osp.dirname(osp.abspath(__file__))
DEFAULT_PROJECT = yaml.safe_load(open(osp.join(HERE, "configs", "logging", "wandb.yaml")))["wandb_project"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all finished runs of an experiment on the test split, "
                                            "one CSV row per run.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--experiment-id", required=True, help="logging.experiment_id of the runs")
    p.add_argument("--entity", default=None, help="wandb entity (default: your default entity)")
    p.add_argument("--project", default=DEFAULT_PROJECT, help="wandb project")
    p.add_argument("--ckpt-dir", default=None,
                   help="directory holding <run name>/<ckpt-name> (default: each run's output_path)")
    p.add_argument("--ckpt-name", default="best.pt", help="checkpoint file of a run")
    p.add_argument("--data-path", default=None,
                   help="dataset location, if it differs from the one the runs were trained with")
    p.add_argument("--model", default=None, help="only evaluate runs of this model (e.g. Transolver, ABUPT)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="evaluation batch size (metrics are per sample, this only affects memory)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None,
                   help="output CSV (default: results/<experiment-id>.csv, or "
                        "results/<experiment-id>__<model>.csv with --model)")
    p.add_argument("--force", action="store_true", help="re-evaluate runs already in the CSV")
    p.add_argument("--no-dedupe", action="store_true", help="keep every finished run of a cell")
    return p.parse_args()


def launch_time(run_name):
    # run names are '<petname>_<YYYYMMDD>_<HHMMSS>' (see main.py)
    m = re.search(r"_(\d{8})_(\d{6})$", run_name)
    return m.group(1) + m.group(2) if m else "0"


def select_runs(runs, data_path, dedupe):
    """Finished runs, deduplicated per cell (keep the latest launch)."""
    hf_costs, cells = {}, {}
    for run in runs:
        hp = dict(run.config["dataset"]["hparams"])
        if data_path:
            hp["data_path"] = data_path
        metadata = hp.get("metadata_path") or osp.join(hp["data_path"], "metadata.csv")
        if metadata not in hf_costs:
            hf_costs[metadata] = hf_cost(hp)
        hf_only = bool(hp.get("hf_only", False))
        cell = (run.config["model"]["name"], hf_only, int(round(float(hp["budget"]) / hf_costs[metadata])),
                -1 if hf_only else int(hp.get("n_samples_target", -1)), int(run.config["seed"]))
        cells.setdefault(cell, []).append(run)
    keep, dropped = [], []
    for cell_runs in cells.values():
        cell_runs = sorted(cell_runs, key=lambda r: launch_time(r.name))
        if dedupe:
            keep.append(cell_runs[-1])
            dropped += cell_runs[:-1]
        else:
            keep += cell_runs
    if dropped:
        print(f"dedupe: dropped {len(dropped)} superseded runs: {[r.name for r in dropped][:4]}"
              f"{' ...' if len(dropped) > 4 else ''}")
    return keep


def main():
    args = parse_args()
    out = args.out or osp.join("results", f"{args.experiment_id}{f'__{args.model}' if args.model else ''}.csv")
    os.makedirs(osp.dirname(out) or ".", exist_ok=True)

    api = wandb.Api(timeout=120)
    path = f"{args.entity}/{args.project}" if args.entity else args.project
    runs = api.runs(path, filters={"config.logging.experiment_id": args.experiment_id}, per_page=500)
    finished = [r for r in runs if r.state == "finished"
                and (args.model is None or r.config["model"]["name"] == args.model)]
    keep = select_runs(finished, args.data_path, dedupe=not args.no_dedupe)

    ckpts = {}
    for run in keep:
        ckpt = osp.join(args.ckpt_dir or run.config["output_path"], run.name, args.ckpt_name)
        ckpts[run.name] = ckpt
    missing = [name for name, ckpt in ckpts.items() if not osp.exists(ckpt)]
    if missing:
        print(f"WARNING: no {args.ckpt_name} for {len(missing)} finished runs (skipped): {missing[:4]}"
              f"{' ...' if len(missing) > 4 else ''}")
    run_names = sorted(name for name in ckpts if name not in missing)

    rows, done = [], set()
    if osp.exists(out) and not args.force:
        previous = pd.read_csv(out, float_precision="round_trip")  # keep earlier rows bit-exact
        previous = previous[previous.run_id.isin(run_names)]  # drop rows of runs no longer selected
        rows, done = previous.to_dict("records"), set(previous.run_id)
    todo = [name for name in run_names if name not in done]
    print(f"{args.experiment_id}: {len(finished)} finished runs, {len(run_names)} selected with checkpoint, "
          f"{len(todo)} to evaluate")

    cache = {}
    for name in tqdm(todo):
        row = evaluate_checkpoint(ckpts[name], data_path=args.data_path, batch_size=args.batch_size,
                                  device=args.device, cache=cache)
        rows.append({**row, "run_id": name})
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
