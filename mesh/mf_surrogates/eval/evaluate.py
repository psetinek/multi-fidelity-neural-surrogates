"""Test-split evaluation of trained checkpoints (used by the root evaluate.py)."""

import json
import os.path as osp
from copy import deepcopy

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from mf_surrogates.utils import load_ckpt
from .airfoil_utils import eval_epoch_with_coeffs as airfoil_eval_epoch
from .plate_utils import eval_epoch as plate_eval_epoch

TEST_EVAL_FNS = {"AirfoilData": airfoil_eval_epoch, "PlateData": plate_eval_epoch}

# dataset hparams that only shape the training mixture, not the test split
TRAIN_MIX_HPARAMS = ("budget", "n_samples_target", "lambd", "lambda_bounds", "hf_only")


def hf_cost(dataset_hparams):
    """Mean cost of one high-fidelity sample (the highest fidelity level in metadata.csv)."""
    metadata_path = dataset_hparams.get("metadata_path") or osp.join(dataset_hparams["data_path"], "metadata.csv")
    metadata = pd.read_csv(metadata_path)
    hf = metadata[metadata["fidelity_raw"] == metadata["fidelity_raw"].max()]
    return float(hf[dataset_hparams["cost_column"]].mean())


def _test_split_key(dataset_cfg):
    cfg = OmegaConf.to_container(dataset_cfg, resolve=True)
    cfg["hparams"] = {k: v for k, v in cfg["hparams"].items() if k not in TRAIN_MIX_HPARAMS}
    return json.dumps(cfg, sort_keys=True)


def evaluate_checkpoint(ckpt_path, data_path=None, batch_size=32, device="cuda", cache=None):
    """Evaluate a checkpoint on the test split and return one result row.

    The row holds the run's training mixture (budget, samples, lambda, mean fidelity) and all
    metrics of the domain's eval function. cache (a dict) keeps test sets and HF costs across
    calls, so runs that share a test split load it only once.
    """
    cache = {} if cache is None else cache
    cfg_changes = {"dataset.hparams.data_path": data_path} if data_path else None
    ck = load_ckpt(ckpt_path=ckpt_path, load_model=True, load_opt=False, load_trainset=False, load_valset=False,
                   load_testset=False, normalize_datasets=False, cfg_changes=cfg_changes)
    cfg = ck["cfg"]
    hparams = OmegaConf.to_container(cfg.dataset.hparams, resolve=True)

    key = _test_split_key(cfg.dataset)
    if key not in cache:
        testset = load_ckpt(ckpt_path=ckpt_path, load_model=False, load_opt=False, load_trainset=False,
                            load_valset=False, load_testset=True, normalize_datasets=False,
                            cfg_changes=cfg_changes)["testset"]
        cache[key] = (testset, hf_cost(hparams))
    testset_raw, c_hf = cache[key]

    testset = deepcopy(testset_raw)
    testset.normalize(normalization_stats=ck["trainset_normalization_stats"])
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, collate_fn=testset.collate)
    _, metrics = TEST_EVAL_FNS[cfg.dataset.name](model=ck["model"], valset=testset, valloader=testloader, device=device)

    # fidelity_mix entries: (sample_id, fidelity_id, linearized fidelity, cost)
    fidelities = np.array([entry[2] for entry in ck["fidelity_mix"]], dtype=float) if ck["fidelity_mix"] else np.array([])
    return dict(
        run_id=cfg.logging.run_id,
        model_name=cfg.model.name,
        seed=int(cfg.seed),
        hf_only=bool(hparams["hf_only"]),
        no_conditioning=bool(hparams.get("no_conditioning", False)),
        budget_target=float(hparams["budget"]),
        budget_realized=float(ck["budget"]),
        b_tilde=int(round(float(hparams["budget"]) / c_hf)),
        n_samples_target=hparams.get("n_samples_target", np.nan),
        n_samples_realized=len(fidelities),
        n_hf=int(np.isclose(fidelities, 1.0).sum()) if len(fidelities) else 0,
        lambd=float(ck["lambd"]) if ck.get("lambd") is not None else np.nan,
        xi_bar=float(fidelities.mean()) if len(fidelities) else np.nan,
        ckpt_val_loss=float(ck["val_loss"]),
        **metrics,
    )
