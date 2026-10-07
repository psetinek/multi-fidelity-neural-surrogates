import os
from datetime import timedelta
import random
import socket
import numpy as np
import torch
import torch.distributed as dist
import logging
from typing import Optional, Callable, Dict, Any

from omegaconf import OmegaConf

from mf_surrogates.models import get_model
from mf_surrogates.train.utils import get_optimizer
from mf_surrogates.data import get_data


def set_seed(seed):
    # seeding
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # quick tf32, flash_sdpa settings
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    torch.backends.cuda.enable_flash_sdp(True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    print(f"Random seed set as {seed}")


class Logger(logging.Logger):
    def __init__(
        self, name: str, n_steps: int = 0, wandb_writer: Optional[Callable] = None
    ):
        super().__init__(name, level=logging.INFO)

        self.n_steps = n_steps
        self.wandb_writer = wandb_writer

        if not self.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

    def log(
        self,
        step: int,
        log_dict: Dict[str, Any],
        *args,
        **kwargs,
    ):
        if self.wandb_writer is not None:
            # log to wandb
            self.wandb_writer.log(log_dict, step=step)

        step_s = str(step).zfill(len(str(self.n_steps)))
        msg = f"[{step_s}] "
        for i, (k, v) in enumerate(log_dict.items()):
            if v is not None and v != float("-inf"):
                msg += f"{k}: {v:.5f}"
                if i != len(log_dict) - 1:
                    msg += ", "

        super().info(msg, *args, **kwargs)


def _apply_cfg_changes(cfg, cfg_changes=None, *, allow_new_keys=True):
    """
    Supports:
      - nested dicts: {"training": {"n_epochs": 100}}
      - dot-keys:     {"training.n_epochs": 100, "logging.run_id": "foo"}
      - dotlist:      ["training.n_epochs=100", "logging.run_id=foo"]
    """
    if not cfg_changes:
        return cfg

    # Ensure we can write new keys if desired
    if allow_new_keys:
        OmegaConf.set_struct(cfg, False)

    # Case 1: list/tuple of dotlist strings
    if isinstance(cfg_changes, (list, tuple)):
        dot = OmegaConf.from_dotlist(list(cfg_changes))
        cfg = OmegaConf.merge(cfg, dot)
        return cfg

    # Case 2: dict (may contain nested dicts and/or dotted keys)
    if isinstance(cfg_changes, dict):
        # Merge nested dicts in one shot
        nested_part = {k: v for k, v in cfg_changes.items()
                       if isinstance(v, dict) and ("." not in k and "[" not in k)}
        if nested_part:
            cfg = OmegaConf.merge(cfg, OmegaConf.create(nested_part))

        # Apply dotted (path-like) keys individually
        dotted_part = {k: v for k, v in cfg_changes.items()
                       if not isinstance(v, dict) or ("." in k or "[" in k)}
        for k, v in dotted_part.items():
            OmegaConf.update(cfg, k, v, merge=True)
        return cfg

    raise TypeError("cfg_changes must be a dict, list/tuple (dotlist), or None")


def load_ckpt(ckpt_path, load_model=True, load_opt=True, load_trainset=True, load_valset=True, load_testset=True, normalize_datasets=True, cfg_changes=None):
    dict_out = {}
    ckpt_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt_dict["cfg"])
    cfg = _apply_cfg_changes(cfg, cfg_changes, allow_new_keys=False)
    dict_out["budget"] = ckpt_dict["budget_used"]
    dict_out["fidelities_linearized"] = ckpt_dict["fidelities_linearized"]
    dict_out["fidelity_mix"] = ckpt_dict["fidelity_mix"]
    dict_out["lambd"] = ckpt_dict.get("lambd", None)  # realized mixing lambda (config value is null when solved from n_samples_target)
    dict_out["epoch"] = ckpt_dict["epoch"]
    dict_out["val_loss"] = ckpt_dict["val_loss"]
    dict_out["trainset_normalization_stats"] = ckpt_dict["trainset_normalization_stats"]

    if load_model:
        model = get_model(cfg=cfg)
        state_dict = ckpt_dict["model_state_dict"]

        # Check if keys are prefixed with "module." (from DDP)
        if any(k.startswith("module.") for k in state_dict.keys()):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k.replace("module.", "")] = v
            state_dict = new_state_dict

        model.load_state_dict(state_dict, strict=True)
        dict_out["model"] = model

    if load_opt:
        assert load_model, "Model must be loaded if you want to load the optimizer!"
        opt = get_optimizer(cfg=cfg, model=model)
        opt.load_state_dict(ckpt_dict["optimizer_state_dict"])
        dict_out["optimizer"] = opt

    if load_trainset:
        trainset, trainloader = get_data(cfg=cfg, split="train", normalize=normalize_datasets)
        # check if normalization stats are the same as the ones saved
        for key in ckpt_dict["trainset_normalization_stats"].keys():
            saved_value = ckpt_dict["trainset_normalization_stats"][key]
            new_value = trainset.normalization_stats[key]
            assert torch.allclose(saved_value, new_value) if saved_value is not None else True
        dict_out["trainset"] = trainset
        dict_out["trainloader"] = trainloader

    if load_valset:
        valset, valloader = get_data(cfg=cfg, split="val", normalize=normalize_datasets, normalization_stats=ckpt_dict["trainset_normalization_stats"])
        dict_out["valset"] = valset
        dict_out["valloader"] = valloader

    if load_testset:
        testset, testloader = get_data(cfg=cfg, split="test", normalize=normalize_datasets, normalization_stats=ckpt_dict["trainset_normalization_stats"])
        dict_out["testset"] = testset
        dict_out["testloader"] = testloader

    dict_out["cfg"] = cfg
    return dict_out


def find_free_port():
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return s.getsockname()[1]  # Return the port number assigned.


def ddp_setup(rank, world_size):
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=20)
    )
