import torch
from torch import optim
from torch.optim import Adam, AdamW
from omegaconf import DictConfig, OmegaConf
from transformers import get_scheduler as get_scheduler_transformers_lib

from .loss import MSELoss, MSELossSplit, RelativeL2Loss, RelativeL2LossAirfoil


OPTIMIZERS = {
    "adam": Adam,
    "adamw": AdamW
}

LOSS_CRITERIA = {
    "rel_l2_loss": RelativeL2Loss,
    "rel_l2_airfoil": RelativeL2LossAirfoil,
    "mse": MSELoss,
    "mse_split": MSELossSplit,
}


def get_optimizer(cfg: DictConfig, model) -> optim.Optimizer:
    name = cfg.training.optimizer.name
    opt_hparams = OmegaConf.to_container(cfg.training.optimizer.hparams)
    opt_cls = OPTIMIZERS.get(name)
    if opt_cls is None:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer.name}")
    return opt_cls(
        params=model.parameters(),
        **opt_hparams
    )


def get_loss_criterion(cfg: DictConfig, dataset):
    name = cfg.training.loss.name
    loss_cls = LOSS_CRITERIA.get(name)
    if loss_cls is None:
        raise ValueError(f"Unsupported loss function: {name}")
    loss_hparams = {} if cfg.training.loss.hparams is None else OmegaConf.to_container(cfg.training.loss.hparams, resolve=True)
    return loss_cls(dataset=dataset, normalized=cfg.training.loss.normalized, **loss_hparams)


def get_scheduler(cfg: DictConfig, optimizer, num_training_steps):
    name = cfg.training.scheduler.name
    scheduler_hparams = OmegaConf.to_container(cfg.training.scheduler.hparams)
    if name in ["linear", "constant", "cosine"]:
        return get_scheduler_transformers_lib(
            name=name,
            optimizer=optimizer,
            num_training_steps=num_training_steps,
            **scheduler_hparams
        )
    elif name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=num_training_steps,
            **scheduler_hparams
        )
        return scheduler
    else:
        raise ValueError(f"Unsupported scheduler: {name}")
