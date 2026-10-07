from omegaconf import OmegaConf
from torch import nn

from .transolver import Transolver
from .ab_upt import ABUPT

from .utils.registry import get_model_class


__all__ = ["Transolver", "ABUPT"]

def get_model(cfg) -> nn.Module:
    ModelClass = get_model_class(cfg.model.name)
    hparams = OmegaConf.to_container(cfg.model.hparams)
    return ModelClass(
        input_channels=cfg.dataset.input_channels,
        output_channels=cfg.dataset.output_channels,
        space=cfg.dataset.space,
        n_cond=cfg.dataset.n_cond,
        **hparams,
    )
