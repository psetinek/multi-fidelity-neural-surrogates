from typing import Literal
import warnings
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


from .registry import get_dataset_class
from .airfoil import AirfoilData
from .plate import PlateData


__all__ = ["AirfoilData", "PlateData"]


def get_data(
    cfg: DictConfig,
    split: Literal["train", "val", "test"] = "train",
    normalize: bool = True,
    normalization_stats: dict = None,
    use_ddp: bool = False,
    world_size: int =1,
) -> Dataset:
    # create dataset
    DatasetClass = get_dataset_class(cfg.dataset.name)
    dataset_hparams = OmegaConf.to_container(cfg.dataset.hparams)
    ss = np.random.SeedSequence(cfg.dataset.composition_seed)
    child_ss = ss.spawn(1)
    rng_ds = np.random.default_rng(child_ss[0])    
    dataset = DatasetClass(split=split, n_nodes_subsampling=cfg.training.n_nodes_subsampling if split == "train" else None, composition_rng=rng_ds, **dataset_hparams)
    if normalize:
        if normalization_stats is None and split in ["val", "test"]:
            warnings.warn(
                f"Careful, you are trying to load a normalized {split} dataset but have\
                      not specified any normalization stats. This causes data leakage!"
            )
        dataset.normalize(normalization_stats=normalization_stats)

    # create dataloader
    generator = torch.Generator().manual_seed(cfg.dataset.shuffling_seed)
    effective_batch_size = cfg.training.train_batch_size // world_size
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=effective_batch_size if split == "train" else cfg.training.val_batch_size,
        shuffle= True if (split == "train" and not use_ddp) else False,
        collate_fn=dataset.collate,
        num_workers=cfg.training.num_workers,
        generator=generator,
        sampler=DistributedSampler(dataset, shuffle=True) if (use_ddp and split=="train") else None,
    )

    return dataset, dataloader
