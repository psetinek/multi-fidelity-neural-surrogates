import warnings
from omegaconf import DictConfig, OmegaConf
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from mf_surrogates.data import get_data
from mf_surrogates.models import get_model
from mf_surrogates.train import Trainer
from mf_surrogates.utils import Logger, ddp_setup


def run(rank: int, cfg: DictConfig, world_size: int = 1) -> None:
    use_ddp = False
    try:
        if cfg.training.device == "cpu":
            device = torch.device("cpu")
        elif not torch.cuda.is_available():
            warnings.warn("CUDA is not available, training on CPU!")
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{rank}")
        if cfg.use_ddp and world_size > 1:
            ddp_setup(rank, world_size)
            use_ddp = True
        else:
            use_ddp = False

        # setup logging
        if not rank:  # only log on root process
            wandb_run = None
            if cfg.logging.writer == "wandb":
                config = OmegaConf.to_container(cfg)
                wandb_run = wandb.init(
                    project=cfg.logging.wandb_project,
                    name=cfg.logging.run_id,
                    save_code=True,
                    config=config,
                )
            logger = Logger(__name__, n_steps=cfg.training.n_steps, wandb_writer=wandb_run)
        else:
            logger = None

        # load model
        model = get_model(cfg=cfg)
        if cfg.training.compile:
            model.compile(dynamic=True)

        model.to(device)
        if use_ddp:
            model = DDP(model, device_ids=[rank])
        print(
            f"Model parameters: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M"
        )

        # load datasets and dataloaders
        trainset, trainloader = get_data(cfg=cfg, split="train", normalize=True, use_ddp=use_ddp, world_size=world_size)
        valset, valloader = get_data(
            cfg=cfg,
            split="val",
            normalize=True,
            normalization_stats=trainset.normalization_stats,
            world_size=world_size,
        )

        trainer = Trainer(
            trainset=trainset,
            valset=valset,
            trainloader=trainloader,
            valloader=valloader,
            model=model,
            device=device,
            use_ddp=use_ddp,
            rank=rank,
            logger=logger,
            cfg=cfg,
        )

        trainer.run()

        # close wandb logger
        if cfg.logging.writer == "wandb":
            wandb.finish()
    finally:
        if use_ddp:
            dist.destroy_process_group()
