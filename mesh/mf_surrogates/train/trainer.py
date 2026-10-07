from collections import defaultdict
from dataclasses import replace
from omegaconf import OmegaConf, DictConfig
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time

from .utils import get_optimizer, get_loss_criterion, get_scheduler
from mf_surrogates.data import AirfoilData, PlateData
from mf_surrogates.eval.airfoil_utils import eval_epoch as airfoil_eval_epoch
from mf_surrogates.eval.plate_utils import eval_epoch as plate_eval_epoch


EVAL_EPOCH_MAPPING = {
    AirfoilData: airfoil_eval_epoch,
    PlateData: plate_eval_epoch,
}

SUMMARY_SELECTORS = {
    AirfoilData: ["train/rel_l2_error_per_channel", "val/rel_l2_volume", "val/rel_l2_surface", "val/rel_l2_loss_fields_avg"],
    PlateData:   ["train/rel_l2_error_per_channel", "val/rel_l2_loss_channels_avg", "val/rel_l2_loss_fields_avg", "val/nmse_avg"],
}


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _subset_node_batch(batch, node_indices: torch.Tensor):
    node_indices = node_indices.to(device=batch.x.device, dtype=torch.long)
    updates = {
        "coords": batch.coords.index_select(0, node_indices),
        "x": batch.x.index_select(0, node_indices),
        "y": batch.y.index_select(0, node_indices),
    }
    if batch.batch_index is not None:
        updates["batch_index"] = batch.batch_index.index_select(0, node_indices)
    if batch.pad_mask is not None and batch.pad_mask.ndim > 0 and batch.pad_mask.shape[0] == batch.x.shape[0]:
        updates["pad_mask"] = batch.pad_mask.index_select(0, node_indices)
    return replace(batch, **updates)


class Trainer:
    def __init__(
        self,
        trainset: Dataset,
        valset: Dataset,
        trainloader: DataLoader,
        valloader: DataLoader,
        model: nn.Module,
        device: torch.device,
        use_ddp: bool,
        rank: int,
        logger = None,
        cfg: DictConfig = None,
    ):
        self.trainset, self.trainloader = trainset, trainloader
        self.valset, self.valloader = valset, valloader

        self.eval_epoch_fn = EVAL_EPOCH_MAPPING[type(self.trainset)]
        self.device = device
        self.use_ddp = use_ddp
        self.rank = rank
        self.n_steps = cfg.training.n_steps
        self.eval_every = cfg.training.eval_every
        self.early_stopping_patience = cfg.training.early_stopping_patience
        self.clip_grad = cfg.training.gradient_clipping
        self.use_amp = cfg.training.use_amp
        self.cfg = cfg
        self.logger = logger

        self.model = model

        self.optimizer = get_optimizer(cfg=cfg, model=self.model)

        self.loss_criterion = get_loss_criterion(cfg=cfg, dataset=self.trainset).to(device)

        self.scheduler = get_scheduler(
            cfg=cfg,
            optimizer=self.optimizer,
            num_training_steps=self.n_steps,
        )

        self.scaler = torch.amp.GradScaler(
            str(self.device), enabled=cfg.training.use_amp
        )

    def run(self):
        val_loss_min = float("inf")
        early_stop_counter = 0
        stop_training = False

        global_step = 0
        epoch = 1

        # variables to track training loss over the evaluation window
        train_loss_accum = 0.0
        loss_components_accum = defaultdict(float)
        sum_bs = 0

        start = time.perf_counter()

        while global_step < self.n_steps:
            if self.use_ddp:
                self.trainloader.sampler.set_epoch(epoch)

            self.model.train()

            for batch in self.trainloader:
                if global_step >= self.n_steps:
                    break

                # prep batch
                batch = batch.to(self.device)
                bs = int(batch.batch_index.max().item()) + 1

                model_for_hooks = _unwrap_model(self.model)
                query_indices = None
                loss_batch = batch
                if (self.model.training
                    and hasattr(model_for_hooks, "sample_training_query_indices")
                ):
                    query_indices = model_for_hooks.sample_training_query_indices(
                        x=batch.x,
                        batch_index=batch.batch_index,
                    )
                    loss_batch = _subset_node_batch(batch, query_indices)

                # forward
                with torch.autocast(
                    device_type=str(self.device.type), dtype=torch.bfloat16, enabled=self.use_amp
                ):
                    if query_indices is not None:
                        preds = self.model(
                            x=batch.x,
                            mesh_coords=batch.coords,
                            cond=batch.cond,
                            batch_index=batch.batch_index,
                            query_indices=query_indices,
                        )
                    else:
                        preds = self.model(
                            x=batch.x,
                            mesh_coords=batch.coords,
                            cond=batch.cond,
                            batch_index=batch.batch_index,
                        )
                    loss, components = self.loss_criterion(preds, loss_batch)

                # backward
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                if self.clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
  
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                global_step += 1

                # metrics accumulation
                train_loss_accum += loss.item() * bs
                for k, v in components.items():
                    loss_components_accum[k] += v * bs
                sum_bs += bs

                # eval and logging
                if not self.rank:
                    if (global_step % self.eval_every) == 0 or global_step == self.n_steps:
                        end = time.perf_counter()
                        print(f"Step {global_step} training time since last eval: {end - start:.2f} seconds")

                        # finalize training metrics for the logging window
                        lr = self.scheduler.get_last_lr()[0]
                        train_dict = {
                            f"train/{getattr(self.loss_criterion, 'name', 'loss')}": train_loss_accum / sum_bs,
                            "train/lr": lr,
                        }
                        train_dict.update({f"train/{k}": v / sum_bs for k, v in loss_components_accum.items()})

                        # eval
                        start_eval = time.perf_counter()
                        # evaluate the unwrapped model: a DDP forward on rank 0 alone would issue
                        # collectives (buffer broadcast) that the other ranks never join
                        _, eval_dict = self.eval_epoch_fn(_unwrap_model(self.model), self.valset, self.valloader, self.device)
                        end_eval = time.perf_counter()
                        print(f"Step {global_step} evaluation time: {end_eval - start_eval:.2f} seconds")

                        # logging
                        val_loss = eval_dict["val/rel_l2_loss_fields_avg"]
                        log_dict = train_dict | eval_dict
                        self.logger.log(
                            step=global_step,
                            log_dict=log_dict,
                        )

                        # checkpointing
                        ckpt_dict = {
                            "step": global_step,
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "val_loss": val_loss,
                            "trainset_normalization_stats": self.trainset.normalization_stats,
                            "cfg": OmegaConf.to_container(self.cfg),
                            **getattr(self.trainset, "ckpt_metadata", {}),
                        }

                        if val_loss < val_loss_min:
                            early_stop_counter = 0
                            val_loss_min = val_loss
                            if self.logger and hasattr(self.logger, "wandb_writer") and self.logger.wandb_writer is not None:
                                for key in SUMMARY_SELECTORS.get(type(self.trainset), []):
                                    if key in log_dict:
                                        metric_name = f"best_model/{key.replace('/', '_')}"
                                        self.logger.wandb_writer.summary[metric_name] = log_dict[key]
                                self.logger.wandb_writer.summary["best_model/step"] = global_step
                            torch.save(ckpt_dict, f"{self.cfg.output_path}/{self.cfg.logging.run_id}/best.pt")
                        else:
                            early_stop_counter += self.eval_every
                            if early_stop_counter >= self.early_stopping_patience:
                                print(f"Early stopping triggered at step {global_step} after {self.early_stopping_patience} steps without improvement.")
                                stop_training = True

                        if not stop_training:
                            torch.save(ckpt_dict, f"{self.cfg.output_path}/{self.cfg.logging.run_id}/last.pt")

                        # reset accumulators and timer
                        train_loss_accum = 0.0
                        loss_components_accum = defaultdict(float)
                        sum_bs = 0
                        self.model.train()
                        start = time.perf_counter()

                # only rank 0 evaluates, so we have to stop other rans aswess
                if (global_step % self.eval_every) == 0 or global_step == self.n_steps:
                    if self.use_ddp:
                        stop_flag = torch.tensor(int(stop_training), device=self.device)
                        dist.broadcast(stop_flag, src=0)
                        stop_training = bool(stop_flag.item())
                    if stop_training:
                        return
            epoch += 1
