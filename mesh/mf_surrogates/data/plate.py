from typing import Optional
from dataclasses import dataclass

import torch

from .registry import register_dataset
from .base import BaseSample, BaseMFDataset


@dataclass(kw_only=True)
class PlateSample(BaseSample):
    # for evaluation & metadata
    compliance: Optional[torch.Tensor] = None
    strain_energy: Optional[torch.Tensor] = None

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if self.compliance is not None:
            self.compliance = self.compliance.to(device, **kwargs)
        if self.strain_energy is not None:
            self.strain_energy = self.strain_energy.to(device, **kwargs)
        return self


@register_dataset()
class PlateData(BaseMFDataset):
    def __init__(self, no_conditioning=False, **kwargs):
        # for the ablation: serve cond=0 for every sample (train and val alike)
        self.no_conditioning = no_conditioning
        super().__init__(**kwargs)

    @property
    def x_channel_names(self):
        return ["force", "is_boundary", "sdf", "normals"]

    @property
    def y_channel_names(self):
        return ["vm"]  # prediction target is only vm

    def _init_data_lists(self):
        super()._init_data_lists()
        self.compliance, self.strain_energy = [], []

    def _load_and_append_sample(self, sample_id, fidelity_str):
        super()._load_and_append_sample(sample_id, fidelity_str)
        metadata = self.metadata[-1] 
        self.compliance.append(torch.tensor(metadata.get("compliance_fem", 0.0), dtype=self.dtype))
        self.strain_energy.append(torch.tensor(metadata.get("strain_energy_fem", 0.0), dtype=self.dtype))

    def normalize(self, normalization_stats=None):
        if self.normalized:
            return

        if normalization_stats is None:
            self._compute_normalization_stats()
        else:
            self.normalization_stats = normalization_stats

        stats = self.normalization_stats

        for i, sample in enumerate(self.data):
            # min-max positions to [-1, 1]
            sample[:, self._channels["coords"]] = 2 * (sample[:, self._channels["coords"]] - stats["coords_min"]) / (stats["coords_max"] - stats["coords_min"]) - 1
            # z-score u, sigma, epsilon and cond
            sample[:, self._channels["u"]] = (sample[:, self._channels["u"]] - stats["u_mean"]) / stats["u_std"]
            sample[:, self._channels["sigma"]] = (sample[:, self._channels["sigma"]] - stats["sigma_mean"]) / stats["sigma_std"]
            sample[:, self._channels["epsilon"]] = (sample[:, self._channels["epsilon"]] - stats["epsilon_mean"]) / stats["epsilon_std"]
            sample[:, self._channels["vm"]] = (sample[:, self._channels["vm"]] - stats["vm_mean"]) / stats["vm_std"]
            # z-score sdf
            sample[:, self._channels["sdf"]] = (sample[:, self._channels["sdf"]] - stats["sdf_mean"]) / stats["sdf_std"]
            # fidelity conditioning normalization
            if stats["cond_min"] is not None and stats["cond_max"] is not None:
                self.cond[i] = (self.cond[i] - stats["cond_min"]) / (stats["cond_max"] - stats["cond_min"])
            # force field (only count nodes that are non 0 at the edge)
            mask = sample[:, self._channels["force"]] != 0
            sample[:, self._channels["force"]] = (sample[:, self._channels["force"]] - stats["force_mean"]) / stats["force_std"]
            sample[:, self._channels["force"]] = torch.where(mask, sample[:, self._channels["force"]], torch.zeros_like(sample[:, self._channels["force"]]))

        self.normalized = True


    def denormalize(self, fields, coords=None):
        # z-score inverse per target field, following y_channel_names
        stats = self.normalization_stats
        fields_denormalized = torch.cat(
            [fields[:, _slice] * stats[f"{name}_std"].to(fields.device) + stats[f"{name}_mean"].to(fields.device)
             for name, _slice in self.y_channels.items()],
            dim=-1,
        )
        if coords is not None:
            return fields_denormalized, self.denormalize_coords(coords)
        return fields_denormalized


    def denormalize_coords(self, coords):
        coords_min = self.normalization_stats["coords_min"].to(coords.device)
        coords_max = self.normalization_stats["coords_max"].to(coords.device)
        coords_denormalized = (coords + 1) * (coords_max - coords_min) / 2 + coords_min
        return coords_denormalized


    def denormalize_x(self, x):
        # TODO: also denormalize sdf here!
        # only denormalize force field, not is_boundary, sdf or normals
        force_slice = self.x_channels["force"]
        x_denormalized = x.clone()
        mask = x[:, force_slice] != 0
        x_denormalized[:, force_slice] = x[:, force_slice] * self.normalization_stats["force_std"].to(x.device) + self.normalization_stats["force_mean"].to(x.device)
        x_denormalized[:, force_slice] = torch.where(mask, x_denormalized[:, force_slice], torch.zeros_like(x_denormalized[:, force_slice]))
        return x_denormalized
    

    def denormalize_cond(self, cond):
        if self.normalization_stats["cond_min"] is None or self.normalization_stats["cond_max"] is None:
            return cond
        cond_denormalized = cond * (self.normalization_stats["cond_max"] - self.normalization_stats["cond_min"]).to(cond.device) + self.normalization_stats["cond_min"].to(cond.device)
        return cond_denormalized


    def _compute_normalization_stats(self):
        all_fields = torch.cat(self.data, dim=0)
        all_conds = torch.cat(self.cond, dim=0) # TODO: check if this works
        # min-max positions
        coords_min = all_fields[:, self._channels["coords"]].min(dim=0).values
        coords_max = all_fields[:, self._channels["coords"]].max(dim=0).values
        # z-score u, sigma, epsilon and cond
        u_mean = all_fields[:, self._channels["u"]].mean(dim=0)
        u_std = all_fields[:, self._channels["u"]].std(dim=0)
        sigma_mean = all_fields[:, self._channels["sigma"]].mean(dim=0)
        sigma_std = all_fields[:, self._channels["sigma"]].std(dim=0)
        epsilon_mean = all_fields[:, self._channels["epsilon"]].mean(dim=0)
        epsilon_std = all_fields[:, self._channels["epsilon"]].std(dim=0)
        vm_mean = all_fields[:, self._channels["vm"]].mean(dim=0)
        vm_std = all_fields[:, self._channels["vm"]].std(dim=0)
        sdf_mean = all_fields[:, self._channels["sdf"]].mean(dim=0)
        sdf_std = all_fields[:, self._channels["sdf"]].std(dim=0)
        cond_min = all_conds.min(dim=0).values
        cond_max = all_conds.max(dim=0).values
        if torch.equal(cond_min, cond_max): 
            cond_min = None
            cond_max = None
        # force field (only count nodes that are non 0 at the edge)
        force_field = all_fields[:, self._channels["force"]]
        force_mean = torch.zeros(force_field.shape[1], dtype=force_field.dtype, device=force_field.device)
        force_std = torch.zeros(force_field.shape[1], dtype=force_field.dtype, device=force_field.device)
        for i in range(force_field.shape[1]):
            mask_i = force_field[:, i] != 0
            if mask_i.any():
                force_mean[i] = force_field[:, i][mask_i].mean()
                force_std[i] = force_field[:, i][mask_i].std()
            else:
                force_mean[i] = 0.0
                force_std[i] = 1.0
            # force_mean[i] = 0.0
            # force_std[i] = 1.0

        self.normalization_stats = {
            "coords_min": coords_min,
            "coords_max": coords_max,
            "u_mean": u_mean,
            "u_std": u_std,
            "sigma_mean": sigma_mean,
            "sigma_std": sigma_std,
            "vm_mean": vm_mean,
            "vm_std": vm_std,
            "epsilon_mean": epsilon_mean,
            "epsilon_std": epsilon_std,
            "sdf_mean": sdf_mean,
            "sdf_std": sdf_std,
            "force_mean": force_mean,
            "force_std": force_std,
            "cond_min": cond_min,
            "cond_max": cond_max,
        }

    def __getitem__(self, idx):
        base_sample = super().__getitem__(idx)

        compliance = self.compliance[idx]
        strain_energy = self.strain_energy[idx]
        return PlateSample(coords=base_sample.coords,
                           connectivity=base_sample.connectivity,
                           x=base_sample.x,
                           y=base_sample.y,
                           cond=torch.zeros_like(base_sample.cond) if self.no_conditioning else base_sample.cond,
                           metadata=base_sample.metadata,
                           compliance=compliance,
                           strain_energy=strain_energy,
                           )

    @staticmethod
    def collate(batch):
        base_samples = BaseMFDataset.collate(batch)
        compliance = torch.stack([s.compliance for s in batch], dim=0).unsqueeze(-1) # (B, 1)
        strain_energy = torch.stack([s.strain_energy for s in batch], dim=0).unsqueeze(-1) # (B, 1)
        return PlateSample(coords=base_samples.coords, x=base_samples.x, y=base_samples.y, cond=base_samples.cond, batch_index=base_samples.batch_index, connectivity=base_samples.connectivity, connectivity_batch_index=base_samples.connectivity_batch_index, compliance=compliance, strain_energy=strain_energy, pad_mask=base_samples.pad_mask)
