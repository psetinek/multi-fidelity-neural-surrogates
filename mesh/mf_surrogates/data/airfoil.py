import json
import os.path as osp
from dataclasses import dataclass
from typing import Optional

import torch

from .base import BaseMFDataset, BaseSample
from .registry import register_dataset


@dataclass(kw_only=True)
class AirfoilSample(BaseSample):
    # geometry/reference data for coefficient processing.
    aero_line_connectivity: Optional[torch.Tensor] = None
    aero_line_length: Optional[torch.Tensor] = None
    aero_line_normals: Optional[torch.Tensor] = None
    aero_to_internal: Optional[torch.Tensor] = None
    cd_openfoam: Optional[torch.Tensor] = None
    cl_openfoam: Optional[torch.Tensor] = None

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        if self.aero_line_connectivity is not None:
            self.aero_line_connectivity = self.aero_line_connectivity.to(device, **kwargs)
        if self.aero_line_length is not None:
            self.aero_line_length = self.aero_line_length.to(device, **kwargs)
        if self.aero_line_normals is not None:
            self.aero_line_normals = self.aero_line_normals.to(device, **kwargs)
        if self.aero_to_internal is not None:
            self.aero_to_internal = self.aero_to_internal.to(device, **kwargs)
        if self.cd_openfoam is not None:
            self.cd_openfoam = self.cd_openfoam.to(device, **kwargs)
        if self.cl_openfoam is not None:
            self.cl_openfoam = self.cl_openfoam.to(device, **kwargs)
        return self


@register_dataset()
class AirfoilData(BaseMFDataset):
    @property
    def x_channel_names(self):
        return ["u_in", "sdf", "normals", "on_surface"]

    @property
    def y_channel_names(self):
        return ["u", "p", "wss"]

    def _init_data_lists(self):
        super()._init_data_lists()
        self.aero_line_connectivity = []
        self.aero_line_length = []
        self.aero_line_normals = []
        self.aero_to_internal = []
        self.cd_openfoam = []
        self.cl_openfoam = []

    def _load_and_append_sample(self, sample_id, fidelity_str):
        # base fields + aero geometry/reference for Cd/Cl
        sample_path = osp.join(self.data_path, sample_id, f"{fidelity_str}.h5")

        def read_sample(h5f):
            out = {
                "channels": {k: torch.tensor(v[:], dtype=torch.long) for k, v in h5f["channels"].items()},
                "cond": torch.tensor(h5f["cond"]["fidelity_raw"][()], dtype=self.dtype).unsqueeze(0),
                "data": torch.tensor(h5f["data"]["fields"][:], dtype=self.dtype),
                "tri": torch.tensor(h5f["mesh"]["connectivity"][:], dtype=torch.long),
                "metadata": json.loads(h5f.attrs["metadata_json"]),
            }
            mesh_group = h5f["mesh"]
            aero_keys = ("aero_line_connectivity", "aero_line_length", "aero_line_normals", "aero_to_internal")
            if all(k in mesh_group for k in aero_keys):
                out["aero_line_connectivity"] = torch.tensor(mesh_group["aero_line_connectivity"][:], dtype=torch.long)
                out["aero_line_length"] = torch.tensor(mesh_group["aero_line_length"][:], dtype=self.dtype)
                out["aero_line_normals"] = torch.tensor(mesh_group["aero_line_normals"][:], dtype=self.dtype)
                out["aero_to_internal"] = torch.tensor(mesh_group["aero_to_internal"][:], dtype=torch.long)
            else:
                out.update({k: None for k in aero_keys})
            if "reference" in h5f and "cd_openfoam" in h5f["reference"] and "cl_openfoam" in h5f["reference"]:
                out["cd_openfoam"] = torch.tensor(h5f["reference"]["cd_openfoam"][()], dtype=self.dtype)
                out["cl_openfoam"] = torch.tensor(h5f["reference"]["cl_openfoam"][()], dtype=self.dtype)
            else:
                out["cd_openfoam"], out["cl_openfoam"] = None, None
            return out

        r = self._read_h5_with_retry(sample_path, read_sample)

        # append only after the whole read succeeded, so a retried attempt cannot double-append
        self._channels = r["channels"]
        self.cond.append(r["cond"])
        self.data.append(r["data"])
        self.tri.append(r["tri"])
        self.metadata.append(r["metadata"])
        self.aero_line_connectivity.append(r["aero_line_connectivity"])
        self.aero_line_length.append(r["aero_line_length"])
        self.aero_line_normals.append(r["aero_line_normals"])
        self.aero_to_internal.append(r["aero_to_internal"])
        self.cd_openfoam.append(r["cd_openfoam"])
        self.cl_openfoam.append(r["cl_openfoam"])

    def normalize(self, normalization_stats=None):
        if self.normalized:
            return

        if normalization_stats is None:
            self._compute_normalization_stats()
        else:
            self.normalization_stats = normalization_stats

        stats = self.normalization_stats

        for i, sample in enumerate(self.data):
            # minmax coords [-1, 1]
            sample[:, self._channels["coords"]] = (
                2
                * (sample[:, self._channels["coords"]] - stats["coords_min"])
                / (stats["coords_max"] - stats["coords_min"])
                - 1
            )

            # zscore u, p, wss, sdf
            sample[:, self._channels["u"]] = (
                sample[:, self._channels["u"]] - stats["u_mean"]
            ) / stats["u_std"]
            sample[:, self._channels["p"]] = (
                sample[:, self._channels["p"]] - stats["p_mean"]
            ) / stats["p_std"]
            sample[:, self._channels["wss"]] = (
                sample[:, self._channels["wss"]] - stats["wss_mean"]
            ) / stats["wss_std"]
            sample[:, self._channels["sdf"]] = (
                sample[:, self._channels["sdf"]] - stats["sdf_mean"]
            ) / stats["sdf_std"]

            u_in_slice = self._channels["u_in"]
            mask = sample[:, u_in_slice] != 0.0
            sample[:, u_in_slice] = (sample[:, u_in_slice] - stats["u_in_mean"]) / stats[
                "u_in_std"
            ]
            sample[:, u_in_slice] = torch.where(
                mask,
                sample[:, u_in_slice],
                torch.zeros_like(sample[:, u_in_slice]),
            )

            if stats["cond_min"] is not None and stats["cond_max"] is not None:
                self.cond[i] = (self.cond[i] - stats["cond_min"]) / (
                    stats["cond_max"] - stats["cond_min"]
                )

        self.normalized = True

    def denormalize(self, fields, coords=None):
        stats = self.normalization_stats
        fields_denormalized = []
        for field_name, field_slice in self.y_channels.items():
            fields_denormalized.append(
                fields[:, field_slice] * stats[f"{field_name}_std"].to(fields.device)
                + stats[f"{field_name}_mean"].to(fields.device)
            )
        fields_denormalized = torch.cat(fields_denormalized, dim=-1)

        if coords is not None:
            return fields_denormalized, self.denormalize_coords(coords)
        return fields_denormalized

    def denormalize_coords(self, coords):
        coords_min = self.normalization_stats["coords_min"].to(coords.device)
        coords_max = self.normalization_stats["coords_max"].to(coords.device)
        return (coords + 1) * (coords_max - coords_min) / 2 + coords_min

    def denormalize_x(self, x):
        # only u_in is normalized among x channels, keep sdf/normals/on_surface unchanged.
        stats = self.normalization_stats
        x_denormalized = x.clone()
        u_in_slice = self.x_channels["u_in"]
        mask = x[:, u_in_slice] != 0.0
        x_denormalized[:, u_in_slice] = (
            x[:, u_in_slice] * stats["u_in_std"].to(x.device)
            + stats["u_in_mean"].to(x.device)
        )
        x_denormalized[:, u_in_slice] = torch.where(
            mask,
            x_denormalized[:, u_in_slice],
            torch.zeros_like(x_denormalized[:, u_in_slice]),
        )
        return x_denormalized

    def denormalize_cond(self, cond):
        stats = self.normalization_stats
        if stats["cond_min"] is None or stats["cond_max"] is None:
            return cond
        return (
            cond * (stats["cond_max"] - stats["cond_min"]).to(cond.device)
            + stats["cond_min"].to(cond.device)
        )

    def _compute_normalization_stats(self):
        all_fields = torch.cat(self.data, dim=0)

        # minmax coords
        coords_min = all_fields[:, self._channels["coords"]].min(dim=0).values
        coords_max = all_fields[:, self._channels["coords"]].max(dim=0).values

        # zscore u, p, wss, sdf
        u_mean = all_fields[:, self._channels["u"]].mean(dim=0)
        u_std = all_fields[:, self._channels["u"]].std(dim=0).clamp_min(1e-12)
        p_mean = all_fields[:, self._channels["p"]].mean(dim=0)
        p_std = all_fields[:, self._channels["p"]].std(dim=0).clamp_min(1e-12)
        wss_mean = all_fields[:, self._channels["wss"]].mean(dim=0)
        wss_std = all_fields[:, self._channels["wss"]].std(dim=0).clamp_min(1e-12)
        sdf_mean = all_fields[:, self._channels["sdf"]].mean(dim=0)
        sdf_std = all_fields[:, self._channels["sdf"]].std(dim=0).clamp_min(1e-12)

        u_in = all_fields[:, self._channels["u_in"]]
        u_in_mean = torch.zeros(u_in.shape[1], dtype=u_in.dtype, device=u_in.device)
        u_in_std = torch.ones(u_in.shape[1], dtype=u_in.dtype, device=u_in.device)
        for i in range(u_in.shape[1]):
            mask_i = u_in[:, i] != 0.0
            if mask_i.any():
                u_in_mean[i] = u_in[:, i][mask_i].mean()
                u_in_std[i] = u_in[:, i][mask_i].std().clamp_min(1e-12)

        # dont normalize fidelity conditioning, since its already [0, 1]
        cond_min = None
        cond_max = None

        self.normalization_stats = {
            "coords_min": coords_min,
            "coords_max": coords_max,
            "u_mean": u_mean,
            "u_std": u_std,
            "p_mean": p_mean,
            "p_std": p_std,
            "wss_mean": wss_mean,
            "wss_std": wss_std,
            "sdf_mean": sdf_mean,
            "sdf_std": sdf_std,
            "u_in_mean": u_in_mean,
            "u_in_std": u_in_std,
            "cond_min": cond_min,
            "cond_max": cond_max,
        }

    def __getitem__(self, idx):
        base_sample = super().__getitem__(idx)

        return AirfoilSample(
            coords=base_sample.coords,
            connectivity=base_sample.connectivity,
            x=base_sample.x,
            y=base_sample.y,
            cond=base_sample.cond,
            metadata=base_sample.metadata,
            aero_line_connectivity=self.aero_line_connectivity[idx],
            aero_line_length=self.aero_line_length[idx],
            aero_line_normals=self.aero_line_normals[idx],
            aero_to_internal=self.aero_to_internal[idx],
            cd_openfoam=self.cd_openfoam[idx],
            cl_openfoam=self.cl_openfoam[idx],
        )

    @staticmethod
    def collate(batch):
        # keep batch payload minimal for training, aero integration geometry is per-sample only.
        base_samples = BaseMFDataset.collate(batch)
        return AirfoilSample(
            coords=base_samples.coords,
            x=base_samples.x,
            y=base_samples.y,
            cond=base_samples.cond,
            batch_index=base_samples.batch_index,
            connectivity=base_samples.connectivity,
            connectivity_batch_index=base_samples.connectivity_batch_index,
            pad_mask=base_samples.pad_mask,
        )
