"""Supernode pooling for AB-UPT geometry encoding.

Messages from nearby geometry points are aggregated into supernode tokens, following the
supernode pooling of UPT/AB-UPT (see mf_surrogates/models/ab_upt.py).
"""

from __future__ import annotations

import torch
from torch import nn

from .layers import ContinuousSincosEmbed, MLP


class SupernodePooling(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        input_dim: int,
        num_supernodes: int = 256,
        k: int = 32,
        dropout_prob: float = 0.0,
        input_features_dim: int | None = None,
        feature_embed_dim: int | None = None,
        sampling: str = "evenly_spaced",
        neighborhood: str = "knn",
        radius: float | None = None,
        max_degree: int | None = None,
    ) -> None:
        super().__init__()
        if num_supernodes <= 0:
            raise ValueError("num_supernodes must be positive.")
        if k <= 0:
            raise ValueError("supernode k must be positive.")
        if sampling not in {"evenly_spaced", "random"}:
            raise ValueError("supernode sampling must be one of: evenly_spaced, random.")
        if neighborhood not in {"knn", "radius"}:
            raise ValueError("supernode neighborhood must be one of: knn, radius.")
        if neighborhood == "radius" and radius is None:
            raise ValueError("supernode neighborhood='radius' requires a positive radius.")

        self.hidden_dim = int(hidden_dim)
        self.input_dim = int(input_dim)
        self.num_supernodes = int(num_supernodes)
        self.k = int(k)
        self.sampling = sampling
        self.neighborhood = neighborhood
        self.radius = None if radius is None else float(radius)
        # For radius graphs we still gather a fixed budget of candidates and mask by distance.
        self.max_degree = int(max_degree) if max_degree is not None else int(k)
        self.input_features_dim = input_features_dim
        self.feature_embed_dim = int(feature_embed_dim or hidden_dim)

        self.pos_embed = ContinuousSincosEmbed(hidden_dim, input_dim)
        self.rel_embed = ContinuousSincosEmbed(hidden_dim, input_dim + 1)
        if input_features_dim is not None:
            self.feature_proj: nn.Module | None = MLP(
                [input_features_dim, self.feature_embed_dim, hidden_dim],
                dropout_prob=dropout_prob,
            )
            message_input_dim = hidden_dim * 4
        else:
            self.feature_proj = None
            message_input_dim = hidden_dim * 3

        self.message = MLP([message_input_dim, hidden_dim, hidden_dim], dropout_prob=dropout_prob)
        self.out = MLP([hidden_dim * 2, hidden_dim], dropout_prob=dropout_prob)

    def _select_supernodes(self, coords: torch.Tensor, mask: torch.Tensor, num_supernodes: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, dim = coords.shape
        out = coords.new_zeros(batch, num_supernodes, dim)
        out_mask = torch.zeros(batch, num_supernodes, dtype=torch.bool, device=coords.device)
        # Random sampling only while training; eval stays deterministic (evenly spaced) for reproducible metrics.
        use_random = self.sampling == "random" and self.training
        for b in range(batch):
            valid_idx = torch.nonzero(mask[b], as_tuple=False).flatten()
            n_valid = valid_idx.numel()
            if n_valid == 0:
                raise ValueError("ABUPT received a graph with no valid geometry nodes.")
            # Clamp to available points so we never produce duplicate supernodes; pad+mask the remainder.
            n_eff = min(num_supernodes, n_valid)
            if use_random:
                sel = torch.randperm(n_valid, device=coords.device)[:n_eff]
            else:
                sel = torch.linspace(0, n_valid - 1, steps=n_eff, device=coords.device).round().long()
            out[b, :n_eff] = coords[b, valid_idx[sel]]
            out_mask[b, :n_eff] = True
        return out, out_mask

    def forward(
        self,
        coords: torch.Tensor,
        mask: torch.Tensor,
        input_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if coords.ndim != 3:
            raise ValueError("SupernodePooling expects coords with shape (B, N, D).")
        if mask.shape != coords.shape[:2]:
            raise ValueError("SupernodePooling mask must have shape (B, N).")
        if self.input_features_dim is None and input_features is not None:
            raise ValueError("SupernodePooling received input_features but was not configured for them.")
        if self.input_features_dim is not None:
            if input_features is None:
                raise ValueError("SupernodePooling requires input_features for this configuration.")
            if input_features.shape[:2] != coords.shape[:2] or input_features.shape[-1] != self.input_features_dim:
                raise ValueError("SupernodePooling input_features shape does not match the configured feature dim.")

        super_pos, super_mask = self._select_supernodes(coords, mask, self.num_supernodes)
        point_embed = self.pos_embed(coords)
        super_embed = self.pos_embed(super_pos)
        feature_embed = self.feature_proj(input_features) if self.feature_proj is not None else None

        distances = torch.cdist(super_pos.float(), coords.float(), compute_mode="donot_use_mm_for_euclid_dist")
        distances = distances.masked_fill(~mask[:, None, :], torch.finfo(distances.dtype).max)
        # Gather a fixed neighbor budget via top-k; for radius graphs, additionally invalidate neighbors
        # outside the radius (top-k of a fixed budget keeps the tensor shape static for batching).
        if self.neighborhood == "radius":
            budget = min(self.max_degree, coords.shape[1])
            nn_dist, nn_idx = torch.topk(distances, k=budget, dim=-1, largest=False)
            neighbor_extra_valid = nn_dist <= self.radius
        else:
            k_eff = min(self.k, coords.shape[1])
            _, nn_idx = torch.topk(distances, k=k_eff, dim=-1, largest=False)
            neighbor_extra_valid = None

        gather_idx = nn_idx.unsqueeze(-1).expand(-1, -1, -1, coords.shape[-1])
        neighbor_pos = torch.gather(coords[:, None].expand(-1, self.num_supernodes, -1, -1), 2, gather_idx)

        embed_idx = nn_idx.unsqueeze(-1).expand(-1, -1, -1, self.hidden_dim)
        neighbor_embed = torch.gather(point_embed[:, None].expand(-1, self.num_supernodes, -1, -1), 2, embed_idx)
        neighbor_valid = torch.gather(mask[:, None].expand(-1, self.num_supernodes, -1), 2, nn_idx)
        if neighbor_extra_valid is not None:
            neighbor_valid = neighbor_valid & neighbor_extra_valid
        if feature_embed is not None:
            neighbor_feature_embed = torch.gather(
                feature_embed[:, None].expand(-1, self.num_supernodes, -1, -1),
                2,
                embed_idx,
            )

        rel = super_pos[:, :, None, :] - neighbor_pos
        rel_mag = rel.norm(dim=-1, keepdim=True)
        rel_embed = self.rel_embed(torch.cat([rel, rel_mag], dim=-1))
        super_embed_expanded = super_embed[:, :, None, :].expand_as(neighbor_embed)
        message_parts = [neighbor_embed, super_embed_expanded, rel_embed]
        if feature_embed is not None:
            message_parts.append(neighbor_feature_embed)
        messages = self.message(torch.cat(message_parts, dim=-1))
        messages = messages * neighbor_valid.unsqueeze(-1).to(messages.dtype)
        denom = neighbor_valid.sum(dim=2, keepdim=True).clamp_min(1).to(messages.dtype)
        pooled = messages.sum(dim=2) / denom
        return self.out(torch.cat([pooled, super_embed], dim=-1)), super_pos, super_mask
