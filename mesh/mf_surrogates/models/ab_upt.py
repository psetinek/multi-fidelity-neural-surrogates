"""Anchored-Branched Universal Physics Transformer (AB-UPT).

The architecture and several design choices follow Emmi AI's Noether reference
implementation (https://github.com/Emmi-AI/noether).

Adapted from https://arxiv.org/abs/2502.09692
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from .ab_upt_utils import (
    AnchorTransformerBlock,
    ContinuousSincosEmbed,
    MLP,
    PerceiverBlock,
    RopeFrequency,
    SelfTransformerBlock,
    SupernodePooling,
)
from .ab_upt_utils.attention import TokenGroup
from .ab_upt_utils.layers import init_trunc_normal_zero_bias
from .utils.registry import register_model


@dataclass
class DenseBatch:
    x: torch.Tensor
    coords: torch.Tensor
    mask: torch.Tensor
    flat_indices: torch.Tensor
    num_nodes: int


@dataclass
class DomainBatch:
    positions: torch.Tensor
    mask: torch.Tensor
    flat_indices: torch.Tensor
    anchor_size: int
    features: torch.Tensor | None = None


@register_model(name="ABUPT")
class ABUPT(nn.Module):
    """Anchored-Branched UPT adapted to flattened multi-fidelity batches."""

    domain_names = ("surface", "volume")

    def __init__(
        self,
        input_channels: int = 6,
        output_channels: int = 1,
        space: int = 2,
        n_cond: int | None = 1,
        hidden_dim: int = 128,
        num_heads: int = 4,
        geometry_depth: int = 1,
        physics_blocks: list[str] | tuple[str, ...] = ("perceiver", "self", "cross", "self", "cross", "self"),
        num_domain_decoder_blocks: dict[str, int] | None = None,
        num_domain_anchor_points: dict[str, int] | None = None,
        anchor_sampling: str = "evenly_spaced",
        condition_dim: int = 128,
        fidelity_embed_dim: int = 64,
        feature_embed_dim: int = 64,
        feature_injection_mode: str = "tokens",
        token_feature_embed_dim: int | None = None,
        inject_token_features_into_geometry: bool = True,
        surface_indicator_index: int = 2,
        surface_indicator_threshold: float = 0.5,
        num_supernodes: int = 256,
        supernode_k: int = 32,
        supernode_sampling: str = "evenly_spaced",
        supernode_neighborhood: str = "knn",
        supernode_radius: float | None = None,
        supernode_max_degree: int | None = None,
        geometry_source: str = "all",
        mlp_expansion_factor: int = 4,
        dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        perceiver_query_chunk_size: int | None = None,
        anchor_query_chunk_size: int | None = None,
        max_wavelength: int = 10_000,
        coordinate_scale: float = 1.0,
        cond_scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if kwargs:
            # Surface unrecognised hparams instead of silently dropping them (e.g. mistyped config keys).
            import warnings

            warnings.warn(f"ABUPT received unused hparams that will be ignored: {sorted(kwargs)}", stacklevel=2)

        if geometry_source not in {"all", "surface"}:
            raise ValueError("geometry_source must be one of: all, surface.")
        self.geometry_source = geometry_source

        # RoPE / sincos positional encodings use frequencies designed for O(1e3)-scale positions
        # (Noether normalizes positions with strategy="position", scale=1000). With coords min-max
        # normalized to [-1, 1] the encodings are near-degenerate (adjacent nodes are positionally
        # identical), which cripples fitting of high-frequency fields. Scale coords before encoding.
        self.coordinate_scale = float(coordinate_scale)
        self.cond_scale = float(cond_scale)

        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if (hidden_dim // num_heads) % 2 != 0:
            raise ValueError("ABUPT requires an even per-head dimension for RoPE.")
        if not (0 <= surface_indicator_index < input_channels):
            raise ValueError(
                f"surface_indicator_index={surface_indicator_index} is outside input_channels={input_channels}."
            )

        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.space = int(space)
        self.n_cond = None if n_cond is None else int(n_cond)
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.condition_dim = int(condition_dim)
        self.surface_indicator_index = int(surface_indicator_index)
        self.surface_indicator_threshold = float(surface_indicator_threshold)
        # "tokens": the node input features are embedded into the surface/volume tokens (and the
        # geometry encoder); "none": tokens carry positions only
        if feature_injection_mode not in {"tokens", "none"}:
            raise ValueError("feature_injection_mode must be one of: tokens, none.")
        self.feature_injection_mode = feature_injection_mode
        self.use_token_features = feature_injection_mode == "tokens"
        self.inject_token_features_into_geometry = bool(inject_token_features_into_geometry)

        decoder_depths = {"surface": 2, "volume": 2}
        decoder_depths.update(num_domain_decoder_blocks or {})
        self.num_domain_decoder_blocks = decoder_depths
        self.num_domain_anchor_points = self._parse_domain_anchor_points(num_domain_anchor_points)
        if anchor_sampling not in {"evenly_spaced", "random"}:
            raise ValueError("anchor_sampling must be one of: evenly_spaced, random.")
        self.anchor_sampling = anchor_sampling

        self.token_feature_dim = self.input_channels if self.use_token_features else 0
        self.token_feature_embed_dim = int(token_feature_embed_dim or feature_embed_dim)

        self.pos_embed = ContinuousSincosEmbed(hidden_dim, self.space, max_wavelength=max_wavelength)
        self.rope = RopeFrequency(hidden_dim // num_heads, self.space, max_wavelength=max_wavelength)
        self.supernode_pooling = SupernodePooling(
            hidden_dim=hidden_dim,
            input_dim=self.space,
            num_supernodes=num_supernodes,
            k=supernode_k,
            dropout_prob=dropout_prob,
            input_features_dim=self.token_feature_dim
            if self.use_token_features and self.inject_token_features_into_geometry and self.token_feature_dim > 0
            else None,
            feature_embed_dim=self.token_feature_embed_dim,
            sampling=supernode_sampling,
            neighborhood=supernode_neighborhood,
            radius=supernode_radius,
            max_degree=supernode_max_degree,
        )

        if self.use_token_features and self.token_feature_dim > 0:
            self.token_feature_encoder: nn.Module | None = MLP(
                [self.token_feature_dim, self.token_feature_embed_dim, self.token_feature_embed_dim],
                dropout_prob=dropout_prob,
            )
            domain_bias_input_dim = hidden_dim + self.token_feature_embed_dim
        else:
            self.token_feature_encoder = None
            domain_bias_input_dim = hidden_dim

        self.domain_biases = nn.ModuleDict(
            {
                name: MLP([domain_bias_input_dim, hidden_dim, hidden_dim], dropout_prob=dropout_prob)
                for name in self.domain_names
            }
        )

        self.geometry_blocks = nn.ModuleList(
            [
                SelfTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_expansion_factor=mlp_expansion_factor,
                    dropout_prob=dropout_prob,
                    drop_path_rate=drop_path_rate,
                    condition_dim=condition_dim,
                    use_rope=True,
                )
                for _ in range(int(geometry_depth))
            ]
        )

        physics_modules: list[nn.Module] = []
        for block in physics_blocks:
            block = str(block)
            if block == "perceiver":
                physics_modules.append(
                    PerceiverBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        mlp_expansion_factor=mlp_expansion_factor,
                        dropout_prob=dropout_prob,
                        drop_path_rate=drop_path_rate,
                        condition_dim=condition_dim,
                        use_rope=True,
                        query_chunk_size=perceiver_query_chunk_size,
                    )
                )
            elif block in {"self", "cross", "joint"}:
                physics_modules.append(
                    AnchorTransformerBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        branches=self.domain_names,
                        mode=block,
                        mlp_expansion_factor=mlp_expansion_factor,
                        dropout_prob=dropout_prob,
                        drop_path_rate=drop_path_rate,
                        condition_dim=condition_dim,
                        use_rope=True,
                        query_chunk_size=anchor_query_chunk_size,
                    )
                )
            else:
                raise ValueError(f"Unsupported ABUPT physics block '{block}'.")
        self.physics_blocks = nn.ModuleList(physics_modules)

        self.domain_decoder_blocks = nn.ModuleDict()
        self.domain_decoder_projections = nn.ModuleDict()
        for name in self.domain_names:
            self.domain_decoder_blocks[name] = nn.ModuleList(
                [
                    AnchorTransformerBlock(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        branches=(name,),
                        mode="self",
                        mlp_expansion_factor=mlp_expansion_factor,
                        dropout_prob=dropout_prob,
                        drop_path_rate=drop_path_rate,
                        condition_dim=condition_dim,
                        use_rope=True,
                        query_chunk_size=anchor_query_chunk_size,
                    )
                    for _ in range(int(self.num_domain_decoder_blocks[name]))
                ]
            )
            projection = nn.Linear(hidden_dim, output_channels)
            init_trunc_normal_zero_bias(projection)
            self.domain_decoder_projections[name] = projection

        fidelity_ndim = self.n_cond if self.n_cond is not None and self.n_cond > 0 else 1
        self.fidelity_embed = ContinuousSincosEmbed(fidelity_embed_dim, fidelity_ndim, max_wavelength=max_wavelength)
        self.fidelity_encoder = MLP(
            [fidelity_embed_dim, fidelity_embed_dim, fidelity_embed_dim],
            dropout_prob=dropout_prob,
        )

        self.condition_fuser = MLP(
            [fidelity_embed_dim, max(condition_dim, fidelity_embed_dim), condition_dim],
            dropout_prob=dropout_prob,
        )

    @classmethod
    def _parse_domain_anchor_points(cls, values: dict[str, int] | None) -> dict[str, int | None]:
        parsed: dict[str, int | None] = {name: None for name in cls.domain_names}
        for name, value in (values or {}).items():
            if name not in parsed:
                raise ValueError(f"Unknown ABUPT domain '{name}' in num_domain_anchor_points.")
            value = int(value)
            if value <= 0:
                raise ValueError("num_domain_anchor_points values must be positive.")
            parsed[name] = value
        return parsed

    @staticmethod
    def _to_dense(
        x: torch.Tensor,
        mesh_coords: torch.Tensor,
        batch_index: torch.Tensor | None,
    ) -> DenseBatch:
        # flattened nodes of a batch of graphs (N, C) + batch_index -> padded (B, N_max, C)
        if mesh_coords.ndim != 2 or x.ndim != 2:
            raise ValueError("ABUPT expects flattened inputs of shape (N, C) with a batch_index.")
        if batch_index is None:
            batch_index = torch.zeros(mesh_coords.shape[0], dtype=torch.long, device=mesh_coords.device)
        batch_index = batch_index.to(device=mesh_coords.device, dtype=torch.long)

        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        if batch_size == 0:
            raise ValueError("ABUPT received an empty batch.")
        counts = torch.bincount(batch_index, minlength=batch_size)
        max_nodes = int(counts.max().item())

        x_dense = x.new_zeros(batch_size, max_nodes, x.shape[-1])
        coords_dense = mesh_coords.new_zeros(batch_size, max_nodes, mesh_coords.shape[-1])
        mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=mesh_coords.device)
        flat_indices = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=mesh_coords.device)

        for b in range(batch_size):
            idx = torch.nonzero(batch_index == b, as_tuple=False).flatten()
            n = idx.numel()
            if n == 0:
                raise ValueError(f"ABUPT received an empty graph at batch index {b}.")
            x_dense[b, :n] = x[idx]
            coords_dense[b, :n] = mesh_coords[idx]
            mask[b, :n] = True
            flat_indices[b, :n] = idx

        return DenseBatch(
            x=x_dense,
            coords=coords_dense,
            mask=mask,
            flat_indices=flat_indices,
            num_nodes=mesh_coords.shape[0],
        )

    @staticmethod
    def _compact_subset(
        coords: torch.Tensor,
        subset_mask: torch.Tensor,
        features: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Pack the True entries of ``subset_mask`` into a compact dense batch (B, M_max, ...).

        Lets the geometry encoder run supernode pooling over just the surface cloud (~1e3 pts) instead of
        masking the full mesh (~1e5 pts) — same supernodes, but the pooling cdist shrinks accordingly.
        """
        counts = subset_mask.sum(dim=1)
        m_max = int(counts.max().item())
        batch_size, _, dim = coords.shape
        out_coords = coords.new_zeros(batch_size, m_max, dim)
        out_mask = torch.zeros(batch_size, m_max, dtype=torch.bool, device=coords.device)
        out_features = None if features is None else features.new_zeros(batch_size, m_max, features.shape[-1])
        for b in range(batch_size):
            idx = torch.nonzero(subset_mask[b], as_tuple=False).flatten()
            n = idx.numel()
            out_coords[b, :n] = coords[b, idx]
            out_mask[b, :n] = True
            if out_features is not None:
                out_features[b, :n] = features[b, idx]
        return out_coords, out_mask, out_features

    def _anchor_first_indices(self, idx: torch.Tensor, max_anchor_points: int | None) -> tuple[torch.Tensor, int]:
        n = idx.numel()
        if max_anchor_points is None or max_anchor_points >= n:
            return idx, n

        anchor_count = max(1, int(max_anchor_points))
        if self.anchor_sampling == "random" and self.training:
            anchor_local = torch.randperm(n, device=idx.device)[:anchor_count]
        else:
            anchor_local = torch.linspace(0, n - 1, steps=anchor_count, device=idx.device).round().long()
        is_anchor = torch.zeros(n, dtype=torch.bool, device=idx.device)
        is_anchor[anchor_local] = True
        query_local = torch.arange(n, device=idx.device)[~is_anchor]
        ordered_local = torch.cat([anchor_local, query_local], dim=0)
        return idx[ordered_local], anchor_count

    def sample_training_query_indices(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample anchor-only training query indices from flattened model inputs."""

        if x.ndim != 2:
            raise ValueError("ABUPT training query sampling expects flattened x with shape (N, C).")
        if not (0 <= self.surface_indicator_index < x.shape[-1]):
            raise ValueError(
                f"surface_indicator_index={self.surface_indicator_index} is outside x channels={x.shape[-1]}."
            )

        if batch_index is None:
            batch_index = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        else:
            batch_index = batch_index.to(device=x.device, dtype=torch.long)

        batch_size = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        if batch_size == 0:
            raise ValueError("ABUPT received an empty batch.")

        surface_indicator = x[:, self.surface_indicator_index] > self.surface_indicator_threshold
        sampled = []
        for b in range(batch_size):
            graph_mask = batch_index == b
            for name, domain_mask in (
                ("surface", graph_mask & surface_indicator),
                ("volume", graph_mask & ~surface_indicator),
            ):
                idx = torch.nonzero(domain_mask, as_tuple=False).flatten()
                if idx.numel() == 0:
                    raise ValueError(f"ABUPT requires at least one {name} node per graph; missing in graph {b}.")
                ordered_idx, anchor_count = self._anchor_first_indices(idx, self.num_domain_anchor_points[name])
                sampled.append(ordered_idx[:anchor_count])

        return torch.cat(sampled, dim=0)

    def _pack_domain(
        self,
        coords: torch.Tensor,
        domain_mask: torch.Tensor,
        flat_indices: torch.Tensor,
        name: str,
        features: torch.Tensor | None = None,
        max_anchor_points: int | None = None,
    ) -> DomainBatch:
        counts = domain_mask.sum(dim=1)
        if torch.any(counts == 0):
            bad = torch.nonzero(counts == 0, as_tuple=False).flatten().tolist()
            raise ValueError(f"ABUPT requires at least one {name} node per graph; missing in graphs {bad}.")

        batch_size, _, dim = coords.shape
        max_nodes = int(counts.max().item())
        anchor_counts = counts if max_anchor_points is None else counts.clamp_max(int(max_anchor_points))
        anchor_size = int(anchor_counts.max().item())
        packed = coords.new_zeros(batch_size, max_nodes, dim)
        packed_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool, device=coords.device)
        packed_flat = torch.full((batch_size, max_nodes), -1, dtype=torch.long, device=coords.device)
        packed_features = None
        if features is not None:
            packed_features = features.new_zeros(batch_size, max_nodes, features.shape[-1])

        for b in range(batch_size):
            idx = torch.nonzero(domain_mask[b], as_tuple=False).flatten()
            idx, _ = self._anchor_first_indices(idx, max_anchor_points)
            n = idx.numel()
            packed[b, :n] = coords[b, idx]
            packed_mask[b, :n] = True
            packed_flat[b, :n] = flat_indices[b, idx]
            if packed_features is not None:
                packed_features[b, :n] = features[b, idx]
        return DomainBatch(
            positions=packed,
            mask=packed_mask,
            flat_indices=packed_flat,
            anchor_size=anchor_size,
            features=packed_features,
        )

    def _normalize_condition(self, cond: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cond_dim = self.n_cond if self.n_cond is not None and self.n_cond > 0 else 1
        if cond is None:
            return torch.zeros(batch_size, cond_dim, device=device, dtype=dtype)

        cond = cond.to(device=device, dtype=dtype)
        if cond.ndim == 0:
            cond = cond.view(1, 1).expand(batch_size, cond_dim)
        elif cond.ndim == 1:
            if batch_size == 1:
                cond = cond.view(1, -1)
            else:
                cond = cond.unsqueeze(-1)
        else:
            cond = cond.view(cond.shape[0], -1)

        if cond.shape[0] != batch_size:
            raise ValueError(f"Condition batch size {cond.shape[0]} does not match input batch size {batch_size}.")
        if cond.shape[1] != cond_dim:
            raise ValueError(f"Condition dim {cond.shape[1]} does not match expected dim {cond_dim}.")
        return cond

    def _build_condition(self, x: torch.Tensor, cond: torch.Tensor | None) -> torch.Tensor:
        fidelity = self._normalize_condition(cond, x.shape[0], device=x.device, dtype=x.dtype)
        return self.condition_fuser(self.fidelity_encoder(self.fidelity_embed(fidelity)))

    def _build_token_features(
        self,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.use_token_features or self.token_feature_dim == 0:
            return None
        if x.shape[-1] != self.token_feature_dim:
            raise ValueError(f"ABUPT expected {self.token_feature_dim} token feature channels, got {x.shape[-1]}.")
        return x * valid_mask.unsqueeze(-1).to(x.dtype)

    def _encode_domain_tokens(self, domain: DomainBatch, name: str) -> torch.Tensor:
        pos_features = self.pos_embed(domain.positions)
        if domain.features is not None:
            if self.token_feature_encoder is None:
                raise ValueError("Domain features were provided but token feature encoding is disabled.")
            feature_features = self.token_feature_encoder(domain.features)
            token_input = torch.cat([pos_features, feature_features], dim=-1)
        else:
            token_input = pos_features
        tokens = self.domain_biases[name](token_input)
        return tokens * domain.mask.unsqueeze(-1).to(tokens.dtype)

    @staticmethod
    def _build_query_mask(dense: DenseBatch, query_indices: torch.Tensor | None) -> torch.Tensor | None:
        if query_indices is None:
            return None

        query_indices = query_indices.to(device=dense.coords.device, dtype=torch.long).flatten()
        if query_indices.numel() == 0:
            raise ValueError("ABUPT received empty query_indices.")
        total_nodes = dense.num_nodes
        if int(query_indices.min().item()) < 0 or int(query_indices.max().item()) >= total_nodes:
            raise ValueError("ABUPT query_indices contain indices outside the flattened input range.")
        if torch.unique(query_indices).numel() != query_indices.numel():
            raise ValueError("ABUPT query_indices must be unique.")

        flat_mask = torch.zeros(total_nodes, dtype=torch.bool, device=dense.coords.device)
        flat_mask[query_indices] = True
        query_mask = torch.zeros_like(dense.mask)
        query_mask[dense.mask] = flat_mask[dense.flat_indices[dense.mask]]
        return query_mask

    def _scatter_outputs(
        self,
        surface_preds: torch.Tensor,
        surface: DomainBatch,
        volume_preds: torch.Tensor,
        volume: DomainBatch,
        dense: DenseBatch,
    ) -> torch.Tensor:
        out_flat = surface_preds.new_zeros(dense.num_nodes, self.output_channels)
        surface_idx = surface.flat_indices[surface.mask]
        volume_idx = volume.flat_indices[volume.mask]
        out_flat[surface_idx] = surface_preds[surface.mask]
        out_flat[volume_idx] = volume_preds[volume.mask]
        return out_flat

    def forward(
        self,
        x: torch.Tensor,
        mesh_coords: torch.Tensor,
        cond: torch.Tensor | None = None,
        batch_index: Optional[torch.Tensor] = None,
        query_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dense = self._to_dense(x=x, mesh_coords=mesh_coords, batch_index=batch_index)
        if dense.coords.shape[-1] != self.space:
            raise ValueError(f"ABUPT expected coordinate dimension {self.space}, got {dense.coords.shape[-1]}.")
        # Scale coords once so all positional encodings (pos_embed, RoPE, and the supernode pooling's
        # internal embeddings + neighbor distances) operate at the frequency-appropriate scale.
        if self.coordinate_scale != 1.0:
            dense.coords = dense.coords * self.coordinate_scale

        surface_indicator = dense.x[..., self.surface_indicator_index] > self.surface_indicator_threshold
        surface_mask = dense.mask & surface_indicator
        volume_mask = dense.mask & ~surface_indicator
        query_mask = self._build_query_mask(dense, query_indices)
        if query_mask is not None:
            surface_query_mask = surface_mask & query_mask
            volume_query_mask = volume_mask & query_mask
            domain_anchor_points = {"surface": None, "volume": None}
        else:
            surface_query_mask = surface_mask
            volume_query_mask = volume_mask
            domain_anchor_points = self.num_domain_anchor_points

        token_features = self._build_token_features(dense.x, dense.mask)

        surface = self._pack_domain(
            dense.coords,
            surface_query_mask,
            dense.flat_indices,
            "surface",
            features=token_features,
            max_anchor_points=domain_anchor_points["surface"],
        )
        volume = self._pack_domain(
            dense.coords,
            volume_query_mask,
            dense.flat_indices,
            "volume",
            features=token_features,
            max_anchor_points=domain_anchor_points["volume"],
        )

        if self.cond_scale != 1.0:
            cond = cond * self.cond_scale
        condition = self._build_condition(dense.x, cond)

        geometry_input_features = (
            token_features
            if self.use_token_features and self.inject_token_features_into_geometry and self.token_feature_dim > 0
            else None
        )
        # Faithful to the Noether aero recipe, the geometry encoder can operate on the surface (shape) cloud only.
        # For "surface" we compact the surface points first so supernode pooling doesn't pay for the full mesh.
        if self.geometry_source == "surface":
            geo_coords, geo_validity, geo_features = self._compact_subset(
                dense.coords, surface_mask, geometry_input_features
            )
        else:
            geo_coords, geo_validity, geo_features = dense.coords, dense.mask, geometry_input_features
        geometry, geometry_pos, geometry_mask = self.supernode_pooling(
            geo_coords,
            geo_validity,
            input_features=geo_features,
        )
        geometry_freqs = self.rope(geometry_pos)
        for block in self.geometry_blocks:
            geometry = block(geometry, mask=geometry_mask, freqs=geometry_freqs, condition=condition)

        surface_tokens = self._encode_domain_tokens(surface, "surface")
        volume_tokens = self._encode_domain_tokens(volume, "volume")

        x_physics = torch.cat([surface_tokens, volume_tokens], dim=1)
        physics_positions = torch.cat([surface.positions, volume.positions], dim=1)
        physics_freqs = self.rope(physics_positions)
        groups = {
            "surface": TokenGroup("surface", 0, surface.positions.shape[1], surface.anchor_size),
            "volume": TokenGroup("volume", surface.positions.shape[1], x_physics.shape[1], volume.anchor_size),
        }
        masks = {"surface": surface.mask, "volume": volume.mask}
        physics_mask = torch.cat([surface.mask, volume.mask], dim=1)

        for block in self.physics_blocks:
            if isinstance(block, PerceiverBlock):
                x_physics = block(
                    q=x_physics,
                    kv=geometry,
                    q_mask=physics_mask,
                    kv_mask=geometry_mask,
                    q_freqs=physics_freqs,
                    kv_freqs=geometry_freqs,
                    condition=condition,
                )
            elif isinstance(block, AnchorTransformerBlock):
                x_physics = block(
                    x=x_physics,
                    groups=groups,
                    masks=masks,
                    freqs=physics_freqs,
                    condition=condition,
                )
            else:
                raise TypeError(f"Unexpected ABUPT physics block type: {type(block)}.")

        surface_x = x_physics[:, groups["surface"].start : groups["surface"].end]
        volume_x = x_physics[:, groups["volume"].start : groups["volume"].end]
        surface_freqs = self.rope(surface.positions)
        volume_freqs = self.rope(volume.positions)
        surface_decoder_groups = {
            "surface": TokenGroup("surface", 0, surface_x.shape[1], surface.anchor_size),
        }
        volume_decoder_groups = {
            "volume": TokenGroup("volume", 0, volume_x.shape[1], volume.anchor_size),
        }

        for block in self.domain_decoder_blocks["surface"]:
            surface_x = block(
                surface_x,
                groups=surface_decoder_groups,
                masks={"surface": surface.mask},
                freqs=surface_freqs,
                condition=condition,
            )
        for block in self.domain_decoder_blocks["volume"]:
            volume_x = block(
                volume_x,
                groups=volume_decoder_groups,
                masks={"volume": volume.mask},
                freqs=volume_freqs,
                condition=condition,
            )

        surface_preds = self.domain_decoder_projections["surface"](surface_x)
        volume_preds = self.domain_decoder_projections["volume"](volume_x)
        surface_preds = surface_preds * surface.mask.unsqueeze(-1).to(surface_preds.dtype)
        volume_preds = volume_preds * volume.mask.unsqueeze(-1).to(volume_preds.dtype)

        out = self._scatter_outputs(surface_preds, surface, volume_preds, volume, dense)
        if query_indices is not None:
            query_indices = query_indices.to(device=out.device, dtype=torch.long).flatten()
            return out.index_select(0, query_indices)
        return out
