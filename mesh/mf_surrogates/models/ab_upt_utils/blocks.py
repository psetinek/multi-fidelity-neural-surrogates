"""Transformer and Perceiver blocks for the local AB-UPT implementation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn

from .attention import AnchorAttention, PerceiverAttention, SelfAttention, TokenGroup
from .layers import DropPath, MLP


class ConditionedBlockMixin:
    condition_dim: int | None
    hidden_dim: int

    def _build_modulation(self) -> nn.Module | None:
        if self.condition_dim is None:
            return None
        modulation = nn.Linear(self.condition_dim, self.hidden_dim * 6)
        nn.init.trunc_normal_(modulation.weight, std=0.02)
        nn.init.zeros_(modulation.bias)
        return modulation

    @staticmethod
    def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    @staticmethod
    def _gate(x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return x * gate.unsqueeze(1)


class SelfTransformerBlock(nn.Module, ConditionedBlockMixin):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int = 4,
        dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        condition_dim: int | None = None,
        use_rope: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = condition_dim
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = SelfAttention(hidden_dim, num_heads, dropout_prob=dropout_prob, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(
            [hidden_dim, hidden_dim * int(mlp_expansion_factor), hidden_dim],
            dropout_prob=dropout_prob,
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.modulation = self._build_modulation()
        self.apply(self._init_norm)

    @staticmethod
    def _init_norm(module: nn.Module) -> None:
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        freqs: torch.Tensor | None,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.modulation is None:
            if condition is not None:
                raise ValueError("Condition provided to an unconditioned transformer block.")
            x = x + self.drop_path1(self.attn(self.norm1(x), mask=mask, freqs=freqs))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            if condition is None:
                raise ValueError("Conditioned transformer block requires a condition tensor.")
            attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = self.modulation(condition).chunk(6, -1)
            attn_in = self._modulate(self.norm1(x), attn_scale, attn_shift)
            x = x + self.drop_path1(self._gate(self.attn(attn_in, mask=mask, freqs=freqs), attn_gate))
            mlp_in = self._modulate(self.norm2(x), mlp_scale, mlp_shift)
            x = x + self.drop_path2(self._gate(self.mlp(mlp_in), mlp_gate))
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x


class AnchorTransformerBlock(nn.Module, ConditionedBlockMixin):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        branches: Sequence[str],
        mode: str,
        mlp_expansion_factor: int = 4,
        dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        condition_dim: int | None = None,
        use_rope: bool = True,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = condition_dim
        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = AnchorAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            branches=branches,
            mode=mode,
            dropout_prob=dropout_prob,
            use_rope=use_rope,
            query_chunk_size=query_chunk_size,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(
            [hidden_dim, hidden_dim * int(mlp_expansion_factor), hidden_dim],
            dropout_prob=dropout_prob,
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.modulation = self._build_modulation()
        self.apply(SelfTransformerBlock._init_norm)

    def forward(
        self,
        x: torch.Tensor,
        groups: Mapping[str, TokenGroup],
        masks: Mapping[str, torch.Tensor],
        freqs: torch.Tensor | None,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        full_mask = torch.cat([masks[name] for name in groups.keys()], dim=1)
        if self.modulation is None:
            if condition is not None:
                raise ValueError("Condition provided to an unconditioned transformer block.")
            x = x + self.drop_path1(self.attn(self.norm1(x), groups=groups, masks=masks, freqs=freqs))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            if condition is None:
                raise ValueError("Conditioned transformer block requires a condition tensor.")
            attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = self.modulation(condition).chunk(6, -1)
            attn_in = self._modulate(self.norm1(x), attn_scale, attn_shift)
            x = x + self.drop_path1(
                self._gate(self.attn(attn_in, groups=groups, masks=masks, freqs=freqs), attn_gate)
            )
            mlp_in = self._modulate(self.norm2(x), mlp_scale, mlp_shift)
            x = x + self.drop_path2(self._gate(self.mlp(mlp_in), mlp_gate))
        return x * full_mask.unsqueeze(-1).to(x.dtype)


class PerceiverBlock(nn.Module, ConditionedBlockMixin):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_expansion_factor: int = 4,
        dropout_prob: float = 0.0,
        drop_path_rate: float = 0.0,
        condition_dim: int | None = None,
        use_rope: bool = True,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.condition_dim = condition_dim
        self.norm_q = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = PerceiverAttention(
            hidden_dim,
            num_heads,
            dropout_prob=dropout_prob,
            use_rope=use_rope,
            query_chunk_size=query_chunk_size,
        )
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.mlp = MLP(
            [hidden_dim, hidden_dim * int(mlp_expansion_factor), hidden_dim],
            dropout_prob=dropout_prob,
        )
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.modulation = self._build_modulation()
        self.apply(SelfTransformerBlock._init_norm)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_mask: torch.Tensor,
        kv_mask: torch.Tensor,
        q_freqs: torch.Tensor,
        kv_freqs: torch.Tensor,
        condition: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.modulation is None:
            if condition is not None:
                raise ValueError("Condition provided to an unconditioned perceiver block.")
            q = q + self.drop_path1(
                self.attn(
                    self.norm_q(q),
                    self.norm_kv(kv),
                    q_mask=q_mask,
                    kv_mask=kv_mask,
                    q_freqs=q_freqs,
                    kv_freqs=kv_freqs,
                )
            )
            q = q + self.drop_path2(self.mlp(self.norm2(q)))
        else:
            if condition is None:
                raise ValueError("Conditioned perceiver block requires a condition tensor.")
            attn_scale, attn_shift, attn_gate, mlp_scale, mlp_shift, mlp_gate = self.modulation(condition).chunk(6, -1)
            q_in = self._modulate(self.norm_q(q), attn_scale, attn_shift)
            q = q + self.drop_path1(
                self._gate(
                    self.attn(
                        q_in,
                        self.norm_kv(kv),
                        q_mask=q_mask,
                        kv_mask=kv_mask,
                        q_freqs=q_freqs,
                        kv_freqs=kv_freqs,
                    ),
                    attn_gate,
                )
            )
            mlp_in = self._modulate(self.norm2(q), mlp_scale, mlp_shift)
            q = q + self.drop_path2(self._gate(self.mlp(mlp_in), mlp_gate))
        return q * q_mask.unsqueeze(-1).to(q.dtype)
