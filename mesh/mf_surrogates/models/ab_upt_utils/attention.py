"""Attention modules for the AB-UPT implementation (see mf_surrogates/models/ab_upt.py)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .layers import apply_rope, init_trunc_normal_zero_bias


@dataclass(frozen=True)
class TokenGroup:
    name: str
    start: int
    end: int
    num_anchors: int | None = None

    @property
    def size(self) -> int:
        return self.end - self.start

    @property
    def anchor_size(self) -> int:
        if self.num_anchors is None:
            return self.size
        return min(self.size, int(self.num_anchors))


def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, dim = x.shape
    head_dim = dim // num_heads
    return x.view(batch, tokens, num_heads, head_dim).transpose(1, 2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    batch, num_heads, tokens, head_dim = x.shape
    return x.transpose(1, 2).contiguous().view(batch, tokens, num_heads * head_dim)


def _sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_mask: torch.Tensor | None,
    dropout_prob: float,
    training: bool,
) -> torch.Tensor:
    attn_mask = None
    if key_mask is not None:
        attn_mask = key_mask[:, None, None, :]
    # TODO: check if it uses flash attention
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_prob if training else 0.0)


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_prob: float = 0.0, use_rope: bool = True) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout_prob = float(dropout_prob)
        self.use_rope = bool(use_rope)

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.apply(init_trunc_normal_zero_bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (_split_heads(part, self.num_heads) for part in qkv)
        if self.use_rope:
            if freqs is None:
                raise ValueError("RoPE frequencies are required for SelfAttention.")
            q = apply_rope(q, freqs)
            k = apply_rope(k, freqs)

        out = _sdpa(q, k, v, key_mask=mask, dropout_prob=self.dropout_prob, training=self.training)
        out = self.proj(_merge_heads(out))
        if mask is not None:
            out = out * mask.unsqueeze(-1).to(out.dtype)
        return self.dropout(out)


class PerceiverAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout_prob: float = 0.0,
        use_rope: bool = True,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when provided.")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.dropout_prob = float(dropout_prob)
        self.use_rope = bool(use_rope)
        self.query_chunk_size = query_chunk_size

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.apply(init_trunc_normal_zero_bias)

    def forward(
        self,
        q_tokens: torch.Tensor,
        kv_tokens: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        kv_mask: torch.Tensor | None = None,
        q_freqs: torch.Tensor | None = None,
        kv_freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = _split_heads(self.q(q_tokens), self.num_heads)
        k = _split_heads(self.k(kv_tokens), self.num_heads)
        v = _split_heads(self.v(kv_tokens), self.num_heads)

        if self.use_rope:
            if q_freqs is None or kv_freqs is None:
                raise ValueError("RoPE frequencies are required for PerceiverAttention.")
            q = apply_rope(q, q_freqs)
            k = apply_rope(k, kv_freqs)

        if self.query_chunk_size is not None and q.shape[2] > self.query_chunk_size:
            chunks = []
            for start in range(0, q.shape[2], self.query_chunk_size):
                stop = min(start + self.query_chunk_size, q.shape[2])
                chunks.append(
                    _sdpa(
                        q[:, :, start:stop],
                        k,
                        v,
                        key_mask=kv_mask,
                        dropout_prob=self.dropout_prob,
                        training=self.training,
                    )
                )
            out = torch.cat(chunks, dim=2)
        else:
            out = _sdpa(q, k, v, key_mask=kv_mask, dropout_prob=self.dropout_prob, training=self.training)
        out = self.proj(_merge_heads(out))
        if q_mask is not None:
            out = out * q_mask.unsqueeze(-1).to(out.dtype)
        return self.dropout(out)


class AnchorAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        branches: Sequence[str],
        mode: str,
        dropout_prob: float = 0.0,
        use_rope: bool = True,
        query_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive when provided.")
        if mode not in {"self", "cross", "joint"}:
            raise ValueError(f"Unsupported anchor-attention mode: {mode}")
        if mode in {"cross", "joint"} and len(branches) < 2:
            raise ValueError(f"{mode} anchor attention requires at least two branches.")

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.branches = tuple(branches)
        self.mode = mode
        self.dropout_prob = float(dropout_prob)
        self.use_rope = bool(use_rope)
        self.query_chunk_size = query_chunk_size

        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.apply(init_trunc_normal_zero_bias)

    def _patterns(self) -> list[tuple[list[str], list[str]]]:
        if self.mode == "joint":
            return [(list(self.branches), list(self.branches))]

        patterns = []
        for branch in self.branches:
            if self.mode == "self":
                patterns.append(([branch], [branch]))
            else:
                patterns.append(([branch], [other for other in self.branches if other != branch]))
        return patterns

    @staticmethod
    def _cat_groups(
        tensor: torch.Tensor,
        groups: Mapping[str, TokenGroup],
        names: Sequence[str],
        anchors_only: bool = False,
    ) -> torch.Tensor:
        parts = []
        for name in names:
            group = groups[name]
            end = group.start + group.anchor_size if anchors_only else group.end
            parts.append(tensor[:, :, group.start:end])
        return torch.cat(parts, dim=2)

    @staticmethod
    def _cat_token_groups(
        tensor: torch.Tensor,
        groups: Mapping[str, TokenGroup],
        names: Sequence[str],
        anchors_only: bool = False,
    ) -> torch.Tensor:
        parts = []
        for name in names:
            group = groups[name]
            end = group.start + group.anchor_size if anchors_only else group.end
            parts.append(tensor[:, group.start:end])
        return torch.cat(parts, dim=1)

    @staticmethod
    def _cat_masks(
        masks: Mapping[str, torch.Tensor],
        groups: Mapping[str, TokenGroup],
        names: Sequence[str],
        anchors_only: bool = False,
    ) -> torch.Tensor:
        parts = []
        for name in names:
            mask = masks[name]
            parts.append(mask[:, : groups[name].anchor_size] if anchors_only else mask)
        return torch.cat(parts, dim=1)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_mask: torch.Tensor) -> torch.Tensor:
        if self.query_chunk_size is None or q.shape[2] <= self.query_chunk_size:
            return _sdpa(q, k, v, key_mask=kv_mask, dropout_prob=self.dropout_prob, training=self.training)

        out = q.new_empty(q.shape)
        for start in range(0, q.shape[2], self.query_chunk_size):
            stop = min(start + self.query_chunk_size, q.shape[2])
            out[:, :, start:stop] = _sdpa(
                q[:, :, start:stop],
                k,
                v,
                key_mask=kv_mask,
                dropout_prob=self.dropout_prob,
                training=self.training,
            )
        return out

    def forward(
        self,
        x: torch.Tensor,
        groups: Mapping[str, TokenGroup],
        masks: Mapping[str, torch.Tensor],
        freqs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = _split_heads(self.q(x), self.num_heads)
        if self.use_rope:
            if freqs is None:
                raise ValueError("RoPE frequencies are required for AnchorAttention.")
            q = apply_rope(q, freqs)

        outputs: dict[str, torch.Tensor] = {}
        for query_names, kv_names in self._patterns():
            q_part = self._cat_groups(q, groups, query_names)
            kv_x = self._cat_token_groups(x, groups, kv_names, anchors_only=True)
            k_part = _split_heads(self.k(kv_x), self.num_heads)
            v_part = _split_heads(self.v(kv_x), self.num_heads)
            if self.use_rope:
                kv_freqs = self._cat_token_groups(freqs, groups, kv_names, anchors_only=True)
                k_part = apply_rope(k_part, kv_freqs)
            kv_mask = self._cat_masks(masks, groups, kv_names, anchors_only=True)
            out_part = self._attend(q_part, k_part, v_part, kv_mask)
            sizes = [groups[name].size for name in query_names]
            for name, chunk in zip(query_names, out_part.split(sizes, dim=2), strict=True):
                outputs[name] = chunk

        out = torch.empty_like(q)
        for name, group in groups.items():
            out[:, :, group.start : group.end] = outputs[name]
        out = self.proj(_merge_heads(out))
        full_mask = torch.cat([masks[name] for name in groups.keys()], dim=1)
        out = out * full_mask.unsqueeze(-1).to(out.dtype)
        return self.dropout(out)
