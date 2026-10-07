"""Layers used by the AB-UPT implementation (see mf_surrogates/models/ab_upt.py)."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def init_trunc_normal_zero_bias(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(
        self,
        dims: Sequence[int],
        act_fn: type[nn.Module] = nn.GELU,
        dropout_prob: float = 0.0,
        last_act_fn: type[nn.Module] | None = None,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        if len(dims) < 2:
            raise ValueError("MLP requires at least input and output dimensions.")

        layers: list[nn.Module] = []
        for i, (din, dout) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(int(din), int(dout)))
            is_last = i == len(dims) - 2
            if not is_last:
                layers.append(act_fn())
                if dropout_prob > 0:
                    layers.append(nn.Dropout(dropout_prob))
        if last_act_fn is not None:
            layers.append(last_act_fn())
        self.net = nn.Sequential(*layers)

        if init_weights:
            self.apply(init_trunc_normal_zero_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousSincosEmbed(nn.Module):
    """Continuous sine/cosine embedding for coordinates or scalar conditions."""

    def __init__(self, dim: int, ndim: int, max_wavelength: int = 10_000, dtype=torch.float32) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("Embedding dimension must be positive.")
        if ndim <= 0:
            raise ValueError("Input dimension must be positive.")

        self.dim = int(dim)
        self.ndim = int(ndim)
        self.ndim_padding = self.dim % self.ndim
        dim_per_ndim = (self.dim - self.ndim_padding) // self.ndim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * self.ndim
        effective_dim_per_wave = (self.dim - self.padding) // self.ndim
        if effective_dim_per_wave <= 0:
            raise ValueError("Embedding dimension is too small for the requested coordinate dimension.")

        omega = 1.0 / max_wavelength ** (
            torch.arange(0, effective_dim_per_wave, 2, dtype=dtype) / effective_dim_per_wave
        )
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.ndim:
            raise ValueError(f"Expected last dimension {self.ndim}, got {coords.shape[-1]}.")
        out_dtype = coords.dtype
        freqs = coords.to(self.omega.dtype).unsqueeze(-1) @ self.omega.unsqueeze(0)
        emb = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).flatten(start_dim=-2)
        emb = emb.to(out_dtype)
        if self.padding > 0:
            pad_shape = (*emb.shape[:-1], self.padding)
            emb = torch.cat([emb, emb.new_zeros(pad_shape)], dim=-1)
        return emb


class RopeFrequency(nn.Module):
    """Complex rotary-position frequency generator."""

    def __init__(self, head_dim: int, input_dim: int, max_wavelength: int = 10_000) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("AB-UPT RoPE requires an even per-head hidden dimension.")
        if input_dim <= 0:
            raise ValueError("RoPE input dimension must be positive.")

        self.head_dim = int(head_dim)
        self.input_dim = int(input_dim)
        self.ndim_padding = self.head_dim % self.input_dim
        dim_per_ndim = (self.head_dim - self.ndim_padding) // self.input_dim
        self.sincos_padding = dim_per_ndim % 2
        self.padding = self.ndim_padding + self.sincos_padding * self.input_dim
        effective_dim_per_wave = (self.head_dim - self.padding) // self.input_dim
        if effective_dim_per_wave <= 0:
            raise ValueError("RoPE head dimension is too small for the requested coordinate dimension.")

        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        omega = 1.0 / max_wavelength ** (arange / effective_dim_per_wave)
        self.register_buffer("omega", omega, persistent=False)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.shape[-1] != self.input_dim:
            raise ValueError(f"Expected coordinate dimension {self.input_dim}, got {coords.shape[-1]}.")
        out = coords.float().unsqueeze(-1) @ self.omega.unsqueeze(0)
        out = out.flatten(start_dim=-2)
        if self.padding > 0:
            if self.padding % 2 != 0:
                raise ValueError("Complex RoPE padding must be even.")
            out = torch.cat([out, out.new_zeros(*out.shape[:-1], self.padding // 2)], dim=-1)
        return torch.polar(torch.ones_like(out), out)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to ``x`` shaped ``(B, H, N, D_head)``."""

    if x.shape[-1] % 2 != 0:
        raise ValueError("RoPE input head dimension must be even.")
    if freqs.ndim != 3:
        raise ValueError("RoPE frequencies must have shape (B, N, D_head / 2).")
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = x_complex * freqs[:, None, :, :]
    return torch.view_as_real(rotated).flatten(start_dim=-2).to(dtype=x.dtype)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor
