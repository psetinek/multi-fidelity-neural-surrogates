"""
Transolver: A Fast Transformer Solver for PDEs on General Geometries

Adapted from https://github.com/thuml/Transolver
"""

from typing import Optional
from functools import partial

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch
import torch.utils.checkpoint as checkpoint


from .utils.condition import ContinuousSincosEmbed, DiT
from .utils.utils import MLP
from .utils.registry import register_model
from .utils.initialization import deterministic_trunc_normal_


class TransolverAttention(nn.Module):
    """
    Multi-head self-attention with physics-aware slicing for PDE solvers.

    :param dim: Input feature dimension.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dropout_prob: Dropout probability after attention.
    :type dropout_prob: float
    :param attn_dropout_prob: Dropout within attention weights.
    :type attn_dropout_prob: float
    :param slice_base: Base number of slices for token grouping.
    :type slice_base: int
    """


    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.0,
        slice_base: int = 64,
    ):
        super().__init__()

        assert (dim % num_heads) == 0
        self.dim = dim
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.slice_base = slice_base
        self.attn_dropout_prob = attn_dropout_prob

        self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)
        # input projection
        self.x_proj = nn.Linear(dim, dim)
        self.fx_proj = nn.Linear(dim, dim)
        self.slice_proj = nn.Linear(self.head_dim, slice_base)
        nn.init.orthogonal_(self.slice_proj.weight)
        # qkv projection
        self.qkv = nn.Linear(self.head_dim, self.head_dim * 3, bias=False)
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout_prob))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # slices
        x_mid = rearrange(self.x_proj(x), "b n (h c) -> b h n c", c=self.head_dim)
        fx_mid = rearrange(self.fx_proj(x), "b n (h c) -> b h n c", c=self.head_dim)
        slice_weights = F.softmax(
            self.slice_proj(x_mid) / self.temperature, -1
        )  # b h n g
        # mask padded nodes
        if mask is not None:
            mask = rearrange(mask, "b n -> b 1 n 1")  # b 1 n 1
            slice_weights = slice_weights * mask.to(slice_weights.dtype)  # b h n g

        # in-slice attention
        scale = (slice_weights.sum(2) + 1e-5)[..., None].repeat(1, 1, 1, self.head_dim)
        slice_att = einsum(fx_mid, slice_weights, "b h n c, b h n g -> b h g c") / scale
        # global (across slices) attention
        qkv = rearrange(self.qkv(slice_att), "b h g (thr c) -> thr b h g c", thr=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dropout = self.attn_dropout_prob if self.training else 0.0
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        # merge (cross attention)
        x = einsum(att, slice_weights, "b h g c, b h n g -> b h n c")
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.readout(x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.0,
        act_fn: nn.Module = nn.SiLU,
        mlp_ratio: float = 4.0,
        slice_base: int = 64,
        norm_layer: nn.Module = nn.LayerNorm,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.norm1 = norm_layer(dim)
        self.attn = TransolverAttention(
            dim=dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            slice_base=slice_base,
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLP([dim, int(dim * mlp_ratio), dim], act_fn=act_fn)

    def forward_attn(self, x, mask=None):
        return self.attn(self.norm1(x), mask=mask)

    def forward_mlp(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ViT-like structure
        # attention
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_attn, x, mask, use_reentrant=False)
        else:
            x = x + self.forward_attn(x, mask)
        # mlp
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_mlp, x, use_reentrant=False)
        else:
            x = x + self.forward_mlp(x)
        return x


class DiTransolverBlock(TransolverBlock):
    def __init__(self, cond_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dit = DiT(self.dim, cond_dim)
    
    def forward_attn(self, x, scale1, shift1, gate1, mask=None):
        x1 = self.dit.modulate_scale_shift(self.norm1(x), scale1, shift1)
        x2 = self.dit.modulate_gate(self.attn(x1, mask=mask), gate1)
        return x2
    
    def forward_mlp(self, x, scale2, shift2, gate2):
        x3 = self.dit.modulate_scale_shift(self.norm2(x), scale2, shift2)
        x4 = self.dit.modulate_gate(self.mlp(x3), gate2)
        return x4

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor], cond: torch.Tensor) -> torch.Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = self.dit(cond)
        # modulated attention
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_attn, x, scale1, shift1, gate1, mask, use_reentrant=False)
        else:
            x = x + self.forward_attn(x, scale1, shift1, gate1, mask)
        # modulated mlp
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_mlp, x, scale2, shift2, gate2, use_reentrant=False)
        else:
            x = x + self.forward_mlp(x, scale2, shift2, gate2)
        return x


@register_model()
class Transolver(nn.Module):
    """
    Transolver: A Transformer-based PDE solver, with Physics-Attention to model physical
    states and efficiently capture complex geometries.

    Args:
        input_channels (int): Number of input feature channels per node.
        output_channels (int): Number of predicted channels per node.
        act_fn (nn.Module): Activation function.
        dropout_prob (float): Dropout probability in the MLPs and after attention.
        attn_dropout_prob (float): Dropout on the attention weights.
        space (int): Spatial dimensionality of the mesh coordinates.
        transolver_base (int): Latent width of the transformer.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transolver blocks.
        slice_base (int): Number of slices (physical states) per head.
        mlp_ratio (float): Hidden width of the block MLPs relative to transolver_base.
        cond_dim (int): Width of the conditioning embedding fed to the DiT modulation.
        n_cond (Optional[int]): Number of conditioning inputs (the fidelity); None disables
            conditioning.
        gradient_checkpointing (bool): Recompute block activations in the backward pass.
        coordinate_scale (float): Factor applied to the coordinates before the sin-cos embedding.
        cond_scale (float): Factor applied to the conditioning before the sin-cos embedding.

    Paper: https://arxiv.org/abs/2402.02366
    """

    def __init__(
        self,
        input_channels: int = 8,
        output_channels: int = 17,
        act_fn: nn.Module = nn.SiLU,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        space: int = 2,
        transolver_base: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        slice_base: int = 64,
        mlp_ratio: float = 2.0,
        cond_dim: int = 32,
        n_cond=None,
        gradient_checkpointing: bool = False,
        coordinate_scale: float = 1.0,
        cond_scale: float = 1.0,
    ):
        super().__init__()

        self.space = space
        self.output_channels = output_channels
        self.use_conditioning = True if n_cond is not None else False
        self.coordinate_scale = float(coordinate_scale)
        self.cond_scale = float(cond_scale)

        assert (transolver_base % num_heads) == 0

        # encode positions + input fields to latent
        self.coord_embed = ContinuousSincosEmbed(dim=transolver_base // 2, ndim=space)
        self.coord_encoder = MLP([transolver_base // 2, transolver_base // 2], act_fn=act_fn, dropout_prob=dropout_prob)
        self.feature_encoder = MLP([input_channels, transolver_base // 2], act_fn=act_fn, dropout_prob=dropout_prob)
        self.coord_feature_fuser = MLP([transolver_base, transolver_base], act_fn=act_fn, dropout_prob=dropout_prob)

        if self.use_conditioning:
            self.cond_encoder = nn.Sequential(
                ContinuousSincosEmbed(dim=128, ndim=n_cond),
                MLP(
                    [128, 128 // 2, cond_dim],
                    act_fn=act_fn,
                    dropout_prob=dropout_prob,
                ),
            )
            BlockType = partial(DiTransolverBlock, cond_dim)
        else:
            BlockType = TransolverBlock

        blocks = []
        for _ in range(num_layers):
            block = BlockType(
                dim=transolver_base,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                act_fn=act_fn,
                mlp_ratio=mlp_ratio,
                slice_base=slice_base,
                use_checkpoint=gradient_checkpointing
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        self.decoder = MLP(
            [transolver_base, transolver_base, output_channels],
            act_fn,
            dropout_prob=dropout_prob,
        )


        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)
        for block in self.blocks:
            torch.nn.init.orthogonal_(block.attn.slice_proj.weight)  # use a principled initialization
            torch.nn.init.orthogonal_(block.attn.x_proj.weight)
            torch.nn.init.orthogonal_(block.attn.fx_proj.weight)
            torch.nn.init.orthogonal_(block.attn.qkv.weight)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            deterministic_trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        x: torch.Tensor,
        mesh_coords: torch.Tensor,
        cond: torch.Tensor = None,
        batch_index: Optional[torch.Tensor] = None,
    ):
        pad_mask = None

        # pad to max nodes if in sparse format
        if mesh_coords.ndim == 2:
            mesh_coords, pad_mask = to_dense_batch(mesh_coords, batch_index)
            x, _ = to_dense_batch(x, batch_index)

        if self.coordinate_scale != 1.0:
            mesh_coords = mesh_coords * self.coordinate_scale

        # coord encoder
        coord_encoding = self.coord_encoder(self.coord_embed(mesh_coords))  # (B, N, C)

        # feature encoder
        feature_encoding = self.feature_encoder(x)  # (B, N, C)

        x = self.coord_feature_fuser(torch.cat([coord_encoding, feature_encoding], dim=-1))

        # cond encoder
        if self.use_conditioning:
            if self.cond_scale != 1.0:
                cond = cond * self.cond_scale
            cond_encoding = self.cond_encoder(cond)
            cond = {"cond": cond_encoding}
        else:
            cond = {}


        # transolver
        for block in self.blocks:
            x = block(x, mask=pad_mask, **cond)

        # decoder
        x = self.decoder(x)

        # unpad
        if pad_mask is not None:
            x = x[pad_mask]

        return x
