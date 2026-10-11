"""Dimension-agnostic FiLM-conditioned neural architectures.

- `ConditioningNetwork` maps a scalar fidelity signal to a conditioning
  vector. The signal can carry anything the caller wants (V-cycle count,
  log-resolution, discretization error proxy, ...); only the scale
  `t_scale` is required so the sinusoidal embedding is calibrated.
- `FiLMUNet` and `FiLMFNO` are drop-in models with `model(x, t)` call
  signature. They run in 1D, 2D, or 3D depending on `num_spatial_dims`.
"""

from architectures.conditioning import ConditioningNetwork, SinusoidalEmbedding
from architectures.fno import FiLMFNO
from architectures.unet import FiLMUNet

__all__ = [
    "ConditioningNetwork",
    "FiLMFNO",
    "FiLMUNet",
    "SinusoidalEmbedding",
]
