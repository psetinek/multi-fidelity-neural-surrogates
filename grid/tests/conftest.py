"""Shared pytest configuration.

Forces JAX onto the CPU backend and enables float64 before JAX is imported
anywhere else (`jax` must be configured before any traced computation so
x64 applies consistently across the test session).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)
