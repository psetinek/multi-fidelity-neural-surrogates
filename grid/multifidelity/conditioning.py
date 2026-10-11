"""Log-transform the scalar fidelity signal before it enters the model.

Maps ξ ∈ [1, ξ_max] to log(ξ)/log(ξ_max) · ξ_max. At ξ = ξ_max this equals
ξ_max, so the model's internal sinusoidal embedding (which calibrates
against `t_scale = ξ_max`) sees the full dynamic range. The mapping matches
`reference_implementations/mg_film_unet.py:109-118`.
"""

import jax.numpy as jnp


def transform_log(xi, xi_max):
    """xi -> log(xi) / log(xi_max) * xi_max, clamped at log(1) from below."""
    log_normalized = jnp.log(jnp.maximum(xi.astype(jnp.float32), 1.0)) / jnp.log(
        float(xi_max)
    )
    return log_normalized * float(xi_max)
