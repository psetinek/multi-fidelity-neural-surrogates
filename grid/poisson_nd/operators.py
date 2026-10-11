"""(2d+1)-point Laplacian with zero Dirichlet padding.

`neighbor_sum` is shared by the Laplacian and the red-black Gauss-Seidel
smoother. Zero padding is used explicitly instead of `jnp.roll` because
`roll` would wrap around and impose periodic BCs.
"""

import jax.numpy as jnp


def neighbor_sum(u):
    """Sum of the 2d nearest-neighbor values with zero Dirichlet boundaries.

    For u of shape (N_0, ..., N_{d-1}), returns an array of the same shape
    whose entry at index i = (i_0, ..., i_{d-1}) equals the sum over axes
    of u[i with i_a -> i_a - 1] + u[i with i_a -> i_a + 1], with entries
    outside the domain treated as zero.
    """
    padded = jnp.pad(u, 1, mode="constant", constant_values=0.0)
    ndim = u.ndim
    result = jnp.zeros_like(u)
    for axis in range(ndim):
        minus_slc = tuple(
            slice(1, -1) if i != axis else slice(None, -2) for i in range(ndim)
        )
        plus_slc = tuple(
            slice(1, -1) if i != axis else slice(2, None) for i in range(ndim)
        )
        result = result + padded[minus_slc] + padded[plus_slc]
    return result


def laplacian(u, h):
    """Apply the (2d+1)-point Laplacian with zero Dirichlet BCs.

    Returns (sum_a (u_{a-1} + u_{a+1}) - 2d u) / h^2, where u_{a+-1}
    denote shifted copies along axis `a` with zero boundary values.
    """
    return (neighbor_sum(u) - 2 * u.ndim * u) / (h * h)
