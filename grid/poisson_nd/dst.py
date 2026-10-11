"""n-D DST-I and the corresponding direct Poisson solver.

The DST-I is self-inverse up to a constant factor of 2(N+1) per axis, and
the (2d+1)-point Laplacian with zero Dirichlet BCs is diagonalized by the
DST-I with eigenvalues lam_a[k] = (2/h^2)(cos(pi k / (N_a+1)) - 1), summed
over axes.
"""

import jax.numpy as jnp


def _dst_i_axis(x, axis):
    """DST-I along one axis via antisymmetric extension and FFT.

    For a 1D signal of length N, the extension
    y = [0, x_0, ..., x_{N-1}, 0, -x_{N-1}, ..., -x_0]
    of length 2(N+1) has DST-I(x)[k] = -Im(FFT(y))[k+1]. The same trick
    generalizes axis-wise.
    """
    N = x.shape[axis]

    zero_shape = list(x.shape)
    zero_shape[axis] = 1
    z = jnp.zeros(zero_shape, dtype=x.dtype)

    x_rev = jnp.flip(x, axis=axis)
    extended = jnp.concatenate([z, x, z, -x_rev], axis=axis)

    freq = jnp.fft.fft(extended, axis=axis)
    slc = [slice(None)] * x.ndim
    slc[axis] = slice(1, N + 1)
    return -freq.imag[tuple(slc)]


def dstn_type1(x):
    """n-D DST-I: apply 1D DST-I along each axis in turn."""
    result = x
    for a in range(x.ndim):
        result = _dst_i_axis(result, a)
    return result


def dst_resample(u, target_shape):
    """Resample `u` to `target_shape` in the DST-I (Dirichlet sine) basis.

    Forward DST-I, zero-pad (upsample) or truncate (downsample) the sine
    coefficients per axis with the grid-size normalization, then inverse DST-I.
    Exact (to machine precision) for any function band-limited to the retained
    sine modes — which the Dirichlet-Poisson solution is — so it introduces no
    interpolation error, unlike the linear `multigrid.prolongate`.

    Isotropic or anisotropic `target_shape` both handled. Interior-grid
    convention (length-N axis ↔ modes 1..N), matching `solve_direct`.
    """
    target_shape = tuple(target_shape)
    d = u.ndim
    coeffs = dstn_type1(u)

    resized = jnp.zeros(target_shape, dtype=u.dtype)
    low = tuple(slice(0, min(u.shape[a], target_shape[a])) for a in range(d))
    resized = resized.at[low].set(coeffs[low])

    scale = 1.0
    for a in range(d):
        scale *= (target_shape[a] + 1) / (u.shape[a] + 1)
    norm = 1.0
    for M in target_shape:
        norm *= 2 * (M + 1)
    return dstn_type1(resized * scale) / norm


def solve_direct(rhs, h):
    """Solve -Laplacian u = f on the interior grid with Dirichlet BCs.

    Returns the discrete-exact solution of the (2d+1)-point discretization.
    Varying the grid size sweeps out discretization error u_h - u_exact,
    which is O(h^2) for smooth f.

    rhs: array of shape (N_0, ..., N_{d-1})
    h: scalar grid spacing (assumed isotropic)
    """
    d = rhs.ndim
    f_hat = dstn_type1(rhs)

    eigenvalues = jnp.zeros(rhs.shape, dtype=rhs.dtype)
    for a in range(d):
        N = rhs.shape[a]
        k = jnp.arange(1, N + 1, dtype=rhs.dtype)
        lam = (2.0 / (h * h)) * (jnp.cos(jnp.pi * k / (N + 1)) - 1.0)
        reshape_to = [1] * d
        reshape_to[a] = N
        eigenvalues = eigenvalues + lam.reshape(reshape_to)

    u_hat = f_hat / (-eigenvalues)

    norm = 1.0
    for N in rhs.shape:
        norm *= 2 * (N + 1)
    return dstn_type1(u_hat) / norm
