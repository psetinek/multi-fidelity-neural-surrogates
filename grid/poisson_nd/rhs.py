"""n-D random Fourier RHS with a Hann window and unit peak magnitude.

f(x) = W(x) * sum_{k in {1..M}^d} a_k * prod_a sin(k_a (2 pi / L) x_a + phi_{a,k})

- a_k ~ U[-1, 1], phi_{a,k} ~ U[0, 2 pi].
- W is the outer product of 1D Hann windows; its zeros at the domain
  boundaries make f compatible with homogeneous Dirichlet BCs.

`generate_rhs` normalizes so max|f| == 1 on the given grid. For convergence
studies that need the *same* analytical field sampled at multiple
resolutions, use `sample_coefficients` + `evaluate_rhs` directly and supply
an external normalization (e.g. computed once on a reference grid).
"""

import jax
import jax.numpy as jnp
import jax.random as jr


def sample_coefficients(key, num_modes, ndim):
    """Draw random amplitudes and per-axis phases for the Fourier field.

    Returns (amplitudes, phases) where amplitudes has shape (M,)^d and phases
    is a list of d arrays each of shape (M,)^d. These fully determine the
    analytical RHS as a continuous function of x; grid resolution enters
    only when `evaluate_rhs` samples it.
    """
    key_a, key_phase = jr.split(key, 2)
    amplitudes = jr.uniform(key_a, (num_modes,) * ndim, minval=-1.0, maxval=1.0)
    phase_keys = jr.split(key_phase, ndim)
    phases = [
        jr.uniform(phase_keys[a], (num_modes,) * ndim, minval=0.0, maxval=2 * jnp.pi)
        for a in range(ndim)
    ]
    return amplitudes, phases


def evaluate_rhs(amplitudes, phases, shape, domain_extent=1.0):
    """Evaluate the windowed Fourier field at grid points x_a = (i+1) h_a.

    Sum is accumulated mode-tuple by mode-tuple via `jax.lax.fori_loop` so
    the working memory is only `~2 * prod(shape)` at fp32 — crucial in 3D
    where the naive `(M,..,M, N_1,..,N_d)` broadcast blows past GPU VRAM
    (`M=15, N=256, d=3` → 226 GB intermediate).

    No normalization is applied; the caller is expected to divide by a
    reference scale when comparing across grids.
    """
    d = len(shape)
    M = amplitudes.shape[0]
    L = domain_extent
    dtype = amplitudes.dtype
    omega = 2.0 * jnp.pi / L

    x_grids = [
        (jnp.arange(shape[a]) + 1).astype(dtype) * L / (shape[a] + 1) for a in range(d)
    ]
    modes = jnp.arange(1, M + 1, dtype=dtype)

    num_tuples = M**d
    amps_flat = amplitudes.reshape(-1)

    # For each axis a, precompute sin(k_a^{(t)} * ω * x_a + φ_{a,t}) for every
    # (mode tuple t, grid point x) pair. Shape (M^d, N_a) per axis — tiny:
    # at d=3, M=15, N=256 this is 3 × 3.4 MB across all axes.
    mode_grids = jnp.meshgrid(*[modes] * d, indexing="ij")
    k_per_tuple = [g.reshape(-1) for g in mode_grids]  # each shape (M^d,)

    sines_per_axis = []
    for a in range(d):
        phi_flat = phases[a].reshape(-1)  # (M^d,)
        arg = k_per_tuple[a][:, None] * omega * x_grids[a][None, :] + phi_flat[:, None]
        sines_per_axis.append(jnp.sin(arg))  # (M^d, N_a)

    def body(t, vals):
        term = amps_flat[t]
        for a in range(d):
            s = sines_per_axis[a][t]  # (N_a,)
            reshape_to = [1] * d
            reshape_to[a] = shape[a]
            term = term * s.reshape(reshape_to)
        return vals + term

    vals = jax.lax.fori_loop(0, num_tuples, body, jnp.zeros(shape, dtype=dtype))

    window = jnp.ones(shape, dtype=dtype)
    for a in range(d):
        w1d = 0.5 * (1 - jnp.cos(2 * jnp.pi * x_grids[a] / L))
        w_shape = [1] * d
        w_shape[a] = shape[a]
        window = window * w1d.reshape(w_shape)

    return vals * window


def generate_rhs(key, shape, *, domain_extent=1.0, num_modes=15):
    """Random windowed Fourier RHS normalized to max|f| = 1 on this grid.

    shape: tuple of per-axis grid sizes (interior points only).
    domain_extent: side length L of the hyper-cube [0, L]^d.
    num_modes: number of Fourier modes per axis.
    """
    amplitudes, phases = sample_coefficients(key, num_modes, len(shape))
    vals = evaluate_rhs(amplitudes, phases, shape, domain_extent)
    return vals / jnp.max(jnp.abs(vals))
