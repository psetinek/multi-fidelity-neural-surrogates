"""JAX-native Poisson solver for 1D/2D/3D with homogeneous Dirichlet BCs.

Two error-control axes:
- Discretization error: `solve_direct` at varying resolution.
- Truncation error: `build_multigrid_iterator` with a variable V-cycle count.
"""

from poisson_nd.dst import dst_resample, solve_direct
from poisson_nd.multigrid import (
    build_multigrid_iterator,
    build_multigrid_solver,
    prolongate,
    restrict,
)
from poisson_nd.operators import laplacian
from poisson_nd.rhs import generate_rhs

__all__ = [
    "build_multigrid_iterator",
    "build_multigrid_solver",
    "dst_resample",
    "generate_rhs",
    "laplacian",
    "prolongate",
    "restrict",
    "solve_direct",
]
