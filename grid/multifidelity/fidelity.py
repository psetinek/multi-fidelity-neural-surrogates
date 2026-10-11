"""Fidelity specification and budget-bounded sampler.

`FidelitySpec` bundles an axis, a discrete set of fidelity values, a cost
function, and the cached `xi_max` used for log conditioning. `get_exp_pmf`
is a softmax mixture over the menu: min-max-clamp the
fidelity values to [0, 1] before the softmax so λ has a consistent meaning
regardless of the raw ξ range. Boundary cases: `λ = 0` → uniform;
`λ → +∞` → delta on ξ_max; `λ → -∞` → delta on ξ_min.

Clamping (rather than a softmax over raw ξ) keeps λ comparable across
axes: with raw ξ, wide-range discrete menus (e.g. N∈{64..256}) collapse
to a single fidelity for every non-trivial |λ|.

`sample_budget` draws i.i.d. from the PMF via `jax.random.choice`,
accumulates per-sample costs, and truncates at the first sample that
crosses the budget — so the realised cost is always ≥ `budget` by at
most one sample.
"""

import math
from dataclasses import dataclass
from typing import Callable, Literal

import jax.numpy as jnp
import jax.random as jr


@dataclass(frozen=True)
class FidelitySpec:
    axis: Literal["truncation", "discretization"]
    values: tuple
    cost_fn: Callable[[float], float]
    xi_max: float

    def costs(self) -> tuple:
        return tuple(self.cost_fn(v) for v in self.values)


def get_exp_pmf(fidelities, lambd):
    """Softmax PMF over a discrete fidelity set (JAX).

    Steps: min-max clamp fidelities to ξ̄ ∈ [0, 1], then return
    `softmax(λ · ξ̄)`. `lambd is None` or |λ| ≤ 1e-8 (or a degenerate
    single-valued menu) returns uniform.

    With clamping (see module docstring), the same λ produces the same
    PMF shape on any menu — λ ≈ ±20 saturates to a near-delta at
    ξ_max / ξ_min regardless of the raw ξ range.
    """
    fids = jnp.asarray(fidelities).astype(jnp.float32)
    n = len(fids)
    if lambd is None or abs(lambd) <= 1e-8:
        return jnp.full((n,), 1.0 / n)
    fid_min = float(jnp.min(fids))
    span = float(jnp.max(fids) - fid_min)
    if span <= 1e-12:  # degenerate single-valued menu
        return jnp.full((n,), 1.0 / n)
    fids_bar = (fids - fid_min) / span  # ξ̄ ∈ [0, 1]
    exponents = lambd * fids_bar
    exponents = exponents - jnp.max(exponents)
    unnormalized = jnp.exp(exponents)
    return unnormalized / jnp.sum(unnormalized)


def sample_budget(key, spec: FidelitySpec, lambd, budget, *, padded_draw=None):
    """Draw ξ values i.i.d. from the PMF until cumulative cost crosses the
    budget; return the prefix of draws and the realised cost.

    - `padded_draw` upper-bounds how many i.i.d. draws are taken up front.
      Defaults to `ceil(budget / min(costs)) + 1`, which is guaranteed to
      cover any realised draw sequence since even all-minimum-cost draws
      would cross the budget within that many samples.
    - Truncation point: the smallest `M` such that
      `cumulative_cost[M-1] >= budget`. This includes the first sample
      that crosses the budget, i.e. we truncate
      after the first sample that crosses the budget.
    """
    costs_tuple = spec.costs()
    min_cost = float(min(costs_tuple))
    if budget < min_cost:
        raise ValueError(
            f"Budget {budget} is smaller than the minimum per-sample cost "
            f"{min_cost}; no sample would fit."
        )

    if padded_draw is None:
        padded_draw = math.ceil(budget / min_cost) + 1

    pmf = get_exp_pmf(spec.values, lambd)
    idx = jr.choice(
        key,
        len(spec.values),
        shape=(padded_draw,),
        p=pmf,
    )

    values_arr = jnp.asarray(spec.values)
    costs_arr = jnp.asarray(costs_tuple)

    xis = values_arr[idx]
    sample_costs = costs_arr[idx]
    cumcost = jnp.cumsum(sample_costs)

    crossing = int(jnp.searchsorted(cumcost, budget, side="left"))
    if crossing >= padded_draw:
        raise RuntimeError(
            f"padded_draw={padded_draw} was insufficient; budget never crossed"
        )
    M = crossing + 1
    return xis[:M], float(cumcost[M - 1])
