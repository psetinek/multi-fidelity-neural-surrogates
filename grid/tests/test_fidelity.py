"""Tests for the fidelity spec, PMF, and budget-bounded sampler."""

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from multifidelity.fidelity import FidelitySpec, get_exp_pmf, sample_budget


def _default_spec():
    return FidelitySpec(
        axis="truncation",
        values=tuple(range(1, 21)),
        cost_fn=lambda K: float(K),
        xi_max=20.0,
    )


def test_pmf_normalises():
    pmf = get_exp_pmf(tuple(range(1, 21)), 0.3)
    assert jnp.isclose(jnp.sum(pmf), 1.0)
    assert jnp.all(pmf > 0)


@pytest.mark.parametrize("lambd", [None, 0.0, 1e-12])
def test_pmf_uniform_at_zero_lambda(lambd):
    pmf = get_exp_pmf(tuple(range(1, 21)), lambd)
    expected = jnp.full((20,), 1.0 / 20)
    assert jnp.allclose(pmf, expected)


def test_pmf_monotone_skew_with_lambda():
    fidelities = tuple(range(1, 21))
    pmf_neg = get_exp_pmf(fidelities, -2.0)
    pmf_pos = get_exp_pmf(fidelities, 2.0)
    # Negative lambda: weight on low fidelity; first > last.
    assert pmf_neg[0] > pmf_neg[-1]
    # Positive lambda: weight on high fidelity; last > first.
    assert pmf_pos[-1] > pmf_pos[0]


def test_pmf_boundary_cases():
    """Large |λ| saturates to a near-delta on ξ_max/ξ_min."""
    fidelities = (1, 2, 3, 4, 5, 6, 7)
    pmf_pos = get_exp_pmf(fidelities, 50.0)
    assert pmf_pos[-1] > 0.999
    pmf_neg = get_exp_pmf(fidelities, -50.0)
    assert pmf_neg[0] > 0.999


def test_pmf_cross_axis_consistency():
    """Min-max clamping makes the PMF shape depend only on the
    relative spacing of the menu, not on the absolute ξ values. Two menus
    that differ only by an affine transform should give identical PMFs.
    """
    pmf_small = get_exp_pmf(tuple(range(1, 8)), 2.0)
    pmf_shifted = get_exp_pmf(tuple(range(64, 71)), 2.0)  # same shape, shifted
    np.testing.assert_allclose(
        np.asarray(pmf_small), np.asarray(pmf_shifted), rtol=1e-5, atol=1e-6
    )


def test_pmf_no_pathology_on_wide_menu():
    """A softmax over raw ξ collapses to a delta on a wide menu
    (N∈{64..256}) at |λ|≈1; the clamped form gives a genuine mixture there.
    """
    menu = (64, 96, 128, 192, 256)
    pmf = np.asarray(get_exp_pmf(menu, -1.0))
    # Raw-ξ softmax would put ≈1.0 on N=64 and ≈0 elsewhere; we expect a
    # genuine mixture: no cell dominates, no cell collapses, monotone
    # decrease under negative λ.
    assert pmf.max() < 0.5, f"No cell should dominate; got {pmf}"
    assert (pmf > 0.05).all(), f"Every cell should have >5% mass; got {pmf}"
    assert all(pmf[i] > pmf[i + 1] for i in range(len(pmf) - 1))


def test_sample_budget_respects_budget():
    spec = _default_spec()
    xis, cost = sample_budget(jr.PRNGKey(0), spec, 0.1, budget=500.0)
    assert xis.shape[0] >= 1
    # Realised cost should be ≥ budget (we truncate at the first crossing).
    assert cost >= 500.0
    # But not more than budget + max per-sample cost.
    assert cost <= 500.0 + max(spec.costs())


def test_sample_budget_below_min_cost_raises():
    spec = _default_spec()
    with pytest.raises(ValueError):
        sample_budget(jr.PRNGKey(0), spec, 0.0, budget=0.5)


def test_sample_budget_determinism():
    """Same key → same xis."""
    spec = _default_spec()
    key = jr.PRNGKey(42)
    a_xis, a_cost = sample_budget(key, spec, 1.0, budget=200.0)
    b_xis, b_cost = sample_budget(key, spec, 1.0, budget=200.0)
    assert jnp.array_equal(a_xis, b_xis)
    assert a_cost == b_cost
