"""Calibration overhead of the extrapolation recipe (appendix numbers).

Linear-elasticity solver menu and budget ladder: pilot budgets B~ = 100, 200, 400, deployment
at the held-out budgets with the mixtures predicted by the recipe (recipe_results.csv).
A sample at fidelity f serves every dataset containing f; the scan pays for its per-level
envelope, and after deployment only the residual [scan - deploy]_+ counts as overhead.

Needs data/linear_elasticity/solver_truncation/processed/metadata.csv.
Usage:  python overhead.py   (after recipe.py)
"""
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", ".."))
META = f"{REPO}/data/linear_elasticity/solver_truncation/processed/metadata.csv"
CELLS = f"{REPO}/paper/results/linear_elasticity_solver_truncation.csv"
FIT = f"{HERE}/recipe_results.csv"
Q = 0.99993  # configs/dataset/linear_elasticity_solver.yaml

meta = pd.read_csv(META)
K = np.sort(meta.solver_max_its.unique()).astype(float)
E = Q ** K
f = (E[0] - E) / (E[0] - E[-1])   # base.py::_linearize_fidelities
c = K / K[-1]                     # per-sample cost in B~ units (HF sample = 1)


def plan(lam, B):
    """Expected per-level budget allocation (B~ units) of sampling plan (lam, B)."""
    p = np.exp(lam * f)
    p /= p.sum()
    n_total = B / (p * c).sum()
    return n_total * p * c


cells = pd.read_csv(CELLS)
s = cells[(cells.model_name == "Transolver") & (~cells.hf_only)]
PILOT_B = [100, 200, 400]
scan_plans = []
for B in PILOT_B:
    lams = sorted(s[s.b_tilde == B].lambd.unique())
    print(f"pilot B~={B}: lambda grid {np.round(lams, 2)}")
    scan_plans += [plan(lam, B) for lam in lams]

naive = 7 * sum(PILOT_B)
env = np.max(scan_plans, axis=0)
env_b3 = np.max(scan_plans[-7:], axis=0)
assert np.allclose(env, env_b3), "B3-only envelope no longer equals full envelope"
g3 = sorted(s[s.b_tilde == 400].lambd.unique())
env_cont = np.max([plan(l, 400) for l in np.linspace(g3[0], g3[-1], 4000)], axis=0)
print(f"continuous-sweep envelope (sup over [{g3[0]:.1f},{g3[-1]:.1f}] at B3=400): "
      f"{env_cont.sum():7.0f} (+{env_cont.sum()/env.sum()-1:.1%} vs 7-point grid)")

r = pd.read_csv(FIT).set_index("setting").loc["lin_solv_Transolver"]
DEP_B = [800, 1600, 3200, 6400]
lam_hat = [r.intercept + r.slope * np.log2(B) for B in DEP_B]
print(f"menu: {len(K)} levels, cost span {c[-1]/c[0]:.1f}x")
print(f"predicted deployment lambdas: {np.round(lam_hat, 2)} "
      f"(CSV check: {[round(r[f'lambda_pred_{i}'], 2) for i in range(1, 5)]})")
print(f"naive scan cost 7*(B1+B2+B3): {naive} B~-units "
      f"({naive/DEP_B[0]:.3f}x = {naive/DEP_B[0]:.0%} of first deployment)")
print(f"scan envelope: {env.sum():.0f} B~-units")


def sweep_env(B):
    lams = sorted(s[s.b_tilde == B].lambd.unique())
    return np.max([plan(l, B) for l in lams], axis=0)


for B, lh in zip(DEP_B, lam_hat):
    dep = plan(lh, B)
    net = np.maximum(env, dep).sum() / dep.sum() - 1
    alt = sweep_env(B).sum() / dep.sum() - 1
    print(f"deploy once at B~={B:>4} ({B/400:.0f}x B3): NET overhead +{net:.0%} "
          f"(sweep-at-deployment alternative: +{alt:.0%})")
