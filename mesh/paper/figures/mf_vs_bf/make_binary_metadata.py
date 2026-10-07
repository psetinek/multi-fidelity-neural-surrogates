"""Binary-fidelity metadata for the multi- vs bi-fidelity comparison runs.

Restricts the hyperelasticity solver-truncation metadata to the two menu endpoints
(solver_max_its = 20 and 190). Training reads it via dataset.hparams.metadata_path while
data_path stays the full dataset, so sample ids, splits and the HF-only validation and test
sets are identical to the main solver experiment.

Sweep used in the paper: 7 equally spaced sample counts n over the full feasible span
[B/190, B/20] = [B~, 9.5 B~], endpoints included (pure HF to pure LF). With cost_column =
solver_max_its a cell has n_HF = (B - 20 n) / 170 and n_LF = n - n_HF.

Output: metadata_binary_hyperelasticity_solver.csv (this directory)
Usage:  python make_binary_metadata.py
"""
import os.path as osp

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
REPO = osp.abspath(osp.join(HERE, "..", "..", ".."))
SRC = f"{REPO}/data/hyperelasticity/solver_truncation/processed/metadata.csv"
OUT = f"{HERE}/metadata_binary_hyperelasticity_solver.csv"

LEVELS = (20, 190)  # menu endpoints: coarsest + converged
C_HF = 190.0

df = pd.read_csv(SRC, dtype={"sample_id": str, "fidelity_id": str})
sub = df[df.solver_max_its.isin(LEVELS)].copy()
counts = sub.solver_max_its.value_counts()
print(f"source: {len(df)} rows, {df.solver_max_its.nunique()} levels")
print(f"binary: {len(sub)} rows | per level: {counts.to_dict()}")
assert set(counts.index) == set(LEVELS)
assert counts.nunique() == 1, "level counts differ -- incomplete dataset?"
sub.to_csv(OUT, index=False)
print(f"wrote {OUT}")

# 7-point linspace over the full binary-feasible span, endpoints included
GRIDS = {200: [200, 483, 767, 1050, 1333, 1617, 1900],
         1600: [1600, 3867, 6133, 8400, 10667, 12933, 15200]}
for bt, grid in GRIDS.items():
    budget = bt * C_HF
    n_hf = [(budget - 20 * n) / 170 for n in grid]
    assert all(v >= 0 for v in n_hf), "cell infeasible on binary menu"
    print(f"\nB~={bt} (budget={budget:.0f}): n grid = {','.join(map(str, grid))}")
    print(f"   n_HF per cell: {[round(v, 1) for v in n_hf]}")
