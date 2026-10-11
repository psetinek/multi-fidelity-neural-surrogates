"""Convergence studies for the poisson_nd solvers (paper appendix).

- `truncation`: multigrid error vs. number of V-cycles K (all MG variants).
- `mg_at_N64`: the N = 64 slice of the truncation study used in the paper;
  re-plots `results/truncation.csv` without re-running the solves.
- `discretization`: discretization error vs. grid resolution, O(h^2).

Run from the repository root, e.g. `python -m convergence_studies.truncation`.
Each script writes CSV results under `results/` and PNG + PDF figures under
`figures/`, both relative to this package. `style` holds the figure style
shared with `scripts/plot_fidelity_examples.py`.
"""
