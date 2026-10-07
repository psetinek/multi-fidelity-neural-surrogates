# Data generation

The folders [`linear_elasticity/`](linear_elasticity/), [`hyperelasticity/`](hyperelasticity/) and [`airfoil/`](airfoil/) generate the multi-fidelity datasets used in the respective mesh experiments of the paper.
Each pipeline includes simulation and postprocessing such that generated datasets can be used directly with the shipped training code in [`mf_surrogates/data/`](../mf_surrogates/data/).
We include the settings to reproduce the paper's datasets but feel free to adapt them to your needs.

A summary of the simulations:

| Folder | Physics | Solver | Fidelity axis (levels) | Entry points |
|---|---|---|---|---|
| [`linear_elasticity/`](linear_elasticity/) | Perforated plate, linear elasticity (Hooke) | FEniCSx / dolfinx + Gmsh | solver truncation: Gauss-Seidel iterations (11), or mesh refinement (21) | [`generate_dataset_solver_error.py`](linear_elasticity/generate_dataset_solver_error.py), [`generate_dataset_discretization_error.py`](linear_elasticity/generate_dataset_discretization_error.py) |
| [`hyperelasticity/`](hyperelasticity/) | Same geometry, Neo-Hookean hyperelasticity | FEniCSx / dolfinx + Gmsh | solver truncation: modified-Newton iterations (16), or mesh refinement (21) | [`generate_dataset_solver_error.py`](hyperelasticity/generate_dataset_solver_error.py), [`generate_dataset_discretization_error.py`](hyperelasticity/generate_dataset_discretization_error.py) |
| [`airfoil/`](airfoil/) | 2D NACA airfoils, incompressible RANS (SST k-ω), adapted from [1] | OpenFOAM v2506 | mesh refinement (11) | [`run_pipeline.py`](airfoil/run_pipeline.py), then the postprocessing scripts (see [`airfoil/README.md`](airfoil/README.md)) |

All pipelines write `processed/<sample_id>/<fidelity_id>.h5` plus a `metadata.csv` with one row per (sample, level): fidelity value, cost column, mesh size and the error against the finest / converged solution.

[1] F. Bonnet, J. A. Mazari, P. Cinnella, P. Gallinari. *AirfRANS: High Fidelity Computational Fluid Dynamics Dataset for Approximating Reynolds-Averaged Navier–Stokes Solutions.* NeurIPS 2022, Datasets and Benchmarks Track. [arXiv:2212.07564](https://arxiv.org/abs/2212.07564).
The airfoil pipeline builds on the authors' simulation driver [`NACA_simulation/`](airfoil/NACA_simulation/) and post-processing library [`airfrans_lib/`](airfoil/airfrans_lib/), both MIT licensed. the vendored versions and our modifications are listed in [`airfoil/README.md`](airfoil/README.md).

## Prerequisites

### Linear elasticity & hyperelasticity

Because FEniCSx (dolfinx, PETSc, MPI) is distributed through conda-forge and cannot be installed with pip, a dedicated conda environment is needed for the two solid-mechanics pipelines.
The datasets were meshed with [Gmsh](https://gmsh.info) and simulated with [FEniCSx](https://fenicsproject.org).
[`environment_solid_mechanics.yml`](environment_solid_mechanics.yml) recreates this environment.
[`environment_solid_mechanics.lock.yml`](environment_solid_mechanics.lock.yml) is the exact export with build strings, for bit-for-bit reproduction.

In order to install, please run:

```bash
conda env create -f data_generation/environment_solid_mechanics.yml      # or environment_solid_mechanics.lock.yml
conda activate mf_solid_mechanics
```

Generate the datasets from the repository root. The defaults reproduce the paper (90,000 samples for the solver axes, seed 42, 92 worker processes; adjust `n_simulations` / `num_workers` at the bottom of each script). The solver scripts write to `data/<problem>/solver_truncation/{raw,processed}`:

```bash
cd data_generation/linear_elasticity
python generate_dataset_solver_error.py            # 90k samples x 11 Gauss-Seidel budgets
python generate_dataset_discretization_error.py    # mesh-refinement axis, 21 levels
cd ../hyperelasticity
python generate_dataset_solver_error.py            # 90k samples x 16 modified-Newton budgets
python generate_dataset_discretization_error.py    # mesh-refinement axis, 21 levels
```

Every sample is a random perforated plate (5 to `max_holes` holes, random radii and positions) under a random traction on the right edge.
Geometry and load are drawn sequentially from a single `np.random.seed(42)` stream in the parent process before the solves are dispatched, so sample `i` is the same for any `n_simulations >= i`.
The solver axis keeps one fixed fine mesh per sample and truncates the iterative solve at each budget; the discretization axis solves the same sample on 21 meshes of increasing resolution.

### Airfoil

To generate the airfoil simulations, use the same Python environment as for training the mesh-based surrogates (see the [root README](../README.md)); the post-processing needs pyvista 0.45.2 with VTK 9.4.2, newer VTK versions give wrong error columns (see [`airfoil/README.md`](airfoil/README.md), Requirements).
However, you will need OpenFOAM v2506 ([openfoam.com](https://www.openfoam.com), build `_615aae61d7-20250627`) and its bundled MPI.
Install OpenFOAM from the vendor packages or from source, then source it in the shell that runs the pipeline so that `simpleFoam`, `blockMesh` and `mpirun` are on `PATH`.

```bash
source /path/to/OpenFOAM-v2506/etc/bashrc
conda activate <root environment>
```

Each case runs its 11 fidelity levels with 16 OpenFOAM ranks and produces about 1.5 GB of raw output (VTK on) and 86 MB processed.
The full workflow (campaign config, CFD run, completion check, VTK to H5 postprocessing, per-sample error columns, clean-case filtering, batch fusion) is documented step by step in [`airfoil/README.md`](airfoil/README.md).