# Multi-fidelity NACA airfoil RANS dataset generation

This folder generates the airfoil dataset of the paper: 2D incompressible RANS simulations
(SST k-ω, OpenFOAM v2506) of randomly sampled NACA 4- and 5-digit airfoils at Reynolds numbers
between 2·10⁶ and 6·10⁶ and angles of attack between −5° and 15°, each solved on 11 meshes of
increasing resolution. The mesh level is the fidelity axis. The CFD setup follows AirfRANS
([Bonnet et al., NeurIPS 2022](https://arxiv.org/abs/2212.07564)); the case generation and
post-processing code of the AirfRANS authors is vendored here, see [Vendored code](#vendored-code).

## Requirements

- OpenFOAM v2506 ([openfoam.com](https://www.openfoam.com)) with its MPI, sourced in the shell
  that runs the campaign (`source /path/to/OpenFOAM-v2506/etc/bashrc`). The driver calls
  `blockMesh`, `checkMesh`, `decomposePar`, `simpleFoam`, `reconstructPar`, `foamLog` and `foamToVTK`.
- Python 3.12 with numpy, scipy, pandas, pyyaml, matplotlib, h5py, tqdm and
  **pyvista 0.45.2 with VTK 9.4.2**. The VTK pin matters: the error-column stage resamples
  fields with VTK's probe filter, and VTK ≥ 9.5 returns wrong values on these one-cell-thick
  meshes (and permutes the point order of the extracted slices). The root environment of the
  repository provides these packages.
- A CPU machine. Each solve uses `openfoam.n_proc` (16) MPI ranks; `runtime.max_parallel`
  solves run concurrently.
- Disk: about 1.5 GB per case of raw OpenFOAM output including VTK, 86 MB per case processed.
  The finest level of one case takes roughly 25 to 35 minutes on 16 cores.

## Folder layout

| file | purpose |
|---|---|
| [`run_pipeline.py`](run_pipeline.py), [`pipeline.py`](pipeline.py) | campaign driver: fidelity schedule, OpenFOAM runs, validation, convergence analysis |
| [`generate_authors_sampled_config.py`](generate_authors_sampled_config.py) | samples a random case list into a campaign config |
| [`config_default.yaml`](config_default.yaml) | minimal example config (one case) |
| [`config_mf_dataset.yaml`](config_mf_dataset.yaml), [`config_mf_dataset_2.yaml`](config_mf_dataset_2.yaml) | the two campaigns of the paper dataset (1,000 cases each) |
| [`check_run_completion.py`](check_run_completion.py), [`delete_incomplete_folders.py`](delete_incomplete_folders.py) | find and clear partially written runs before resuming a campaign |
| [`postprocessing_parallel.py`](postprocessing_parallel.py) | raw OpenFOAM/VTK output → one HDF5 file per case and level |
| [`postprocessing_parallel_error_calculation.py`](postprocessing_parallel_error_calculation.py) | relative L2 error of every level against the finest level |
| [`filter_metadata.py`](filter_metadata.py) | clean-case filter and cost columns, writes the final `metadata.csv` |
| [`fuse_v2.py`](fuse_v2.py) | merges two processed campaigns into one dataset |
| [`NACA_simulation/`](NACA_simulation/), [`airfrans_lib/`](airfrans_lib/) | vendored AirfRANS code |

## Configuration

A campaign is one YAML file. Paths may be relative, in which case they are resolved against the
repository root; the campaign is written to `<output_root>/runs/<run_name>/`.

```yaml
output_root: _data_test/airfoil      # use shared storage for a real campaign
run_name: my_campaign                # or pass --run-name; default is a timestamp
runtime:
  max_parallel: 11                   # concurrent solves over the whole campaign (16 ranks each)
  skip_existing: true                # makes an interrupted campaign resumable
  stop_on_error: false
  generate_vtk: true
openfoam:
  turbulence: SST
  n_proc: 16
  temperature: 298.15
  domain_L: 200.0
  n_iter: {mode: authors, authors_low_aoa: 20000, authors_high_aoa: 40000, aoa_threshold_deg: 10.0}
fidelity:
  n_levels: 11
  exponent: 2.3
  ranges: {y_h: {coarse: 2.0e-05, fine: 2.0e-06}, ...}
cases:
  - {name: mf_dataset000_naca46119_re4.071e6_aoam3p44, reynolds: 4070646.085, aoa_deg: -3.437, digits: [4, 6, 1, 19]}
```

The driver queues all (case, level) solves of the campaign and runs `max_parallel` of them at a
time, so with `max_parallel: 5` and 16 ranks each, 80 cores are busy and cases overlap.

The 11 fidelity levels are defined by six mesh controls of the AirfRANS block mesh (first cell
height `y_h`, `y_hd`, chordwise spacing `x_h`, and the growth ratios `y_exp`, `x_exp`, `x_expd`).
Level `i` maps to `t = (i/10)^exponent` and every control is interpolated geometrically between
its `coarse` and `fine` value. Level `00` is the coarsest mesh, level `10` the finest. The
iteration count follows the AirfRANS protocol and is not part of the fidelity axis: 20,000
iterations, 40,000 for |AoA| > 10°.

`generate_authors_sampled_config.py` writes a config with a random case list (Reynolds number
and angle of attack uniform in the given ranges, half NACA 4-digit and half 5-digit profiles
with random camber, position and thickness). All non-case settings are copied from
`--template`; use one of the paper configs as the template, otherwise the example schedule of
`config_default.yaml` is copied instead of the paper's:

```bash
python generate_authors_sampled_config.py --template config_mf_dataset.yaml --output config_my_campaign.yaml \
    --num-cases 1000 --seed 42 --re-min-million 2.0 --re-max-million 6.0 --aoa-min -5 --aoa-max 15 --name-prefix mf_dataset
```

With seed 42 this reproduces the case list of `config_mf_dataset.yaml` exactly.

## Running a campaign

```bash
python run_pipeline.py --config config_my_campaign.yaml --run-name my_campaign --max-parallel 5
python run_pipeline.py --config config_my_campaign.yaml --run-name my_campaign --dry-run          # fidelity table and plan only
python run_pipeline.py --config config_my_campaign.yaml --run-name smoke --case-limit 2 --fidelity-ids 00,05,10
python run_pipeline.py --config config_my_campaign.yaml --run-name my_campaign --just-init        # meshes only
```

Options: `--run-name` (run directory name, overrides `run_name` in the config, default a
timestamp), `--max-parallel` (overrides `runtime.max_parallel`), `--case-limit N` (first N cases
of the config), `--fidelity-ids 00,05,10` (subset of levels), `--just-init` (mesh only),
`--dry-run`, `--skip-analysis` (no convergence analysis at the end), `--overwrite-run` (delete an
existing run directory first). A dry run already creates the run directory, so the real run that
follows reports that it resumes the existing directory; that is expected.

The run directory receives `fidelity_table.csv`, `execution_plan.csv` (the `n_iter_effective`
column is the iteration count actually used), `run_metadata.json`, the OpenFOAM cases under
`raw/<case_id>/fid_XX/`, and per-case convergence plots and metrics under `analysis/` (Cd, Cl
and field errors against the finest level, Cp and Cf distributions), with a per-level summary in
`analysis/convergence_summary.csv`. Rerunning the same command resumes an interrupted campaign:
finished case/level folders are skipped.

To find levels that were interrupted mid-solve (missing metadata, truncated solver log), probe the
run directory and clear the partial folders before resuming:

```bash
python check_run_completion.py --run-dir <output_root>/runs/<run_name>
python delete_incomplete_folders.py --completion-csv <output_root>/runs/<run_name>/completion_check.csv
```

## From raw runs to the dataset

### 1. Post-process to HDF5

```bash
python postprocessing_parallel.py --dataset-dir <output_root>/runs/<run_name>/raw --out-dir <processed_dir> --max-workers 32
```

Each level's VTK output is clipped and sliced to the 2D AirfRANS convention (internal field,
airfoil surface, freestream boundary), the surface points are mapped onto the internal wall
nodes, and the result is written as `<processed_dir>/<sample_id>/<fidelity_id>.h5` together with
`metadata_unfiltered.csv` and `sample_info.csv` (sample id ↔ case id).

### 2. Error columns

```bash
python postprocessing_parallel_error_calculation.py --dataset-dir <output_root>/runs/<run_name>/raw \
    --metadata-csv <processed_dir>/metadata_unfiltered.csv --max-workers 32
```

Resamples every lower level onto the finest mesh of its case and adds
`rel_l2_error_{ux,uy,p,wss}` and their mean `rel_l2_error` to the metadata.

### 3. Clean-case filter

```bash
python filter_metadata.py --metadata-csv <processed_dir>/metadata_unfiltered.csv \
    --raw-dirs <output_root>/runs/<run_name>/raw
```

A case enters the dataset only if its finest-level solve is stable (relative oscillation of Cd
over the last 20% of the iterations at most 5%) and its coarsest level is still a usable
approximation (relative L2 error at most 0.4). Cases are dropped as a whole, so every case in the
dataset has all 11 levels. The script also sets the cost columns: `simulation_time_raw` is the
solver wall time, `simulation_time` (the cost used for training) is wall time × 16 cores, halved
for |AoA| > 10° because those cases run twice the iterations. It writes `metadata.csv`, which is
what the training code reads; `metadata_unfiltered.csv` keeps all cases with an `is_clean` flag.

### 4. Merging campaigns

```bash
python fuse_v2.py --root <runs_dir> --batch1 mf_dataset_processed --batch2 mf_dataset_2_processed --out mf_dataset_processed_v2
```

`--batch1`, `--batch2` and `--out` are folder names under `--root`. Both campaigns go through
steps 1 to 3 independently (the filter uses absolute thresholds, so filtering per campaign equals
filtering jointly). The merge hardlinks the sample folders, offsets the sample ids of the second
campaign by `--offset` (default 1000) and concatenates the metadata tables.

## Dataset layout

`<processed_dir>/<sample_id>/<fidelity_id>.h5` holds:

- `data/fields`, shape (N, 13), with the columns named in `channels/`: `coords`, `u_in` (inlet
  velocity), `sdf`, `normals`, `on_surface`, `u`, `p`, `wss`;
- `data/edge_index`, the undirected mesh edges, and `mesh/connectivity`, the triangles;
- the airfoil surface line: `mesh/aero_line_connectivity`, `mesh/aero_line_length`,
  `mesh/aero_line_normals`, `mesh/aero_to_internal` (surface node → internal node);
- `cond/fidelity_raw`, the level's fidelity value, and `reference/cd_openfoam`,
  `reference/cl_openfoam`, the force coefficients of that level;
- attributes `metadata_json` (the level's metadata), `coeff_ref_json` and `vtk_internal_path`
  (the raw file the sample was built from).

`metadata.csv` has one row per (sample, level) with the case parameters, mesh size, errors,
costs and `cd_rel_amp`.

## Reproducibility

The campaign driver is deterministic (same config → same meshes, cell counts and plan), but the
CFD result is reproducible only to solver noise across machines: the airfoil polyline written into
`blockMeshDict` can differ by one ulp between platforms, after which the SIMPLE iteration takes a
different path. In a rerun of five cases on a different machine with the same OpenFOAM build,
about half of the 55 levels were bitwise identical to the published files and the rest agreed to
1e-8 to 1e-5 in relative L2, with single levels settling on a slightly different oscillating state
(up to a few percent). Force coefficients agree to four digits. The post-processing stages are
deterministic given the pinned VTK version.

## The paper dataset

Two campaigns of 1,000 cases each (`config_mf_dataset.yaml`, drawn with the sampler above and
seed 42, and `config_mf_dataset_2.yaml`), post-processed, filtered and merged as described:
1,627 clean cases × 11 levels. Mean mesh size grows from about 7,600 nodes at level `00` to
180,000 at level `10`, the mean field error of the coarsest level is about 30%. The training
code reads it through `data/airfoil/processed`.

## Vendored code

`NACA_simulation/` and `airfrans_lib/` are plain copies of the AirfRANS authors' repositories,
MIT licensed:

| folder | upstream | commit | changes |
|---|---|---|---|
| [`airfrans_lib/`](airfrans_lib/) | [Extrality/airfrans_lib](https://github.com/Extrality/airfrans_lib) (v0.1.5.1) | [`d35d403`](https://github.com/Extrality/airfrans_lib/commit/d35d4035d8ba6fa98c1a6662be925c2fc777610b) | none |
| [`NACA_simulation/`](NACA_simulation/) | [Extrality/NACA_simulation](https://github.com/Extrality/NACA_simulation) | [`2dbf00c`](https://github.com/Extrality/NACA_simulation/commit/2dbf00c5ecbb8d6cef720db1e72c1fe40307eeab) | three files, see below |

Changes in `NACA_simulation/`:

- [`simulation_generator.py`](NACA_simulation/simulation_generator.py): OpenFOAM calls raise on
  failure and log stderr, `reconstructPar` only runs for parallel cases, `foamToVTK` also exports
  the force-coefficient fields, seaborn is optional.
- [`Simulations/airFoil2DInit/system/controlDict.orig`](NACA_simulation/Simulations/airFoil2DInit/system/controlDict.orig):
  force coefficients are written every iteration (the stability filter reads this history), and a
  second `forceCoeffs` function object writes the coefficient fields.
- [`Simulations/airFoil2DInit/system/fvSolution`](NACA_simulation/Simulations/airFoil2DInit/system/fvSolution):
  under-relaxation factors p = 0.3 and 0.7 for the equations (upstream: 1 and 0.9).

The fidelity schedule, the iteration protocol and the orchestration live in `pipeline.py`.
