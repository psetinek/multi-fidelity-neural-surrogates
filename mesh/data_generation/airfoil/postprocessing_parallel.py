import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd
import pyvista as pv

from tqdm import tqdm


# Preprocessing parameters from authors
CLIP_BOUNDS = (-2, 4, -1.5, 1.5, 0, 1)
SLICE_NORMAL = (0, 0, 1)
SLICE_ORIGIN = (0, 0, 0.5)

INTERNAL_FIELDS = ["p", "U", "implicit_distance", "wallShearStress"]
AERO_FIELDS = ["p", "U", "Normals", "wallShearStress"]
FREE_FIELDS = ["p", "U", "wallShearStress"]


def select_fields(ds: pv.DataSet, names: List[str]) -> pv.DataSet:
    # Return a copy of ds carrying only selected point/cell arrays.
    out = ds.copy()
    for arr in list(out.point_data.keys()):
        if arr not in names:
            out.point_data.remove(arr)
    for arr in list(out.cell_data.keys()):
        if arr not in names:
            out.cell_data.remove(arr)
    return out


def find_vtk_paths(fid_dir: Path) -> tuple[Path, Path, Path]:
    vtk_root = fid_dir / "VTK"
    internal_candidates = sorted(vtk_root.glob("*/internal.vtu"))
    if not internal_candidates:
        raise FileNotFoundError(f"No internal.vtu found under {vtk_root}")

    internal_path = internal_candidates[-1]
    case_dir = internal_path.parent
    aero_path = case_dir / "boundary" / "aerofoil.vtp"
    free_path = case_dir / "boundary" / "freestream.vtp"

    if not aero_path.exists() or not free_path.exists():
        raise FileNotFoundError("Missing aerofoil.vtp and/or freestream.vtp")

    return internal_path, aero_path, free_path


def parse_force_coeff_reference(coeff_path: Path) -> Dict[str, float]:
    # Parse final row from forceCoeffs and keep named columns (robust to column order).
    with coeff_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_lines = [ln for ln in lines if ln.startswith("# Time")]
    header_cols = header_lines[-1].replace("#", "").split() if header_lines else []

    arr = np.loadtxt(coeff_path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    last = arr[-1]

    if header_cols and len(header_cols) == last.shape[0]:
        coeff = {col: float(last[i]) for i, col in enumerate(header_cols)}
    else:
        # fallback (expected OpenFOAM order)
        coeff = {"Time": float(last[0]), "Cd": float(last[1]), "Cl": float(last[4])}

    return coeff


def preprocess_case(fid_dir: Path) -> Dict[str, Any]:
    # Apply the same mesh transforms as preprocessing_parallel.py for one fidelity folder.
    internal_path, aero_path, free_path = find_vtk_paths(fid_dir)

    internal_raw = pv.read(internal_path)
    aero_raw = pv.read(aero_path)
    free_raw = pv.read(free_path)

    # Internal field preprocessing
    internal = internal_raw.clip_box(bounds=CLIP_BOUNDS, crinkle=True, invert=False)
    internal = internal.slice(normal=SLICE_NORMAL, origin=SLICE_ORIGIN, generate_triangles=False)
    internal = internal.compute_implicit_distance(aero_raw)
    internal = select_fields(internal, INTERNAL_FIELDS)

    # Airfoil patch preprocessing
    aero = aero_raw.compute_normals(point_normals=True, cell_normals=True, flip_normals=False)
    aero = aero.slice(normal=SLICE_NORMAL, origin=SLICE_ORIGIN, generate_triangles=False)
    aero = select_fields(aero, AERO_FIELDS)
    aero = aero.compute_cell_sizes(area=False, volume=False)

    # Freestream patch preprocessing
    freestream = free_raw.slice(normal=SLICE_NORMAL, origin=SLICE_ORIGIN, generate_triangles=False)
    freestream = select_fields(freestream, FREE_FIELDS)

    metadata = json.loads((fid_dir / "metadata.json").read_text(encoding="utf-8"))
    coeff_ref = parse_force_coeff_reference(
        fid_dir / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
    )

    return {
        "internal": internal,
        "aero": aero,
        "freestream": freestream,
        "metadata": metadata,
        "coeff_ref": coeff_ref,
        "vtk_internal_path": str(internal_path),
    }


def build_surface_mapping(internal: pv.DataSet, aero: pv.DataSet) -> tuple[np.ndarray, np.ndarray]:
    # Build mapping from aero point order -> internal point index on wall points.
    U = np.asarray(internal.point_data["U"])
    on_surface = U[:, 0] == 0.0

    surface_indices = np.where(on_surface)[0]
    surface_pts = np.asarray(internal.points)[surface_indices, :2]
    aero_pts = np.asarray(aero.points)[:, :2]

    lut = {tuple(pt.tolist()): i for i, pt in enumerate(surface_pts)}

    map_local = []
    for pt in aero_pts:
        key = tuple(pt.tolist())
        if key in lut:
            map_local.append(lut[key])
            continue

        d2 = np.sum((surface_pts - pt) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] > 1e-12:
            raise RuntimeError("Could not map aero point to internal wall point.")
        map_local.append(j)

    map_local = np.asarray(map_local, dtype=np.int64)
    aero_to_internal = surface_indices[map_local]

    return on_surface, aero_to_internal


def build_internal_edge_index(internal: pv.DataSet) -> np.ndarray:
    # Build undirected edge_index (2, 2E) from internal mesh edges.
    edges = internal.extract_all_edges()
    lines = np.asarray(edges.lines, dtype=np.int64).reshape(-1, 3)
    if not np.all(lines[:, 0] == 2):
        raise ValueError("Unexpected edge line format.")

    edge_dir = lines[:, 1:]
    n_edges = edge_dir.shape[0]

    edge_index = np.empty((2, 2 * n_edges), dtype=np.int64)
    edge_index[:, 0::2] = edge_dir.T
    edge_index[:, 1::2] = edge_dir[:, ::-1].T
    return edge_index


def build_internal_connectivity(internal: pv.DataSet) -> np.ndarray:
    # Triangular connectivity for BaseMFDataset/Poisson-style loaders.
    tri = internal.triangulate()
    faces = np.asarray(tri.faces, dtype=np.int64)
    if faces.size == 0:
        raise RuntimeError("Internal mesh has no faces after triangulation.")
    if faces.size % 4 != 0:
        raise RuntimeError("Unexpected triangulated face buffer shape.")

    arr = faces.reshape(-1, 4)
    if not np.all(arr[:, 0] == 3):
        raise RuntimeError("Non-triangle cells found after triangulate().")

    return arr[:, 1:].astype(np.int32)


def _fidelity_raw_from_index(fidelity_index: int, n_levels: int = 11) -> float:
    return float(fidelity_index) / float(max(n_levels - 1, 1))


def write_h5_sample(out_path: Path, processed: Dict[str, Any], sample_id: str, case_id: str) -> Dict[str, Any]:
    internal = processed["internal"]
    aero = processed["aero"]
    metadata = processed["metadata"]
    coeff_ref = processed["coeff_ref"]

    sim = metadata["simulation_params"]

    coords = np.asarray(internal.points)[:, :2].astype(np.float64)
    U = np.asarray(internal.point_data["U"])[:, :2].astype(np.float64)
    p = np.asarray(internal.point_data["p"]).reshape(-1, 1).astype(np.float64)
    wss = np.asarray(internal.point_data["wallShearStress"])[:, :2].astype(np.float64)
    sdf = -np.asarray(internal.point_data["implicit_distance"]).reshape(-1, 1).astype(np.float64)

    on_surface, aero_to_internal = build_surface_mapping(internal, aero)

    normals = np.zeros((coords.shape[0], 2), dtype=np.float64)
    aero_point_normals = np.asarray(aero.point_data["Normals"])[:, :2].astype(np.float64)
    normals[aero_to_internal] = aero_point_normals

    aoa_rad = np.deg2rad(float(sim["aoa"]))
    u_in = np.tile(
        np.array([[sim["Uinf"] * np.cos(aoa_rad), sim["Uinf"] * np.sin(aoa_rad)]], dtype=np.float64),
        (coords.shape[0], 1),
    )

    # Poisson-style channel metadata + one big fields matrix.
    fields_list = [
        coords,
        u_in,
        sdf,
        normals,
        on_surface[:, None].astype(np.float64),
        U,
        p,
        wss,
    ]
    fields_names = [
        "coords",
        "u_in",
        "sdf",
        "normals",
        "on_surface",
        "u",
        "p",
        "wss",
    ]
    fields = np.concatenate(fields_list, axis=-1)

    line_cells = np.asarray(aero.lines, dtype=np.int64).reshape(-1, 3)
    if not np.all(line_cells[:, 0] == 2):
        raise ValueError("Unexpected aero line-cell format.")
    aero_line_conn = line_cells[:, 1:]
    aero_line_length = np.asarray(aero.cell_data["Length"], dtype=np.float64)
    aero_line_normals = np.asarray(aero.cell_data["Normals"], dtype=np.float64)[:, :2]

    edge_index = build_internal_edge_index(internal)
    tri_conn = build_internal_connectivity(internal)

    fidelity_index = int(metadata["fidelity_index"])
    fidelity_raw = _fidelity_raw_from_index(fidelity_index)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as h5f:
        cond = h5f.create_group("cond")
        cond.create_dataset("fidelity_raw", data=fidelity_raw, dtype=np.float64)

        channels = h5f.create_group("channels")
        curr = 0
        for name, arr in zip(fields_names, fields_list):
            channels.create_dataset(name, data=np.arange(curr, curr + arr.shape[1]), dtype=np.int64)
            curr += arr.shape[1]

        data = h5f.create_group("data")
        data.create_dataset("fields", data=fields, dtype=np.float64)
        data.create_dataset("edge_index", data=edge_index, dtype=np.int64)

        mesh = h5f.create_group("mesh")
        mesh.create_dataset("connectivity", data=tri_conn, dtype=np.int32)
        mesh.create_dataset("aero_line_connectivity", data=aero_line_conn, dtype=np.int64)
        mesh.create_dataset("aero_line_length", data=aero_line_length, dtype=np.float64)
        mesh.create_dataset("aero_line_normals", data=aero_line_normals, dtype=np.float64)
        mesh.create_dataset("aero_to_internal", data=aero_to_internal, dtype=np.int64)

        reference = h5f.create_group("reference")
        reference.create_dataset("cd_openfoam", data=float(coeff_ref["Cd"]), dtype=np.float64)
        reference.create_dataset("cl_openfoam", data=float(coeff_ref["Cl"]), dtype=np.float64)

        h5f.attrs["metadata_json"] = json.dumps(metadata, ensure_ascii=False)
        h5f.attrs["coeff_ref_json"] = json.dumps(coeff_ref, ensure_ascii=False)
        h5f.attrs["vtk_internal_path"] = processed["vtk_internal_path"]

    simulation_time = metadata.get("wall_time_seconds")
    if simulation_time is None:
        simulation_time = metadata.get("solver_clock_time_s")
    if simulation_time is None:
        simulation_time = metadata.get("solver_execution_time_s")

    return {
        "sample_id": sample_id,
        "case_id": case_id,
        "fidelity_id": str(metadata.get("fidelity_id", f"{fidelity_index:02d}")),
        "fidelity_index": fidelity_index,
        "fidelity_raw": fidelity_raw,
        "meshing_fidelity": fidelity_raw,
        "dofs": int(coords.shape[0]),
        "n_nodes": int(coords.shape[0]),
        "simulation_time": float(simulation_time) if simulation_time is not None else np.nan,
        "status": metadata.get("status", "unknown"),
        "reynolds": float(sim.get("reynolds", np.nan)),
        "aoa_deg": float(sim.get("aoa", np.nan)),
        "u_inf": float(sim.get("Uinf", np.nan)),
        "cd_openfoam": float(coeff_ref.get("Cd", np.nan)),
        "cl_openfoam": float(coeff_ref.get("Cl", np.nan)),
        "rel_l2_error": np.nan,
        "rel_residual": np.nan,
    }


def _process_one_fidelity(
    fid_dir_path: Path,
    out_path: Path,
    sample_id: str,
    case_id: str,
) -> Dict[str, Any]:
    processed = preprocess_case(fid_dir_path)
    row = write_h5_sample(out_path, processed, sample_id=sample_id, case_id=case_id)
    return row


def postprocess_dataset(
    dataset_dir: Path,
    out_dir: Path,
    skip_failed: bool = True,
    raise_on_error: bool = False,
    max_workers: int | None = None,
) -> pd.DataFrame:
    case_dirs = sorted([d for d in os.listdir(dataset_dir) if (dataset_dir / d).is_dir()])
    metadata_rows: List[Dict[str, Any]] = []

    sample_rows: List[Dict[str, str]] = []
    jobs: List[tuple[Path, Path, str, str]] = []

    for case_idx, case_dir in tqdm(enumerate(case_dirs), total=len(case_dirs), desc="Queueing cases"):
        case_dir_path = dataset_dir / case_dir
        sample_id = f"{case_idx+1:05d}"
        out_case_dir = out_dir / sample_id
        out_case_dir.mkdir(parents=True, exist_ok=True)
        sample_rows.append({"sample_id": sample_id, "case_id": case_dir})

        fid_dirs = sorted([d for d in os.listdir(case_dir_path) if (case_dir_path / d).is_dir() and d.startswith("fid_")])
        for fid_dir in fid_dirs:
            fid_dir_path = case_dir_path / fid_dir
            metadata_path = fid_dir_path / "metadata.json"
            if not metadata_path.exists():
                print(f"[skip] Missing metadata: {fid_dir_path}")
                continue

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            status = metadata.get("status", "unknown")
            if skip_failed and status != "ok":
                print(f"[skip] status={status}: {fid_dir_path}")
                continue

            out_path = out_case_dir / f"{metadata['fidelity_id']}.h5"
            jobs.append((fid_dir_path, out_path, sample_id, case_dir))

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, int(max_workers))
    print(f"Running postprocessing with ProcessPoolExecutor(max_workers={max_workers}) for {len(jobs)} jobs.")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_one_fidelity,
                fid_dir_path,
                out_path,
                sample_id,
                case_id,
            ): (fid_dir_path, sample_id, case_id)
            for (fid_dir_path, out_path, sample_id, case_id) in jobs
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Processing fidelities"):
            fid_dir_path, _, _ = future_map[future]
            try:
                row = future.result()
                metadata_rows.append(row)
            except Exception as exc:  # pragma: no cover - bulk data conversion guard
                print(f"[error] Failed processing {fid_dir_path}: {exc}")
                if raise_on_error:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_df = metadata_df.sort_values(["sample_id", "fidelity_raw"]).reset_index(drop=True)
        metadata_df.to_csv(out_dir / "metadata_unfiltered.csv", index=False)
    else:
        metadata_df = pd.DataFrame(
            columns=[
                "sample_id",
                "case_id",
                "fidelity_id",
                "fidelity_index",
                "fidelity_raw",
                "meshing_fidelity",
                "dofs",
                "n_nodes",
                "simulation_time",
                "status",
                "reynolds",
                "aoa_deg",
                "u_inf",
                "cd_openfoam",
                "cl_openfoam",
                "rel_l2_error",
                "rel_residual",
            ]
        )
        metadata_df.to_csv(out_dir / "metadata_unfiltered.csv", index=False)

    pd.DataFrame(sample_rows).drop_duplicates().sort_values("sample_id").to_csv(
        out_dir / "sample_info.csv", index=False
    )

    print(f"Wrote metadata table: {out_dir / 'metadata_unfiltered.csv'}")
    return metadata_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess OpenFOAM airfoil runs into Poisson-style H5 dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Input run/raw directory containing case folders with fid_XX subfolders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output processed dataset root.",
    )
    parser.add_argument("--include-failed", action="store_true", help="Also process cases with status != ok.")
    parser.add_argument("--raise-on-error", action="store_true", help="Raise immediately if one case fails.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel postprocessing (default: os.cpu_count()).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    postprocess_dataset(
        dataset_dir=args.dataset_dir,
        out_dir=args.out_dir,
        skip_failed=not args.include_failed,
        raise_on_error=args.raise_on_error,
        max_workers=args.max_workers,
    )
