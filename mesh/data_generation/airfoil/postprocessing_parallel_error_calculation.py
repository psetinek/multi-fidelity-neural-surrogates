import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyvista as pv
from tqdm import tqdm


EPS = 1e-12

ERROR_COLUMNS = [
    "rel_l2_error_ux",
    "rel_l2_error_uy",
    "rel_l2_error_p",
    "rel_l2_error_wss",
    "rel_l2_error",
]


def _latest_file(candidates: Sequence[Path]) -> Optional[Path]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _find_internal_vtu(case_dir: Path) -> Optional[Path]:
    candidates = [p for p in case_dir.rglob("*.vtu") if "internal" in p.name.lower()]
    if not candidates:
        candidates = list(case_dir.rglob("*.vtu"))
    return _latest_file(candidates)


def _as_point_field(mesh: pv.DataSet, field_name: str) -> pv.DataSet:
    if field_name in mesh.point_data:
        return mesh
    if field_name in mesh.cell_data:
        return mesh.cell_data_to_point_data(pass_cell_data=True)
    return mesh


def _get_field(mesh: pv.DataSet, field_name: str) -> np.ndarray:
    if field_name in mesh.point_data:
        arr = np.asarray(mesh.point_data[field_name])
    elif field_name in mesh.cell_data:
        arr = np.asarray(mesh.cell_data[field_name])
    else:
        raise KeyError(f"Field '{field_name}' not found in mesh.")

    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _relative_l2(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denom = np.linalg.norm(a.ravel())
    if denom < EPS:
        return None
    return float(np.linalg.norm((b - a).ravel()) / denom)


def _vector_magnitude(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return np.abs(arr)
    if arr.shape[1] == 1:
        return np.abs(arr[:, 0])
    return np.linalg.norm(arr[:, :2], axis=1)


def _compute_rel_l2_metrics(
    reference_internal: pv.DataSet,
    candidate_internal: pv.DataSet,
) -> Dict[str, Optional[float]]:
    ref = _as_point_field(reference_internal, "U")
    ref = _as_point_field(ref, "p")
    ref = _as_point_field(ref, "wallShearStress")

    cand = _as_point_field(candidate_internal, "U")
    cand = _as_point_field(cand, "p")
    cand = _as_point_field(cand, "wallShearStress")

    sampled = ref.sample(cand)
    valid = np.asarray(
        sampled.point_data.get("vtkValidPointMask", np.ones(ref.n_points, dtype=np.uint8))
    ).astype(bool)
    if valid.sum() == 0:
        return {
            "rel_l2_error_ux": None,
            "rel_l2_error_uy": None,
            "rel_l2_error_p": None,
            "rel_l2_error_wss": None,
            "rel_l2_error": None,
        }

    ref_u = _get_field(ref, "U")[valid, :2]
    cand_u = _get_field(sampled, "U")[valid, :2]
    ref_p = _get_field(ref, "p")[valid]
    cand_p = _get_field(sampled, "p")[valid]

    ref_wss = _get_field(ref, "wallShearStress")[valid]
    cand_wss = _get_field(sampled, "wallShearStress")[valid]

    # Surface points follow the same convention as in postprocessing:
    # wall points have near-zero velocity.
    on_surface = np.isclose(ref_u[:, 0], 0.0, atol=1e-12) & np.isclose(ref_u[:, 1], 0.0, atol=1e-12)
    if not np.any(on_surface):
        # fallback for tiny numerical noise
        on_surface = np.linalg.norm(ref_u, axis=1) < 1e-10

    rel_ux = _relative_l2(ref_u[:, 0], cand_u[:, 0])
    rel_uy = _relative_l2(ref_u[:, 1], cand_u[:, 1])
    rel_p = _relative_l2(ref_p, cand_p)

    rel_wss: Optional[float]
    if np.any(on_surface):
        ref_wss_mag = _vector_magnitude(ref_wss)[on_surface]
        cand_wss_mag = _vector_magnitude(cand_wss)[on_surface]
        rel_wss = _relative_l2(ref_wss_mag, cand_wss_mag)
    else:
        rel_wss = None

    values = [rel_ux, rel_uy, rel_p, rel_wss]
    valid_values = [float(v) for v in values if v is not None and np.isfinite(v)]
    rel_avg = float(np.mean(valid_values)) if valid_values else None

    return {
        "rel_l2_error_ux": rel_ux,
        "rel_l2_error_uy": rel_uy,
        "rel_l2_error_p": rel_p,
        "rel_l2_error_wss": rel_wss,
        "rel_l2_error": rel_avg,
    }


def _normalize_fidelity_id(fid_value: Any) -> Optional[int]:
    try:
        return int(float(fid_value))
    except Exception:
        return None


def _process_case_errors(
    case_id: str,
    case_dir: Path,
    fidelity_ids: List[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not case_dir.exists():
        return rows

    unique_fids = sorted(set(int(fid) for fid in fidelity_ids))
    if not unique_fids:
        return rows

    ref_fid = max(unique_fids)
    ref_case_dir = case_dir / f"fid_{ref_fid:02d}"
    ref_internal_path = _find_internal_vtu(ref_case_dir)
    if ref_internal_path is None:
        for fid in unique_fids:
            rows.append(
                {
                    "case_id": case_id,
                    "__fid_int": fid,
                    "rel_l2_error_ux": np.nan,
                    "rel_l2_error_uy": np.nan,
                    "rel_l2_error_p": np.nan,
                    "rel_l2_error_wss": np.nan,
                    "rel_l2_error": np.nan,
                }
            )
        return rows

    ref_internal = pv.read(ref_internal_path)

    for fid in unique_fids:
        fid_dir = case_dir / f"fid_{fid:02d}"
        internal_path = _find_internal_vtu(fid_dir)
        if internal_path is None:
            rels = {
                "rel_l2_error_ux": None,
                "rel_l2_error_uy": None,
                "rel_l2_error_p": None,
                "rel_l2_error_wss": None,
                "rel_l2_error": None,
            }
        else:
            cand_internal = pv.read(internal_path)
            rels = _compute_rel_l2_metrics(ref_internal, cand_internal)

        rows.append(
            {
                "case_id": case_id,
                "__fid_int": fid,
                "rel_l2_error_ux": rels["rel_l2_error_ux"],
                "rel_l2_error_uy": rels["rel_l2_error_uy"],
                "rel_l2_error_p": rels["rel_l2_error_p"],
                "rel_l2_error_wss": rels["rel_l2_error_wss"],
                "rel_l2_error": rels["rel_l2_error"],
            }
        )

    return rows


def augment_metadata_with_errors(
    dataset_dir: Path,
    metadata_csv: Path,
    out_csv: Optional[Path] = None,
    max_workers: Optional[int] = None,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    # str dtypes: preserve zero-padded ids ("00001", "00") through the rewrite
    df = pd.read_csv(metadata_csv, dtype={"sample_id": str, "fidelity_id": str})
    if "case_id" not in df.columns or "fidelity_id" not in df.columns:
        raise KeyError("metadata.csv must contain 'case_id' and 'fidelity_id' columns.")

    df["case_id"] = df["case_id"].astype(str)
    df["__fid_int"] = df["fidelity_id"].apply(_normalize_fidelity_id)
    if df["__fid_int"].isna().any():
        bad = df[df["__fid_int"].isna()][["case_id", "fidelity_id"]].head(5)
        raise ValueError(f"Could not parse some fidelity_id values to integers. Examples:\n{bad}")
    df["__fid_int"] = df["__fid_int"].astype(int)

    cases = (
        df.groupby("case_id", sort=True)["__fid_int"]
        .apply(lambda s: sorted(set(int(v) for v in s.tolist())))
        .to_dict()
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 1
    max_workers = max(1, int(max_workers))

    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(
                _process_case_errors,
                case_id,
                dataset_dir / case_id,
                fid_list,
            ): case_id
            for case_id, fid_list in cases.items()
        }

        for future in tqdm(as_completed(future_map), total=len(future_map), desc="Computing rel-L2 errors"):
            case_id = future_map[future]
            try:
                rows.extend(future.result())
            except Exception as exc:
                print(f"[error] Failed error-calculation for case '{case_id}': {exc}")
                if raise_on_error:
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise

    err_df = pd.DataFrame(rows)
    if err_df.empty:
        for col in ERROR_COLUMNS:
            df[col] = np.nan
    else:
        err_df = err_df.drop_duplicates(subset=["case_id", "__fid_int"], keep="last")
        for col in ERROR_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
        df = df.merge(err_df, on=["case_id", "__fid_int"], how="left")

    df = df.drop(columns=["__fid_int"])

    out_path = out_csv if out_csv is not None else metadata_csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote augmented metadata: {out_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel rel-L2 error calculation against highest fidelity and metadata.csv augmentation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Input raw run directory containing case folders with fid_XX subfolders.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Existing metadata.csv to augment.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: overwrite --metadata-csv).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--raise-on-error",
        action="store_true",
        help="Raise immediately if one case error-calculation fails.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    augment_metadata_with_errors(
        dataset_dir=args.dataset_dir,
        metadata_csv=args.metadata_csv,
        out_csv=args.out_csv,
        max_workers=args.max_workers,
        raise_on_error=args.raise_on_error,
    )
