#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


NUMERIC_DIR_RE = re.compile(r"^\d+(\.\d+)?$")
END_RE = re.compile(r"(?m)^\s*End\s*$")


@dataclass
class ProbeResult:
    state: str
    has_solver_log: bool
    solver_finished: bool
    has_force_coeffs: bool
    has_time_dirs: bool
    metadata_tmp_exists: bool
    latest_mtime_epoch: float | None


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_log_tail(log_path: Path, max_bytes: int = 131072) -> str:
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(size - max_bytes, 0))
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _solver_log_path(fid_dir: Path) -> Path | None:
    for name in ("log.simpleFoam", "log.rhoSimpleFoam"):
        path = fid_dir / name
        if path.exists():
            return path
    return None


def _force_coeffs_has_data(fid_dir: Path) -> bool:
    coeff_path = fid_dir / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
    if not coeff_path.exists():
        return False
    try:
        with coeff_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                return True
    except Exception:
        return False
    return False


def _has_numeric_time_dirs(fid_dir: Path) -> bool:
    try:
        for child in fid_dir.iterdir():
            if child.is_dir() and NUMERIC_DIR_RE.match(child.name):
                return True
    except Exception:
        return False
    return False


def _latest_mtime(fid_dir: Path) -> float | None:
    latest = None
    try:
        for path in fid_dir.rglob("*"):
            try:
                mt = path.stat().st_mtime
            except Exception:
                continue
            if latest is None or mt > latest:
                latest = mt
    except Exception:
        return None
    return latest


def probe_missing_metadata(fid_dir: Path) -> ProbeResult:
    if not fid_dir.exists():
        return ProbeResult(
            state="missing_dir",
            has_solver_log=False,
            solver_finished=False,
            has_force_coeffs=False,
            has_time_dirs=False,
            metadata_tmp_exists=False,
            latest_mtime_epoch=None,
        )

    log_path = _solver_log_path(fid_dir)
    has_solver_log = log_path is not None
    solver_finished = False
    if log_path is not None:
        tail = _read_log_tail(log_path)
        # "End" marker and/or OpenFOAM finalizing message.
        solver_finished = bool(END_RE.search(tail) or ("Finalising parallel run" in tail))

    has_force_coeffs = _force_coeffs_has_data(fid_dir)
    has_time_dirs = _has_numeric_time_dirs(fid_dir)
    metadata_tmp_exists = (fid_dir / "metadata.json.tmp").exists()
    latest = _latest_mtime(fid_dir)

    if solver_finished and has_force_coeffs:
        state = "missing_metadata_likely_finished"
    elif has_solver_log or has_force_coeffs or has_time_dirs:
        state = "missing_metadata_partial"
    else:
        state = "missing_metadata_no_outputs"

    return ProbeResult(
        state=state,
        has_solver_log=has_solver_log,
        solver_finished=solver_finished,
        has_force_coeffs=has_force_coeffs,
        has_time_dirs=has_time_dirs,
        metadata_tmp_exists=metadata_tmp_exists,
        latest_mtime_epoch=latest,
    )


def analyze_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    plan_path = run_dir / "execution_plan.csv"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing execution plan: {plan_path}")

    plan_df = pd.read_csv(plan_path, dtype={"case_id": str, "fidelity_id": str})
    plan_df["fidelity_id"] = plan_df["fidelity_id"].astype(str).str.zfill(2)
    plan_df = plan_df[["case_id", "fidelity_id"]].drop_duplicates().reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, row in plan_df.iterrows():
        case_id = str(row["case_id"])
        fidelity_id = str(row["fidelity_id"])
        fid_dir = run_dir / "raw" / case_id / f"fid_{fidelity_id}"
        metadata_path = fid_dir / "metadata.json"

        out: dict[str, Any] = {
            "case_id": case_id,
            "fidelity_id": fidelity_id,
            "fid_dir": str(fid_dir),
            "metadata_exists": metadata_path.exists(),
            "metadata_status": None,
            "state": None,
            "error": None,
            "wall_time_seconds": None,
            "has_solver_log": None,
            "solver_finished": None,
            "has_force_coeffs": None,
            "has_time_dirs": None,
            "metadata_tmp_exists": None,
            "latest_mtime_epoch": None,
        }

        if metadata_path.exists():
            meta = _safe_read_json(metadata_path)
            if meta is None:
                out["state"] = "metadata_unreadable"
                out["metadata_status"] = "unreadable"
            else:
                status = str(meta.get("status", "unknown"))
                out["metadata_status"] = status
                out["error"] = meta.get("error")
                out["wall_time_seconds"] = meta.get("wall_time_seconds")
                if status == "ok":
                    out["state"] = "ok"
                elif status == "failed":
                    out["state"] = "failed"
                elif status == "pending":
                    out["state"] = "pending"
                else:
                    out["state"] = f"metadata_{status}"
        else:
            probe = probe_missing_metadata(fid_dir)
            out["state"] = probe.state
            out["has_solver_log"] = probe.has_solver_log
            out["solver_finished"] = probe.solver_finished
            out["has_force_coeffs"] = probe.has_force_coeffs
            out["has_time_dirs"] = probe.has_time_dirs
            out["metadata_tmp_exists"] = probe.metadata_tmp_exists
            out["latest_mtime_epoch"] = probe.latest_mtime_epoch

        rows.append(out)

    result_df = pd.DataFrame(rows)
    state_counts = result_df["state"].value_counts(dropna=False).to_dict()
    total = int(len(result_df))
    n_ok = int(state_counts.get("ok", 0))
    n_incomplete = total - n_ok

    summary = {
        "run_dir": str(run_dir),
        "total_planned": total,
        "completed_ok": n_ok,
        "incomplete": n_incomplete,
        "state_counts": state_counts,
    }
    return result_df, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check which planned airfoil runs did not finish cleanly."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run directory (contains execution_plan.csv and raw/).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Default: <run-dir>/completion_check.csv",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=None,
        help="Optional output summary JSON. Default: <run-dir>/completion_summary.json",
    )
    parser.add_argument(
        "--only-incomplete",
        action="store_true",
        help="Write only non-ok entries to CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    out_csv = args.out_csv or (run_dir / "completion_check.csv")
    out_summary = args.out_summary_json or (run_dir / "completion_summary.json")

    result_df, summary = analyze_run(run_dir)

    write_df = result_df[result_df["state"] != "ok"].copy() if args.only_incomplete else result_df
    write_df.to_csv(out_csv, index=False)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote CSV: {out_csv}")
    print(f"Wrote summary: {out_summary}")


if __name__ == "__main__":
    main()
