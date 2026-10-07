from __future__ import annotations

import contextlib
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime as dt
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
import yaml

HERE = Path(__file__).resolve().parent            # data_generation/airfoil
REPO_ROOT = HERE.parents[1]


EPS = 1e-12


def _log(message: str) -> None:
    print(message, flush=True)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


@contextlib.contextmanager
def _pushd(path: Path) -> Iterable[None]:
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _air_kinematic_viscosity(temperature_k: float) -> float:
    # Same polynomial as the upstream NACA_simulation implementation.
    return (
        -3.400747e-6
        + 3.452139e-8 * temperature_k
        + 1.00881778e-10 * temperature_k**2
        - 1.363528e-14 * temperature_k**3
    )


def _air_density(temperature_k: float) -> float:
    mol = 28.965338e-3
    p_ref = 1.01325e5
    return p_ref * mol / (8.3144621 * temperature_k)


def _estimate_y_plus(u_inf: float, y_h: float, temperature_k: float, chord: float = 1.0) -> float:
    nu = _air_kinematic_viscosity(temperature_k)
    reynolds = max(abs(u_inf) * chord / max(nu, EPS), EPS)
    # Fully turbulent flat-plate correlation (Schlichting).
    cf = 0.026 / (reynolds ** (1.0 / 7.0))
    u_tau = abs(u_inf) * math.sqrt(max(cf, EPS) / 2.0)
    return y_h * u_tau / max(nu, EPS)


def _slugify(value: str) -> str:
    out = value.lower()
    out = re.sub(r"[^a-z0-9]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out


DEFAULT_CONFIG: Dict[str, Any] = {
    # Relative paths are resolved against the repository root. A campaign writes to
    # <output_root>/runs/<run_name>/{raw,analysis}. The vendored code lives next to this file.
    "output_root": "_data_test/airfoil",
    "run_name": None,
    "naca_simulation_dir": str(HERE / "NACA_simulation"),
    "airfrans_lib_dir": str(HERE / "airfrans_lib"),
    "runtime": {
        "stop_on_error": False,
        "skip_existing": True,
        "compute_gradients": False,
        "write_geometry_figures": False,
        "generate_vtk": True,
        "max_parallel": 1,
    },
    "openfoam": {
        "turbulence": "SST",
        "compressible": False,
        "n_proc": 16,
        "temperature": 298.15,
        "domain_L": 200.0,
        "n_iter": {
            "mode": "interpolated",  # "interpolated" or "authors"
            "coarse": 14000,
            "fine": 24000,
            "authors_low_aoa": 20000,
            "authors_high_aoa": 40000,
            "aoa_threshold_deg": 10.0,
        },
    },
    "fidelity": {
        "n_levels": 11,
        "exponent": 1.35,
        "target_lowest_rel_l2": [0.15, 0.20],
        "ranges": {
            "y_h": {"coarse": 8.0e-6, "fine": 2.0e-6},
            "y_hd": {"coarse": 4.0e-4, "fine": 1.0e-4},
            "x_h": {"coarse": 8.0e-5, "fine": 1.0e-5},
            "y_exp": {"coarse": 1.11, "fine": 1.075},
            "x_exp": {"coarse": 1.08, "fine": 1.025},
            "x_expd": {"coarse": 1.11, "fine": 1.075},
        },
    },
    "cases": [
        {
            "name": "naca0012_re3e6_aoa0",
            "reynolds": 3.0e6,
            "aoa_deg": 0.0,
            "digits": [0, 0, 12],
        }
    ],
}


_REQUIRED_RANGE_KEYS = ["y_h", "y_hd", "x_h", "y_exp", "x_exp", "x_expd"]


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        user_cfg = yaml.safe_load(handle) or {}
    cfg = _deep_update(DEFAULT_CONFIG, user_cfg)

    for key in ("output_root", "naca_simulation_dir", "airfrans_lib_dir"):
        if cfg.get(key) and not Path(cfg[key]).is_absolute():
            cfg[key] = str(REPO_ROOT / cfg[key])

    if int(cfg["fidelity"]["n_levels"]) != 11:
        raise ValueError("The pipeline is written for exactly 11 fidelity levels (00 to 10).")

    if not cfg.get("cases"):
        raise ValueError("Config must define at least one simulation case.")

    for key in _REQUIRED_RANGE_KEYS:
        if key not in cfg["fidelity"]["ranges"]:
            raise ValueError(f"Missing fidelity range for '{key}'.")
        coarse = float(cfg["fidelity"]["ranges"][key]["coarse"])
        fine = float(cfg["fidelity"]["ranges"][key]["fine"])
        if coarse <= 0 or fine <= 0:
            raise ValueError(f"Fidelity range values must be > 0 for '{key}'.")

    for case in cfg["cases"]:
        digits = case.get("digits", [])
        if len(digits) not in (3, 4):
            raise ValueError(
                f"Case '{case.get('name', '<unnamed>')}' must use 3 (NACA 4-digit) or 4 (NACA 5-digit) digits entries."
            )

    n_iter_cfg = cfg["openfoam"].get("n_iter", {})
    if isinstance(n_iter_cfg, dict):
        mode = str(n_iter_cfg.get("mode", "interpolated")).lower()
        if mode not in {"interpolated", "authors"}:
            raise ValueError("openfoam.n_iter.mode must be one of: interpolated, authors")
        if mode == "interpolated":
            coarse = float(n_iter_cfg.get("coarse", 0))
            fine = float(n_iter_cfg.get("fine", 0))
            if coarse <= 0 or fine <= 0:
                raise ValueError("openfoam.n_iter.coarse/fine must be > 0 for interpolated mode")
        if mode == "authors":
            low = float(n_iter_cfg.get("authors_low_aoa", 0))
            high = float(n_iter_cfg.get("authors_high_aoa", 0))
            if low <= 0 or high <= 0:
                raise ValueError("openfoam.n_iter.authors_low_aoa/authors_high_aoa must be > 0 for authors mode")
    else:
        if float(n_iter_cfg) <= 0:
            raise ValueError("openfoam.n_iter must be > 0")

    max_parallel = int(cfg.get("runtime", {}).get("max_parallel", 1))
    if max_parallel < 1:
        raise ValueError("runtime.max_parallel must be >= 1")

    return cfg


def _geom_interp(coarse: float, fine: float, t: float) -> float:
    # Geometric interpolation keeps positive quantities smooth over several decades.
    return coarse * (fine / coarse) ** t


def _resolve_n_iter(cfg: Dict[str, Any], aoa_deg: float, fallback_fidelity_n_iter: int) -> int:
    n_iter_cfg = cfg["openfoam"].get("n_iter", {})
    if isinstance(n_iter_cfg, dict):
        mode = str(n_iter_cfg.get("mode", "interpolated")).lower()
        if mode == "authors":
            threshold = float(n_iter_cfg.get("aoa_threshold_deg", 10.0))
            low = int(n_iter_cfg.get("authors_low_aoa", 20000))
            high = int(n_iter_cfg.get("authors_high_aoa", 40000))
            return high if abs(float(aoa_deg)) > threshold else low
    return int(fallback_fidelity_n_iter)


def build_fidelity_table(cfg: Dict[str, Any]) -> pd.DataFrame:
    n_levels = int(cfg["fidelity"]["n_levels"])
    exponent = float(cfg["fidelity"].get("exponent", 1.0))

    n_iter_cfg = cfg["openfoam"].get("n_iter", {})
    if isinstance(n_iter_cfg, dict):
        coarse_iters = float(n_iter_cfg.get("coarse", 20000))
        fine_iters = float(n_iter_cfg.get("fine", coarse_iters))
    else:
        coarse_iters = float(n_iter_cfg)
        fine_iters = float(n_iter_cfg)

    rows: List[Dict[str, Any]] = []
    for idx in range(n_levels):
        scalar = idx / (n_levels - 1)
        shaped = scalar**exponent

        row: Dict[str, Any] = {
            "fidelity_id": f"{idx:02d}",
            "fidelity_index": idx,
            "fidelity_scalar": scalar,
            "fidelity_scalar_shaped": shaped,
        }
        for key in _REQUIRED_RANGE_KEYS:
            coarse = float(cfg["fidelity"]["ranges"][key]["coarse"])
            fine = float(cfg["fidelity"]["ranges"][key]["fine"])
            row[key] = _geom_interp(coarse, fine, shaped)

        row["n_iter"] = int(round(_geom_interp(coarse_iters, fine_iters, shaped)))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _case_id(case: Dict[str, Any]) -> str:
    if case.get("name"):
        return _slugify(str(case["name"]))

    digits = "".join(str(int(x)) for x in case["digits"])
    aoa = str(case["aoa_deg"]).replace("-", "m").replace(".", "p")
    re_val = str(case["reynolds"]).replace("+", "")
    return _slugify(f"naca{digits}_re{re_val}_aoa{aoa}")


def _load_simulation_module(naca_simulation_dir: Path):
    sim_file = naca_simulation_dir / "simulation_generator.py"
    if not sim_file.exists():
        raise FileNotFoundError(f"Missing simulation driver: {sim_file}")

    spec = importlib.util.spec_from_file_location("airfoil_simulation_generator", sim_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load import spec for {sim_file}")

    module = importlib.util.module_from_spec(spec)
    with _pushd(naca_simulation_dir):
        if str(naca_simulation_dir) not in sys.path:
            sys.path.insert(0, str(naca_simulation_dir))
        spec.loader.exec_module(module)
    return module


def _compose_simulation_params(
    cfg: Dict[str, Any],
    case: Dict[str, Any],
    fidelity_row: pd.Series,
) -> Dict[str, Any]:
    temperature = float(cfg["openfoam"]["temperature"])
    nu = _air_kinematic_viscosity(temperature)
    reynolds = float(case["reynolds"])
    u_inf = np.round(reynolds * nu, 3)

    return {
        "L": float(cfg["openfoam"]["domain_L"]),
        "y_h": float(fidelity_row["y_h"]),
        "y_hd": float(fidelity_row["y_hd"]),
        "x_h": float(fidelity_row["x_h"]),
        "y_exp": float(fidelity_row["y_exp"]),
        "x_exp": float(fidelity_row["x_exp"]),
        "x_expd": float(fidelity_row["x_expd"]),
        "turbulence": str(cfg["openfoam"]["turbulence"]),
        "compressible": bool(cfg["openfoam"]["compressible"]),
        "n_proc": int(cfg["openfoam"]["n_proc"]),
        "reynolds": reynolds,
        "temperature": temperature,
        "Uinf": float(u_inf),
        "aoa": float(case["aoa_deg"]),
        "digits": tuple(int(d) for d in case["digits"]),
        "n_iter": _resolve_n_iter(
            cfg=cfg,
            aoa_deg=float(case["aoa_deg"]),
            fallback_fidelity_n_iter=int(fidelity_row["n_iter"]),
        ),
    }


def _parse_checkmesh_cells(case_dir: Path) -> Optional[int]:
    log_path = case_dir / "log.checkMesh"
    if not log_path.exists():
        return None

    pattern = re.compile(r"\bcells:\s+([0-9]+)")
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None


def _read_final_coefficients(case_dir: Path) -> Tuple[Optional[float], Optional[float]]:
    coeff_path = case_dir / "postProcessing" / "forceCoeffs1" / "0" / "coefficient.dat"
    if not coeff_path.exists():
        return None, None

    try:
        data = np.loadtxt(coeff_path, comments="#")
        with coeff_path.open("r", encoding="utf-8", errors="ignore") as handle:
            header_lines = [ln for ln in handle if ln.startswith("# Time")]
    except Exception:
        return None, None

    if data.ndim == 1:
        data = data[None, :]
    last = data[-1]

    # Column order differs between OpenFOAM versions (v2506: Time Cd Cd(f) Cd(r) Cl Cl(f) Cl(r) ...),
    # so resolve the columns by name from the last header line.
    header_cols = header_lines[-1].replace("#", "").split() if header_lines else []
    if header_cols and len(header_cols) == last.shape[0] and "Cd" in header_cols and "Cl" in header_cols:
        return float(last[header_cols.index("Cd")]), float(last[header_cols.index("Cl")])
    cd = float(last[1]) if last.shape[0] > 1 else None
    cl = float(last[4]) if last.shape[0] > 4 else None
    return cd, cl


def _parse_solver_log_metrics(case_dir: Path) -> Dict[str, Optional[float]]:
    log_candidates = [case_dir / "log.simpleFoam", case_dir / "log.rhoSimpleFoam"]
    log_path = next((p for p in log_candidates if p.exists()), None)
    if log_path is None:
        return {
            "solver_execution_time_s": None,
            "solver_clock_time_s": None,
            "yplus_min": None,
            "yplus_max": None,
            "yplus_avg": None,
        }

    exec_re = re.compile(
        r"ExecutionTime\s*=\s*([0-9eE+\-.]+)\s*s\s+ClockTime\s*=\s*([0-9eE+\-.]+)\s*s"
    )
    yplus_re = re.compile(
        r"patch\s+aerofoil\s+y\+\s*:\s*min\s*=\s*([0-9eE+\-.]+),\s*max\s*=\s*([0-9eE+\-.]+),\s*average\s*=\s*([0-9eE+\-.]+)",
        re.IGNORECASE,
    )

    exec_time: Optional[float] = None
    clock_time: Optional[float] = None
    yplus_min: Optional[float] = None
    yplus_max: Optional[float] = None
    yplus_avg: Optional[float] = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            m_exec = exec_re.search(line)
            if m_exec:
                exec_time = float(m_exec.group(1))
                clock_time = float(m_exec.group(2))

            m_yplus = yplus_re.search(line)
            if m_yplus:
                yplus_min = float(m_yplus.group(1))
                yplus_max = float(m_yplus.group(2))
                yplus_avg = float(m_yplus.group(3))

    return {
        "solver_execution_time_s": exec_time,
        "solver_clock_time_s": clock_time,
        "yplus_min": yplus_min,
        "yplus_max": yplus_max,
        "yplus_avg": yplus_avg,
    }


_SOLVER_FAILURE_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"FOAM FATAL", re.IGNORECASE),
    re.compile(r"BAD TERMINATION", re.IGNORECASE),
    re.compile(r"Process received signal", re.IGNORECASE),
    re.compile(r"YOUR APPLICATION TERMINATED", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
    re.compile(r"Floating point exception(?! trapping enabled)", re.IGNORECASE),
)


def _solver_log_path(case_dir: Path) -> Optional[Path]:
    candidates = [case_dir / "log.simpleFoam", case_dir / "log.rhoSimpleFoam"]
    return next((p for p in candidates if p.exists()), None)


def _parse_solver_last_time(case_dir: Path) -> Optional[int]:
    log_path = _solver_log_path(case_dir)
    if log_path is None:
        return None

    last_time: Optional[int] = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("Time = "):
                try:
                    last_time = int(float(line.split("=", 1)[1].strip()))
                except Exception:
                    continue
    return last_time


def _scan_solver_log_failure(case_dir: Path) -> Optional[str]:
    log_path = _solver_log_path(case_dir)
    if log_path is None:
        return "missing solver log (log.simpleFoam/log.rhoSimpleFoam)"

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for idx, line in enumerate(handle, start=1):
            for pattern in _SOLVER_FAILURE_PATTERNS:
                if pattern.search(line):
                    snippet = line.strip()
                    return (
                        f"{pattern.pattern} at {log_path.name}:{idx}"
                        + (f" -> {snippet[:180]}" if snippet else "")
                    )
    return None


def _validate_fidelity_result(case_dir: Path, expected_n_iter: int, just_init: bool) -> Optional[str]:
    if just_init:
        return None

    fatal = _scan_solver_log_failure(case_dir)
    if fatal is not None:
        return f"solver failure detected: {fatal}"

    last_time = _parse_solver_last_time(case_dir)
    if last_time is None:
        return "solver log has no 'Time = ...' entries"
    if expected_n_iter > 0 and last_time < expected_n_iter:
        return f"solver stopped early at Time={last_time} (expected >= {expected_n_iter})"

    cd, cl = _read_final_coefficients(case_dir)
    if cd is None or cl is None:
        return "missing Cd/Cl in force coefficients output"

    solver_metrics = _parse_solver_log_metrics(case_dir)
    if solver_metrics.get("solver_execution_time_s") is None:
        return "missing ExecutionTime/ClockTime in solver log"

    return None


def _latest_file(candidates: Sequence[Path]) -> Optional[Path]:
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _find_internal_vtu(case_dir: Path) -> Optional[Path]:
    candidates = [p for p in case_dir.rglob("*.vtu") if "internal" in p.name.lower()]
    if not candidates:
        candidates = list(case_dir.rglob("*.vtu"))
    return _latest_file(candidates)


def _find_airfoil_vtp(case_dir: Path) -> Optional[Path]:
    candidates = [
        p
        for p in case_dir.rglob("*.vtp")
        if "aerofoil" in p.name.lower() or "airfoil" in p.name.lower()
    ]
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
        raise KeyError(f"Field '{field_name}' not found in mesh data.")

    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _relative_l2(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    denom = np.linalg.norm(a.ravel())
    if denom < EPS:
        return None
    return float(np.linalg.norm((b - a).ravel()) / denom)


def _compute_field_rel_l2(
    reference_internal: pv.DataSet,
    candidate_internal: pv.DataSet,
) -> Dict[str, Optional[float]]:
    ref = _as_point_field(reference_internal, "U")
    ref = _as_point_field(ref, "p")
    cand = _as_point_field(candidate_internal, "U")
    cand = _as_point_field(cand, "p")

    sampled = ref.sample(cand)
    valid = np.asarray(sampled.point_data.get("vtkValidPointMask", np.ones(ref.n_points, dtype=np.uint8))).astype(bool)
    if valid.sum() == 0:
        return {"rel_l2_ux": None, "rel_l2_uy": None, "rel_l2_p": None}

    ref_u = _get_field(ref, "U")[valid, :2]
    cand_u = _get_field(sampled, "U")[valid, :2]
    ref_p = _get_field(ref, "p")[valid]
    cand_p = _get_field(sampled, "p")[valid]

    return {
        "rel_l2_ux": _relative_l2(ref_u[:, 0], cand_u[:, 0]),
        "rel_l2_uy": _relative_l2(ref_u[:, 1], cand_u[:, 1]),
        "rel_l2_p": _relative_l2(ref_p, cand_p),
    }


def _extract_cp_cf(airfoil_mesh: pv.DataSet, u_inf: float, temperature_k: float) -> pd.DataFrame:
    if not isinstance(airfoil_mesh, pv.PolyData):
        surf = airfoil_mesh.extract_surface()
    else:
        surf = airfoil_mesh

    centers = surf.cell_centers()
    centers = _as_point_field(centers, "p")
    centers = _as_point_field(centers, "wallShearStress")

    p = _get_field(centers, "p")
    wss = _get_field(centers, "wallShearStress")
    if wss.ndim == 1:
        wss_mag = np.abs(wss)
    else:
        wss_mag = np.linalg.norm(wss[:, :2], axis=1)

    q_inf = 0.5 * max(float(u_inf) ** 2, EPS)
    rho = _air_density(float(temperature_k))

    df = pd.DataFrame(
        {
            "x": centers.points[:, 0],
            "y": centers.points[:, 1],
            "cp": p / q_inf,
            "cf": wss_mag / max(q_inf * rho, EPS),
        }
    )
    df["side"] = np.where(df["y"] >= 0.0, "upper", "lower")
    df.sort_values(["side", "x"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _plot_cp_cf(cp_cf: pd.DataFrame, out_path: Path) -> None:
    if cp_cf.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fidelities = sorted(cp_cf["fidelity_id"].unique())
    cmap = plt.get_cmap("viridis", len(fidelities))

    for idx, fid in enumerate(fidelities):
        subset = cp_cf[cp_cf["fidelity_id"] == fid]
        upper = subset[subset["side"] == "upper"].sort_values("x")
        lower = subset[subset["side"] == "lower"].sort_values("x")
        color = cmap(idx)

        if not upper.empty:
            axes[0].plot(upper["x"], upper["cp"], color=color, linewidth=1.3, label=f"{fid}")
            axes[1].plot(upper["x"], upper["cf"], color=color, linewidth=1.3, label=f"{fid}")
        if not lower.empty:
            axes[0].plot(lower["x"], lower["cp"], color=color, linestyle="--", linewidth=1.0)
            axes[1].plot(lower["x"], lower["cf"], color=color, linestyle="--", linewidth=1.0)

    axes[0].invert_yaxis()
    axes[0].set_title("Cp Distribution Along Chord")
    axes[1].set_title("Cf Distribution Along Chord")
    for axis in axes:
        axis.set_xlabel("x/c")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cp")
    axes[1].set_ylabel("Cf")
    axes[1].legend(title="Fidelity", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_convergence_metrics(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    exclude_fidelity_id: Optional[str] = None,
) -> None:
    if df.empty:
        return

    plot_df = df
    if exclude_fidelity_id is not None and "fidelity_id" in plot_df.columns:
        plot_df = plot_df[plot_df["fidelity_id"].astype(str) != str(exclude_fidelity_id)]
    if plot_df.empty:
        return

    x = plot_df["fidelity_index"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(x, plot_df["cd"], marker="o", label="Cd")
    axes[0].plot(x, plot_df["cl"], marker="s", label="Cl")
    axes[0].set_title("Coefficient Convergence")
    axes[0].set_xlabel("Fidelity Index")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x, plot_df["rel_l2_ux"], marker="o", label="rel L2 Ux")
    axes[1].plot(x, plot_df["rel_l2_uy"], marker="s", label="rel L2 Uy")
    axes[1].plot(x, plot_df["rel_l2_p"], marker="^", label="rel L2 p")
    axes[1].set_title("Field Errors vs. Highest Fidelity")
    axes[1].set_xlabel("Fidelity Index")
    axes[1].set_ylabel("Relative L2")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(x, plot_df["n_cells"], marker="o")
    axes[2].set_title("Mesh Cells")
    axes[2].set_xlabel("Fidelity Index")
    axes[2].set_ylabel("Cell Count")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_error_vs_resources(
    df: pd.DataFrame,
    out_path: Path,
    time_col: Optional[str],
    title: str,
    exclude_fidelity_id: Optional[str] = None,
) -> None:
    if df.empty:
        return

    plot_df = df
    if exclude_fidelity_id is not None and "fidelity_id" in plot_df.columns:
        plot_df = plot_df[plot_df["fidelity_id"].astype(str) != str(exclude_fidelity_id)]
    if plot_df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Errors vs cells
    by_cells = plot_df.dropna(subset=["n_cells"]).sort_values("n_cells")
    if not by_cells.empty:
        x_cells = by_cells["n_cells"].to_numpy()
        rel_l2_ux = np.clip(by_cells["rel_l2_ux"].to_numpy(dtype=float), EPS, None)
        rel_l2_uy = np.clip(by_cells["rel_l2_uy"].to_numpy(dtype=float), EPS, None)
        rel_l2_p = np.clip(by_cells["rel_l2_p"].to_numpy(dtype=float), EPS, None)
        rel_l2_fields = np.clip(
            by_cells[["rel_l2_ux", "rel_l2_uy", "rel_l2_p"]].mean(axis=1).to_numpy(dtype=float),
            EPS,
            None,
        )
        rel_err_cd = np.clip(by_cells["rel_err_cd"].to_numpy(dtype=float), EPS, None)
        rel_err_cl = np.clip(by_cells["rel_err_cl"].to_numpy(dtype=float), EPS, None)
        axes[0, 0].plot(x_cells, rel_l2_ux, marker="o", label="rel L2 Ux")
        axes[0, 0].plot(x_cells, rel_l2_uy, marker="s", label="rel L2 Uy")
        axes[0, 0].plot(x_cells, rel_l2_p, marker="^", label="rel L2 p")
        axes[0, 0].plot(x_cells, rel_l2_fields, marker="d", linestyle="--", label="mean rel L2")
        axes[0, 0].set_xscale("log")
        axes[0, 0].set_yscale("log")
        axes[0, 0].set_xlabel("n_cells")
        axes[0, 0].set_ylabel("Relative L2")
        axes[0, 0].set_title("Field Errors vs n_cells")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        axes[0, 1].plot(x_cells, rel_err_cd, marker="o", label="|Cd-Cd_ref|/|Cd_ref|")
        axes[0, 1].plot(x_cells, rel_err_cl, marker="s", label="|Cl-Cl_ref|/|Cl_ref|")
        axes[0, 1].set_xscale("log")
        axes[0, 1].set_yscale("log")
        axes[0, 1].set_xlabel("n_cells")
        axes[0, 1].set_ylabel("Relative Error")
        axes[0, 1].set_title("Coefficient Errors vs n_cells")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "No n_cells data", ha="center", va="center")
        axes[0, 1].text(0.5, 0.5, "No n_cells data", ha="center", va="center")

    # Errors vs time
    if time_col and time_col in plot_df.columns:
        by_time = plot_df.dropna(subset=[time_col]).sort_values(time_col)
    else:
        by_time = pd.DataFrame()

    if not by_time.empty:
        x_time = np.clip(by_time[time_col].to_numpy(dtype=float), EPS, None)
        rel_l2_ux = np.clip(by_time["rel_l2_ux"].to_numpy(dtype=float), EPS, None)
        rel_l2_uy = np.clip(by_time["rel_l2_uy"].to_numpy(dtype=float), EPS, None)
        rel_l2_p = np.clip(by_time["rel_l2_p"].to_numpy(dtype=float), EPS, None)
        rel_l2_fields = np.clip(
            by_time[["rel_l2_ux", "rel_l2_uy", "rel_l2_p"]].mean(axis=1).to_numpy(dtype=float),
            EPS,
            None,
        )
        rel_err_cd = np.clip(by_time["rel_err_cd"].to_numpy(dtype=float), EPS, None)
        rel_err_cl = np.clip(by_time["rel_err_cl"].to_numpy(dtype=float), EPS, None)
        axes[1, 0].plot(x_time, rel_l2_ux, marker="o", label="rel L2 Ux")
        axes[1, 0].plot(x_time, rel_l2_uy, marker="s", label="rel L2 Uy")
        axes[1, 0].plot(x_time, rel_l2_p, marker="^", label="rel L2 p")
        axes[1, 0].plot(x_time, rel_l2_fields, marker="d", linestyle="--", label="mean rel L2")
        axes[1, 0].set_xscale("log")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlabel(time_col)
        axes[1, 0].set_ylabel("Relative L2")
        axes[1, 0].set_title("Field Errors vs Simulation Time")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()

        axes[1, 1].plot(x_time, rel_err_cd, marker="o", label="|Cd-Cd_ref|/|Cd_ref|")
        axes[1, 1].plot(x_time, rel_err_cl, marker="s", label="|Cl-Cl_ref|/|Cl_ref|")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlabel(time_col)
        axes[1, 1].set_ylabel("Relative Error")
        axes[1, 1].set_title("Coefficient Errors vs Simulation Time")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 0].text(0.5, 0.5, "No simulation-time data", ha="center", va="center")
        axes[1, 1].text(0.5, 0.5, "No simulation-time data", ha="center", va="center")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def analyze_case(
    case_dir: Path,
    analysis_dir: Path,
    case: Dict[str, Any],
    fidelity_table: pd.DataFrame,
) -> pd.DataFrame:
    analysis_dir.mkdir(parents=True, exist_ok=True)

    expected_fids = [str(fid) for fid in fidelity_table["fidelity_id"].tolist()]
    ref_fid = expected_fids[-1]
    ref_case_dir = case_dir / f"fid_{ref_fid}"
    ref_internal_path = _find_internal_vtu(ref_case_dir)

    if ref_internal_path is None:
        raise RuntimeError(
            f"Missing reference VTK for highest fidelity '{ref_fid}'. Expected under {ref_case_dir}"
        )

    ref_internal = pv.read(ref_internal_path)

    rows: List[Dict[str, Any]] = []
    cp_cf_all: List[pd.DataFrame] = []

    for _, fid_row in fidelity_table.iterrows():
        fid = str(fid_row["fidelity_id"])
        fidelity_case_dir = case_dir / f"fid_{fid}"

        params_path = fidelity_case_dir / "metadata.json"
        metadata: Dict[str, Any] = {}
        if params_path.exists():
            metadata = json.loads(params_path.read_text(encoding="utf-8"))

        cd, cl = _read_final_coefficients(fidelity_case_dir)
        n_cells = _parse_checkmesh_cells(fidelity_case_dir)

        rels = {"rel_l2_ux": None, "rel_l2_uy": None, "rel_l2_p": None}
        internal_path = _find_internal_vtu(fidelity_case_dir)
        if internal_path is not None:
            internal_mesh = pv.read(internal_path)
            rels = _compute_field_rel_l2(ref_internal, internal_mesh)

        airfoil_path = _find_airfoil_vtp(fidelity_case_dir)
        if airfoil_path is not None:
            try:
                airfoil_mesh = pv.read(airfoil_path)
                cp_cf = _extract_cp_cf(
                    airfoil_mesh=airfoil_mesh,
                    u_inf=float(metadata.get("simulation_params", {}).get("Uinf", 0.0)),
                    temperature_k=float(metadata.get("simulation_params", {}).get("temperature", 298.15)),
                )
                cp_cf["fidelity_id"] = fid
                cp_cf_all.append(cp_cf)
            except Exception:
                _log(f"[WARN] Failed cp/cf extraction for {fidelity_case_dir}")

        rows.append(
            {
                "case_id": _case_id(case),
                "fidelity_id": fid,
                "fidelity_index": int(fid_row["fidelity_index"]),
                "fidelity_scalar": float(fid_row["fidelity_scalar"]),
                "n_cells": n_cells,
                "cd": _safe_float(cd),
                "cl": _safe_float(cl),
                "rel_l2_ux": _safe_float(rels["rel_l2_ux"]),
                "rel_l2_uy": _safe_float(rels["rel_l2_uy"]),
                "rel_l2_p": _safe_float(rels["rel_l2_p"]),
                "yplus_estimate": _safe_float(metadata.get("yplus_estimate")),
                "yplus_min": _safe_float(metadata.get("yplus_min")),
                "yplus_max": _safe_float(metadata.get("yplus_max")),
                "yplus_avg": _safe_float(metadata.get("yplus_avg")),
                "wall_time_seconds": _safe_float(metadata.get("wall_time_seconds")),
                "solver_execution_time_s": _safe_float(metadata.get("solver_execution_time_s")),
                "solver_clock_time_s": _safe_float(metadata.get("solver_clock_time_s")),
                "status": metadata.get("status", "unknown"),
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values("fidelity_index", inplace=True)

    ref_row = df[df["fidelity_id"] == ref_fid]
    cd_ref = ref_row["cd"].iloc[0] if not ref_row.empty else None
    cl_ref = ref_row["cl"].iloc[0] if not ref_row.empty else None

    if cd_ref is not None and not pd.isna(cd_ref) and abs(cd_ref) > EPS:
        df["rel_err_cd"] = (df["cd"] - cd_ref).abs() / abs(cd_ref)
    else:
        df["rel_err_cd"] = np.nan

    if cl_ref is not None and not pd.isna(cl_ref) and abs(cl_ref) > EPS:
        df["rel_err_cl"] = (df["cl"] - cl_ref).abs() / abs(cl_ref)
    else:
        df["rel_err_cl"] = np.nan

    df.to_csv(analysis_dir / "convergence_metrics.csv", index=False)
    _plot_convergence_metrics(
        df,
        analysis_dir / "convergence_plots.png",
        title=f"Convergence: {_case_id(case)}",
        exclude_fidelity_id=ref_fid,
    )
    time_col: Optional[str] = None
    if df["solver_clock_time_s"].notna().any():
        time_col = "solver_clock_time_s"
    elif df["solver_execution_time_s"].notna().any():
        time_col = "solver_execution_time_s"
    elif df["wall_time_seconds"].notna().any():
        time_col = "wall_time_seconds"
    _plot_error_vs_resources(
        df,
        analysis_dir / "error_vs_resources.png",
        time_col=time_col,
        title=f"Errors vs Resources: {_case_id(case)}",
        exclude_fidelity_id=ref_fid,
    )

    if cp_cf_all:
        cp_cf_all_df = pd.concat(cp_cf_all, ignore_index=True)
        cp_cf_all_df.to_csv(analysis_dir / "cp_cf_distribution.csv", index=False)
        _plot_cp_cf(cp_cf_all_df, analysis_dir / "cp_cf_distribution.png")

    return df


def aggregate_analysis(run_dir: Path) -> Optional[pd.DataFrame]:
    metrics_files = list((run_dir / "analysis").glob("*/convergence_metrics.csv"))
    if not metrics_files:
        return None

    frames = [pd.read_csv(path) for path in metrics_files]
    all_df = pd.concat(frames, ignore_index=True)
    all_df["fidelity_index"] = all_df["fidelity_index"].astype(int)
    all_df["fidelity_id"] = all_df["fidelity_index"].map(lambda idx: f"{int(idx):02d}")

    grouped = (
        all_df.groupby(["fidelity_id", "fidelity_index"], as_index=False)
        .agg(
            n_cases=("case_id", "nunique"),
            mean_rel_l2_ux=("rel_l2_ux", "mean"),
            mean_rel_l2_uy=("rel_l2_uy", "mean"),
            mean_rel_l2_p=("rel_l2_p", "mean"),
            mean_rel_err_cd=("rel_err_cd", "mean"),
            mean_rel_err_cl=("rel_err_cl", "mean"),
            mean_cd=("cd", "mean"),
            mean_cl=("cl", "mean"),
            mean_cells=("n_cells", "mean"),
            mean_wall_time_seconds=("wall_time_seconds", "mean"),
            mean_solver_execution_time_s=("solver_execution_time_s", "mean"),
            mean_solver_clock_time_s=("solver_clock_time_s", "mean"),
        )
        .sort_values("fidelity_index")
    )

    grouped.to_csv(run_dir / "analysis" / "convergence_summary.csv", index=False)
    return grouped


def _build_plan_rows(cases: List[Dict[str, Any]], fidelity_table: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for case in cases:
        temperature = float(cfg["openfoam"]["temperature"])
        nu = _air_kinematic_viscosity(temperature)
        u_inf = float(case["reynolds"]) * nu
        for _, frow in fidelity_table.iterrows():
            rows.append(
                {
                    "case_id": _case_id(case),
                    "fidelity_id": str(frow["fidelity_id"]),
                    "fidelity_index": int(frow["fidelity_index"]),
                    "reynolds": float(case["reynolds"]),
                    "aoa_deg": float(case["aoa_deg"]),
                    "digits": "_".join(str(int(d)) for d in case["digits"]),
                    "Uinf": float(u_inf),
                    "y_h": float(frow["y_h"]),
                    "y_hd": float(frow["y_hd"]),
                    "x_h": float(frow["x_h"]),
                    "y_exp": float(frow["y_exp"]),
                    "x_exp": float(frow["x_exp"]),
                    "x_expd": float(frow["x_expd"]),
                    "n_iter": int(frow["n_iter"]),
                    "n_iter_effective": _resolve_n_iter(
                        cfg=cfg,
                        aoa_deg=float(case["aoa_deg"]),
                        fallback_fidelity_n_iter=int(frow["n_iter"]),
                    ),
                    "yplus_estimate": _estimate_y_plus(
                        u_inf=u_inf,
                        y_h=float(frow["y_h"]),
                        temperature_k=temperature,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _run_fidelity_worker(task: Dict[str, Any]) -> Dict[str, Any]:
    naca_simulation_dir = Path(task["naca_simulation_dir"]).resolve()
    init_path = Path(task["init_path"]).resolve()
    fid_dir = Path(task["fid_dir"]).resolve()
    sim_params = task["simulation_params"]

    entry = copy.deepcopy(task["entry"])
    t0 = time.perf_counter()
    try:
        simulation_module = _load_simulation_module(naca_simulation_dir)
        simulation_module.simulation(
            init_path=f"{init_path}/",
            path=f"{fid_dir}/",
            params=sim_params,
            just_init=bool(task["just_init"]),
            figure=bool(task["write_geometry_figures"]),
            compute_grad=bool(task["compute_gradients"]),
            VTK=bool(task["generate_vtk"] and (not task["just_init"])),
        )
        entry["status"] = "ok"
    except Exception as exc:
        entry["status"] = "failed"
        entry["error"] = str(exc)
        entry["traceback"] = traceback.format_exc(limit=20)

    entry["wall_time_seconds"] = float(time.perf_counter() - t0)
    entry["cd"], entry["cl"] = _read_final_coefficients(fid_dir)
    entry["cd"] = _safe_float(entry["cd"])
    entry["cl"] = _safe_float(entry["cl"])
    entry["n_cells"] = _parse_checkmesh_cells(fid_dir)
    entry["solver_last_time"] = _parse_solver_last_time(fid_dir)
    entry.update(_parse_solver_log_metrics(fid_dir))

    # Verify the solver artifacts before accepting the run as complete
    # classifying the fidelity as successful.
    if entry.get("status") == "ok":
        validation_error = _validate_fidelity_result(
            case_dir=fid_dir,
            expected_n_iter=int(sim_params.get("n_iter", 0)),
            just_init=bool(task["just_init"]),
        )
        if validation_error is not None:
            entry["status"] = "failed"
            entry["error"] = validation_error

    # Persist immediately in the worker so results survive controller interruptions.
    _write_json_atomic(fid_dir / "metadata.json", entry)
    return entry


def run_pipeline(
    config_path: Path,
    dry_run: bool = False,
    just_init: bool = False,
    skip_analysis: bool = False,
    run_name_override: Optional[str] = None,
    case_limit: Optional[int] = None,
    fidelity_ids: Optional[Sequence[str]] = None,
    overwrite_run: bool = False,
    max_parallel_override: Optional[int] = None,
) -> Path:
    cfg = load_config(config_path)

    run_name = run_name_override or cfg.get("run_name")
    if not run_name:
        run_name = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    runtime_cfg = cfg["runtime"]
    skip_existing_cfg = bool(runtime_cfg.get("skip_existing", True))

    run_dir = Path(cfg["output_root"]) / "runs" / run_name
    if run_dir.exists() and not overwrite_run:
        if skip_existing_cfg:
            _log(
                f"Resuming existing run directory: {run_dir} "
                + "(skip_existing=true)"
            )
        else:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                + "Set runtime.skip_existing=true to resume or use --overwrite-run to replace it."
            )
    if run_dir.exists() and overwrite_run:
        _log(f"Removing existing run directory: {run_dir}")
        import shutil

        shutil.rmtree(run_dir)

    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    (run_dir / "analysis").mkdir(parents=True, exist_ok=True)

    fidelity_table = build_fidelity_table(cfg)
    if fidelity_ids:
        fidelity_ids_set = set(str(fid).zfill(2) for fid in fidelity_ids)
        fidelity_table = fidelity_table[fidelity_table["fidelity_id"].isin(fidelity_ids_set)].copy()
        fidelity_table.sort_values("fidelity_index", inplace=True)
        fidelity_table.reset_index(drop=True, inplace=True)

    cases = cfg["cases"]
    if case_limit is not None:
        cases = cases[:case_limit]

    fidelity_table.to_csv(run_dir / "fidelity_table.csv", index=False)
    plan_df = _build_plan_rows(cases=cases, fidelity_table=fidelity_table, cfg=cfg)
    plan_df.to_csv(run_dir / "execution_plan.csv", index=False)

    if dry_run:
        _log("Dry-run completed. No simulations executed.")
        return run_dir

    naca_simulation_dir = Path(cfg["naca_simulation_dir"]).resolve()

    stop_on_error = bool(runtime_cfg.get("stop_on_error", False))
    skip_existing = bool(runtime_cfg.get("skip_existing", True))
    compute_gradients = bool(runtime_cfg.get("compute_gradients", False))
    write_geometry_figures = bool(runtime_cfg.get("write_geometry_figures", False))
    generate_vtk = bool(runtime_cfg.get("generate_vtk", True))
    max_parallel = int(max_parallel_override if max_parallel_override is not None else runtime_cfg.get("max_parallel", 1))
    if max_parallel < 1:
        raise ValueError("max_parallel must be >= 1")
    _log(f"Using max_parallel={max_parallel} fidelity workers")

    init_path = naca_simulation_dir / "Simulations" / "airFoil2DInit"
    if not init_path.exists():
        raise FileNotFoundError(f"Could not find OpenFOAM init template: {init_path}")

    run_metadata: Dict[str, Any] = {
        "config_path": str(config_path),
        "run_name": run_name,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dry_run": dry_run,
        "just_init": just_init,
        "cases": [],
    }

    # Queue all (case, fidelity) runs globally so workers are always kept busy,
    # instead of draining one case at a time.
    pending_tasks: List[Dict[str, Any]] = []
    case_fidelity_entries_by_case: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for case in cases:
        case_id = _case_id(case)
        if case_id in case_fidelity_entries_by_case:
            raise ValueError(f"Duplicate case_id '{case_id}' in configuration; case names must be unique.")

        case_dir = run_dir / "raw" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[CASE] {case_id}")

        case_entry: Dict[str, Any] = {
            "case_id": case_id,
            "reynolds": float(case["reynolds"]),
            "aoa_deg": float(case["aoa_deg"]),
            "digits": [int(d) for d in case["digits"]],
            "fidelities": [],
        }
        run_metadata["cases"].append(case_entry)
        case_fidelity_entries_by_case[case_id] = {}

        for _, fidelity_row in fidelity_table.iterrows():
            fid = str(fidelity_row["fidelity_id"])
            fid_dir = case_dir / f"fid_{fid}"
            metadata_path = fid_dir / "metadata.json"

            sim_params = _compose_simulation_params(cfg, case, fidelity_row)
            yplus = _estimate_y_plus(
                u_inf=float(sim_params["Uinf"]),
                y_h=float(sim_params["y_h"]),
                temperature_k=float(sim_params["temperature"]),
            )

            entry = {
                "fidelity_id": fid,
                "fidelity_index": int(fidelity_row["fidelity_index"]),
                "status": "pending",
                "yplus_estimate": float(yplus),
                "simulation_params": sim_params,
                "error": None,
            }

            if skip_existing and metadata_path.exists():
                old = json.loads(metadata_path.read_text(encoding="utf-8"))
                if old.get("status") == "ok":
                    _log(f"  [SKIP] fid_{fid} already completed")
                    case_fidelity_entries_by_case[case_id][fid] = old
                    continue

            _log(
                "  [QUEUE] "
                + f"fid_{fid} (y_h={sim_params['y_h']:.2e}, x_h={sim_params['x_h']:.2e}, "
                + f"n_iter={sim_params['n_iter']}, n_proc={sim_params['n_proc']}, y+~{yplus:.3f})"
            )
            pending_tasks.append(
                {
                    "case_id": case_id,
                    "naca_simulation_dir": str(naca_simulation_dir),
                    "init_path": str(init_path),
                    "fid_dir": str(fid_dir),
                    "simulation_params": sim_params,
                    "just_init": bool(just_init),
                    "write_geometry_figures": bool(write_geometry_figures),
                    "compute_gradients": bool(compute_gradients),
                    "generate_vtk": bool(generate_vtk),
                    "entry": entry,
                }
            )

    def _store_case_entry(case_id_local: str, _entry: Dict[str, Any]) -> None:
        fid_local = str(_entry["fidelity_id"])
        fid_local_dir = run_dir / "raw" / case_id_local / f"fid_{fid_local}"
        fid_local_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(fid_local_dir / "metadata.json", _entry)
        case_fidelity_entries_by_case[case_id_local][fid_local] = _entry

    _log(f"Dispatching {len(pending_tasks)} total runs across {max_parallel} workers")
    has_failure = False
    if max_parallel == 1:
        for task in pending_tasks:
            case_id_local = str(task["case_id"])
            fid = str(task["entry"]["fidelity_id"])
            entry = _run_fidelity_worker(task)
            _store_case_entry(case_id_local, entry)
            if entry.get("status") == "ok":
                _log(
                    f"  [DONE] {case_id_local}/fid_{fid} "
                    + f"(wall={entry.get('wall_time_seconds', 0):.1f}s, cells={entry.get('n_cells')})"
                )
            else:
                has_failure = True
                _log(f"  [FAIL] {case_id_local}/fid_{fid}: {entry.get('error')}")
                if stop_on_error:
                    raise RuntimeError(
                        f"Stopping due to failure in {case_id_local}/fid_{fid}: {entry.get('error')}"
                    )
    else:
        # Use "spawn" to avoid inheriting forked process state that can
        # interfere with MPI launcher behavior inside worker subprocesses.
        with ProcessPoolExecutor(max_workers=max_parallel, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(_run_fidelity_worker, task): task for task in pending_tasks}
            for future in as_completed(futures):
                task = futures[future]
                case_id_local = str(task["case_id"])
                fid = str(task["entry"]["fidelity_id"])
                try:
                    entry = future.result()
                except Exception as exc:
                    entry = copy.deepcopy(task["entry"])
                    entry["status"] = "failed"
                    entry["error"] = f"Worker crashed: {exc}"
                    entry["traceback"] = traceback.format_exc(limit=20)
                    entry["wall_time_seconds"] = None
                    entry["cd"] = None
                    entry["cl"] = None
                    entry["n_cells"] = None
                _store_case_entry(case_id_local, entry)
                if entry.get("status") == "ok":
                    _log(
                        f"  [DONE] {case_id_local}/fid_{fid} "
                        + f"(wall={entry.get('wall_time_seconds', 0):.1f}s, cells={entry.get('n_cells')})"
                    )
                else:
                    has_failure = True
                    _log(f"  [FAIL] {case_id_local}/fid_{fid}: {entry.get('error')}")
        if has_failure and stop_on_error:
            raise RuntimeError("At least one parallel run failed.")

    for case_entry in run_metadata["cases"]:
        case_id_local = str(case_entry["case_id"])
        case_entry["fidelities"] = sorted(
            case_fidelity_entries_by_case.get(case_id_local, {}).values(),
            key=lambda item: int(item.get("fidelity_index", 0)),
        )

    (run_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2), encoding="utf-8")

    if not just_init and not skip_analysis:
        _log("Running convergence analysis...")
        for case in cases:
            case_id = _case_id(case)
            case_dir = run_dir / "raw" / case_id
            analysis_dir = run_dir / "analysis" / case_id
            try:
                analyze_case(
                    case_dir=case_dir,
                    analysis_dir=analysis_dir,
                    case=case,
                    fidelity_table=fidelity_table,
                )
            except Exception as exc:
                _log(f"[WARN] Analysis failed for {case_id}: {exc}")
                if stop_on_error:
                    raise
        aggregate_analysis(run_dir)

    return run_dir
