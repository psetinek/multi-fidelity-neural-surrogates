#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-fidelity airfoil dataset pipeline (data_generation/airfoil).")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config_default.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the fidelity table and execution plan without running simulations.")
    parser.add_argument("--just-init", action="store_true", help="Generate meshes only (skip CFD solve).")
    parser.add_argument("--skip-analysis", action="store_true", help="Skip convergence post-processing.")
    parser.add_argument("--run-name", type=str, default=None, help="Override run name.")
    parser.add_argument("--case-limit", type=int, default=None, help="Execute only the first N cases from config.")
    parser.add_argument(
        "--fidelity-ids",
        type=str,
        default=None,
        help="Comma-separated fidelity IDs (e.g. 00,05,10) to run a subset.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum number of fidelities to run in parallel for each case.",
    )
    parser.add_argument("--overwrite-run", action="store_true", help="Delete and recreate run directory if it already exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fidelity_ids = None
    if args.fidelity_ids:
        fidelity_ids = [chunk.strip() for chunk in args.fidelity_ids.split(",") if chunk.strip()]

    run_dir = run_pipeline(
        config_path=args.config,
        dry_run=bool(args.dry_run),
        just_init=bool(args.just_init),
        skip_analysis=bool(args.skip_analysis),
        run_name_override=args.run_name,
        case_limit=args.case_limit,
        fidelity_ids=fidelity_ids,
        overwrite_run=bool(args.overwrite_run),
        max_parallel_override=args.max_parallel,
    )
    print(f"Run complete: {run_dir}")


if __name__ == "__main__":
    main()
