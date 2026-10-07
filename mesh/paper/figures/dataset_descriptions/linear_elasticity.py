"""Linear elasticity dataset-description figures (appendix: discretization / solver error).

Usage:  python linear_elasticity.py [--sample-id 00001]
Reads data/linear_elasticity/{discretization,solver_truncation}/processed.
"""
import argparse

from plate_common import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default="00001", help="sample shown in the field-comparison figures")
    args = ap.parse_args()
    # Gauss-Seidel rate rho ~ 0.9999x: five decimals needed to be readable in the legend
    run("linear_elasticity", rho_decimals=5, sample_id=args.sample_id)
