"""Hyperelasticity dataset-description figures (appendix: discretization / solver error).

Usage:  python hyperelasticity.py [--sample-id 00001]
Reads data/hyperelasticity/{discretization,solver_truncation}/processed.
"""
import argparse

from plate_common import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-id", default="00001", help="sample shown in the field-comparison figures")
    args = ap.parse_args()
    # damped chord method, rho ~ 0.95: two decimals in the legend
    run("hyperelasticity", rho_decimals=2, sample_id=args.sample_id)
