"""Clean-case filtering: metadata_unfiltered.csv -> metadata.csv (step 6 of the README).

A case is kept only if both criteria hold, and cases are dropped as a whole (all fidelity
levels), so every case in the final dataset has the complete 11-level menu:

  1. Solver stability of the highest-fidelity run: the relative oscillation amplitude of Cd,
     (max - min) / |mean| over the final --tail-percent of the iteration history, must be
     <= --cutoff (default 0.05).
  2. Usable low-fidelity levels: the relative L2 error of the lowest fidelity level must be
     <= --max-lf-error (default 0.4). Requires the error-calculation stage to have run.

The script also finalizes the cost columns (idempotent): the solver wall time written by the
postprocessing stage is kept as simulation_time_raw, and simulation_time (the cost column used
for training) is wall time x --n-proc, halved for high angle-of-attack cases (|AoA| > 10 deg)
because those run twice the iterations under the AirfRANS protocol.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

COEFF_SUFFIX = "postProcessing/forceCoeffs1/0/coefficient.dat"


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-case solver-stability filter: metadata_unfiltered.csv -> metadata.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--metadata-csv", type=Path, required=True,
                   help="unfiltered metadata (after error calculation); updated in place "
                        "with cd_rel_amp/is_clean + finalized cost schema")
    p.add_argument("--raw-dirs", type=Path, nargs="+", required=True,
                   help="raw run roots searched in order for each case_id (multiple for "
                        "fused batches)")
    p.add_argument("--out-csv", type=Path, default=None,
                   help="clean metadata output (default: metadata.csv next to input)")
    p.add_argument("--cutoff", type=float, default=0.05,
                   help="max relative Cd amplitude for a clean case (batch-1 convention)")
    p.add_argument("--max-lf-error", type=float, default=0.4,
                   help="max rel_l2_error at the LOWEST fidelity for a clean case "
                        "(coarse-mesh outlier screen; batch-1 separation 0.382/0.415)")
    p.add_argument("--tail-percent", type=float, default=0.2,
                   help="fraction of the iteration history profiled")
    p.add_argument("--hf-fid", default="fid_10",
                   help="fidelity folder whose history is profiled")
    p.add_argument("--n-proc", type=int, default=16,
                   help="OpenFOAM ranks per run; cost = wall time * n_proc")
    return p.parse_args()


def read_dat_file(path: Path) -> pd.DataFrame:
    """OpenFOAM coefficient.dat: last '# ... Time ...' comment line = column header."""
    with open(path) as f:
        lines = f.readlines()
    col_line = None
    for line in reversed(lines):
        if line.startswith("#") and "Time" in line:
            col_line = line
            break
    if col_line is None:
        raise ValueError(f"no header line in {path}")
    colnames = [c.strip() for c in col_line.strip("# \n").split("\t")]
    return pd.read_csv(path, sep=r"\s+", comment="#", header=None, names=colnames)


def cd_rel_amp_for_case(case_dir: Path, hf_fid: str, tail_percent: float) -> float:
    coeff_path = case_dir / hf_fid / COEFF_SUFFIX
    if not coeff_path.exists():
        return np.nan
    hist = read_dat_file(coeff_path)
    tail_length = int(len(hist) * tail_percent)
    if tail_length < 2:
        return np.nan
    tail = hist["Cd"].tail(tail_length)
    mean_val = tail.mean()
    if abs(mean_val) <= 1e-6:
        return np.nan
    return float((tail.max() - tail.min()) / abs(mean_val))


def main():
    args = parse_args()
    out_csv = args.out_csv or args.metadata_csv.parent / "metadata.csv"
    df = pd.read_csv(args.metadata_csv, dtype={"sample_id": str, "fidelity_id": str})

    # cost schema finalization (idempotent): CPU cost = wall * n_proc, then normalized to
    # the standard 20k-iteration protocol -- high-AoA (>10 deg) cases run 2x iterations
    # (config openfoam.n_iter.authors_high_aoa), so their cost is divided by 2 to stay
    # comparable across cases (paper convention).
    if "simulation_time_raw" not in df.columns:
        df = df.rename(columns={"simulation_time": "simulation_time_raw"})
        iter_norm = np.where(df["aoa_deg"] > 10.0, 2.0, 1.0)
        df["simulation_time"] = df["simulation_time_raw"] * args.n_proc / iter_norm
        print(f"cost schema: simulation_time = raw * {args.n_proc} / iter_norm "
              f"({int((iter_norm == 2).sum())} high-AoA rows halved)")

    # per-case stability profile (HF history from whichever raw root has the case)
    cases = sorted(df.case_id.unique())
    rel_amp, resolved_root = {}, {}
    for case_id in tqdm(cases, desc="profiling Cd stability"):
        case_dir = next((r / case_id for r in args.raw_dirs if (r / case_id).is_dir()), None)
        if case_dir is None:
            rel_amp[case_id], resolved_root[case_id] = np.nan, "MISSING"
            continue
        rel_amp[case_id] = cd_rel_amp_for_case(case_dir, args.hf_fid, args.tail_percent)
        resolved_root[case_id] = str(case_dir.parent)

    df["cd_rel_amp"] = df.case_id.map(rel_amp)
    amps = pd.Series(rel_amp)
    cd_ok = set(amps[amps.notna() & (amps <= args.cutoff)].index)

    # criterion 2: lowest-fidelity error outlier screen (needs error-calc stage done)
    if df.rel_l2_error.isna().all():
        raise SystemExit("rel_l2_error is empty -- run the error-calculation stage first")
    low = df.loc[df.groupby("case_id").fidelity_raw.idxmin(), ["case_id", "rel_l2_error"]]
    lf_err = low.set_index("case_id").rel_l2_error
    lf_ok = set(lf_err[lf_err.notna() & (lf_err <= args.max_lf_error)].index)

    # criterion 3: menu completeness -- every clean case must have ALL fidelity levels
    # (a failed solve at one level, stop_on_error=false, leaves a 10-level case; the MF
    # machinery assumes full menus per case)
    n_levels = df.fidelity_id.nunique()
    counts = df.groupby("case_id").fidelity_id.count()
    complete = set(counts[counts == n_levels].index)

    clean_cases = cd_ok & lf_ok & complete
    df["is_clean"] = df.case_id.isin(clean_cases)

    # notebook-style summary, per raw root
    n_missing = int(amps.isna().sum())
    n_garbage = int((amps.notna() & (amps > args.cutoff)).sum())
    n_lf_out = len(cd_ok - lf_ok)
    n_incomplete = len((cd_ok & lf_ok) - complete)
    print(f"\n--- Dataset Cleaning Summary (cd cutoff {args.cutoff*100:.1f}%, "
          f"lf-error {args.max_lf_error}) ---")
    print(f"Total cases:          {len(cases)}")
    print(f"Missing/crashed:      {n_missing} (dropped)")
    print(f"Cd-unstable:          {n_garbage} (dropped)")
    print(f"LF-error outliers:    {n_lf_out} (dropped, of the Cd-stable)")
    print(f"Incomplete menus:     {n_incomplete} (dropped, passed other criteria but "
          f"missing levels)")
    print(f"Clean cases kept:     {len(clean_cases)} ({len(clean_cases)/len(cases)*100:.1f}%)")
    roots = pd.Series(resolved_root)
    for root in sorted(roots.unique()):
        cs = set(roots[roots == root].index)
        print(f"  {root}: {len(cs & clean_cases)}/{len(cs)} clean")

    df.to_csv(args.metadata_csv, index=False)
    df[df.is_clean].to_csv(out_csv, index=False)
    print(f"\nupdated {args.metadata_csv} ({len(df)} rows)")
    print(f"wrote   {out_csv} ({int(df.is_clean.sum())} clean rows)")


if __name__ == "__main__":
    main()
