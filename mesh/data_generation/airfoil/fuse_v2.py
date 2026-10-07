"""Fuse batch 1 + batch 2 into mf_dataset_processed_v2 (camera-ready airfoil basis).

PRECONDITION: both batches went through the SAME full pipeline (postprocess ->
error-calc -> filter_metadata.py), so both carry the final cost schema, cd_rel_amp and
is_clean. The filter criteria are absolute per-case thresholds (no population
statistics), so per-batch filtering + concatenation is identical to a joint filter --
the fuse is a pure concatenation with renumbering, guarded by assertions:

  1. Hardlink sample dirs into v2: batch 1 keeps 00001-01000 (numbering identical to the
     paper dataset), batch 2 is renumbered 01001-02000.
  2. Concatenate metadata_unfiltered.csv (renumbered batch-2 sample ids); write
     metadata.csv = is_clean rows. Same renumbering for sample_info.csv.

Run:  python fuse_v2.py --root <runs dir> --batch1 <processed campaign 1> --batch2 <processed campaign 2> --out <merged>
"""
import argparse
import subprocess
from pathlib import Path

import pandas as pd

_ap = argparse.ArgumentParser(description="Fuse two processed airfoil batches into one dataset (hardlinks + renumbering).")
_ap.add_argument("--root", type=Path, required=True, help="directory that holds the processed batch folders")
_ap.add_argument("--batch1", required=True, help="batch-1 folder name under --root (keeps its numbering)")
_ap.add_argument("--batch2", required=True, help="batch-2 folder name under --root (renumbered by --offset)")
_ap.add_argument("--out", required=True, help="output folder name under --root")
_ap.add_argument("--offset", type=int, default=1000, help="sample-id offset applied to batch 2")
_args = _ap.parse_args()
ROOT = _args.root
B1 = ROOT / _args.batch1
B2 = ROOT / _args.batch2
V2 = ROOT / _args.out
OFFSET = _args.offset

STR = {"sample_id": str, "fidelity_id": str}


def renumber(sid):
    return str(int(sid) + OFFSET).zfill(5)


def main():
    assert B1.is_dir() and B2.is_dir(), "both processed batches must exist"
    V2.mkdir(exist_ok=True)

    # 1. hardlink sample dirs
    for src, off in ((B1, 0), (B2, OFFSET)):
        dirs = sorted(d for d in src.iterdir() if d.is_dir() and d.name.isdigit())
        print(f"linking {len(dirs)} sample dirs from {src.name} (offset {off})")
        for d in dirs:
            dst = V2 / str(int(d.name) + off).zfill(5)
            if not dst.exists():
                subprocess.run(["cp", "-al", str(d), str(dst)], check=True)

    # 2. metadata + sample_info (both batches must be in post-filter state)
    m1 = pd.read_csv(B1 / "metadata_unfiltered.csv", dtype=STR)
    m2 = pd.read_csv(B2 / "metadata_unfiltered.csv", dtype=STR)
    for tag, m in (("batch1", m1), ("batch2", m2)):
        for col in ("simulation_time_raw", "cd_rel_amp", "is_clean"):
            assert col in m.columns, f"{tag} missing '{col}' -- run filter_metadata.py on it first"
    m2["sample_id"] = m2.sample_id.map(renumber)
    assert not set(m1.case_id) & set(m2.case_id), "case id collision between batches"
    fused = pd.concat([m1, m2], ignore_index=True)
    assert fused.sample_id.nunique() == m1.sample_id.nunique() + m2.sample_id.nunique(), \
        "sample id collision after renumbering"
    fused.to_csv(V2 / "metadata_unfiltered.csv", index=False)
    fused[fused.is_clean].to_csv(V2 / "metadata.csv", index=False)

    s1 = pd.read_csv(B1 / "sample_info.csv", dtype=STR)
    s2 = pd.read_csv(B2 / "sample_info.csv", dtype=STR)
    s2["sample_id"] = s2.sample_id.map(renumber)
    pd.concat([s1, s2], ignore_index=True).to_csv(V2 / "sample_info.csv", index=False)

    n_h5 = sum(1 for _ in V2.glob("*/*.h5"))
    print(f"v2: {fused.case_id.nunique()} cases ({int(fused.is_clean.sum() / 11)} clean), "
          f"{len(fused)} metadata rows, {n_h5} h5 files")


if __name__ == "__main__":
    main()
