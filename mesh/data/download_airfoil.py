"""Download the multi-fidelity airfoil dataset from the Hugging Face Hub.

The dataset (https://huggingface.co/datasets/psetinek/multi-fidelity-airfoil) holds 1,627
NACA airfoil cases, each simulated on 11 mesh levels (00 = coarsest, 10 = finest):

    <out_dir>/metadata.csv
    <out_dir>/<case>/<level>.h5

By default it is written to data/airfoil/processed next to this script, which is the
data_path of configs/dataset/airfoil.yaml. The full dataset is ~107 GB.

The download resumes where it stopped. To avoid overwriting data that did not come from
this script, it refuses to write into a non-empty directory that holds no previous download
(unless --force).

Only needs huggingface_hub (pip install huggingface_hub). Large downloads go through a
local chunk cache; point HF_XET_CACHE to a disk with enough space if your home is small.

Usage:
    python data/download_airfoil.py                              # into data/airfoil/processed
    python data/download_airfoil.py --out-dir /scratch/airfoil   # elsewhere; set data_path accordingly
"""

import argparse
import os
import os.path as osp

import pandas as pd
from huggingface_hub import snapshot_download

REPO_ID = "psetinek/multi-fidelity-airfoil"
DEFAULT_OUT = osp.join(osp.dirname(osp.abspath(__file__)), "airfoil", "processed")
# written by huggingface_hub into every local_dir it downloads to
MARKER = osp.join(".cache", "huggingface")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", default=DEFAULT_OUT, help="target directory (default: %(default)s)")
    p.add_argument("--max-workers", type=int, default=8, help="parallel downloads")
    p.add_argument("--force", action="store_true",
                   help="write into a non-empty directory that holds no previous download")
    return p.parse_args()


def check_target(out_dir, force):
    if not osp.isdir(out_dir) or not os.listdir(out_dir):
        return
    if osp.isdir(osp.join(out_dir, MARKER)):
        print(f"resuming the download in {out_dir}")
        return
    if not force:
        raise SystemExit(
            f"{out_dir} is not empty and holds no previous download from this script; refusing to "
            "write into it (existing files would be overwritten). Choose another --out-dir or pass --force."
        )


def verify(out_dir):
    # every (case, level) row of metadata.csv must have its h5 file
    metadata = pd.read_csv(osp.join(out_dir, "metadata.csv"), dtype={"sample_id": str, "fidelity_id": str})
    expected = [osp.join(s, f"{l}.h5") for s, l in zip(metadata["sample_id"], metadata["fidelity_id"])]
    missing = [f for f in expected if not osp.isfile(osp.join(out_dir, f))]
    print(f"{metadata['sample_id'].nunique()} cases in metadata.csv: {len(expected) - len(missing)}/{len(expected)} files present")
    if missing:
        raise SystemExit(f"{len(missing)} files missing (e.g. {missing[:3]}); rerun to resume the download")


def main():
    args = parse_args()
    out_dir = osp.abspath(args.out_dir)
    check_target(out_dir, args.force)
    os.makedirs(out_dir, exist_ok=True)
    print(f"downloading {REPO_ID} to {out_dir}")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset", local_dir=out_dir, max_workers=args.max_workers)
    verify(out_dir)


if __name__ == "__main__":
    main()
