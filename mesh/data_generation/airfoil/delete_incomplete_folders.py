"""Remove partially written fid_XX folders flagged by check_run_completion.py, then rerun
the campaign (skip_existing=true) to fill them.  Asks for confirmation before deleting.

    python delete_incomplete_folders.py --completion-csv <run_dir>/completion_check.csv
"""
import argparse
import shutil

import pandas as pd

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--completion-csv", required=True, help="completion_check.csv written by check_run_completion.py")
args = parser.parse_args()

df = pd.read_csv(args.completion_csv)
targets = df[df["state"] == "missing_metadata_partial"]["fid_dir"].dropna().unique()
print("removing", len(targets), "partial fid dirs")
confirm = input("Are you sure you want to delete these folders? (y/n) ")
if confirm.lower() != "y":
    print("aborting")
    raise SystemExit(0)
for d in targets:
    print("rm -rf", d)
    shutil.rmtree(d, ignore_errors=True)
print("done")
