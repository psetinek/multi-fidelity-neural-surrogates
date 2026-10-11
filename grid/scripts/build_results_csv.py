"""Combine the per-run `result.csv` files of all experiments into one CSV.

Every run writes one `result.csv` into its own Hydra output directory. The
per-run CSV carries (axis, dim, architecture, lambd, budget, seed, errors,
base_channels, cond_dim, mean_prediction_error_super); the metadata that
distinguishes the ablations lives only in the directory path:
num_training_steps (`_steps_<n>`, duration/arch ablations) and the top of
the truncation menu (`kmax_<K>`, endpoint ablation). This script walks each
results root, parses that path metadata, tags every row with its
`experiment`, and appends the derived cost columns.

Sources (relative to --results-root; the layout `sweeps/launch.py` writes):
    main            truncation/, discretization/
    endpoint        endpoint_ablation/truncation/kmax_<K>/
    arch            arch_ablation/truncation/
    duration        duration_ablation/truncation/
    dst             dst_ablation/discretization/
    dual_reference  dual_reference/{truncation,discretization}/
    bimodal         bimodal_ablation/{truncation,discretization}/

Columns added beyond the per-run schema:
    experiment            source tag (above)
    num_training_steps    parsed from `_steps_<n>` in the path, else 20000
    dst_interpolation     True only for `dst` rows
    xi_max                top of the cell's fidelity menu (kmax for endpoint)
    c_xi_max              cost of one xi_max sample: K_max (truncation) or
                          N_max**dim (discretization)
    B_tilde               budget / c_xi_max, i.e. the budget in units of
                          highest-fidelity samples

The CSV shipped with the paper is `data/results.csv`; by default this
script writes to `results/results.csv` so it never overwrites it.

Usage:
    uv run python scripts/build_results_csv.py
    uv run python scripts/build_results_csv.py --results-root results --out results/results.csv
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments._common import LEGACY_COLUMN_DEFAULTS  # noqa: E402

# Max fidelity value per (axis, dim) for the standard menus in configs/axis/.
XI_MAX_FID = {
    ("truncation", 1): 3,
    ("truncation", 2): 7,
    ("truncation", 3): 8,
    ("discretization", 1): 48,
    ("discretization", 2): 96,
    ("discretization", 3): 96,
}

# (experiment tag, list of roots). Order defines output grouping.
SOURCES = [
    ("main", ["truncation", "discretization"]),
    ("endpoint", ["endpoint_ablation/truncation"]),
    ("arch", ["arch_ablation/truncation"]),
    ("duration", ["duration_ablation/truncation"]),
    ("dst", ["dst_ablation/discretization"]),
    ("dual_reference", ["dual_reference/truncation", "dual_reference/discretization"]),
    ("bimodal", ["bimodal_ablation/truncation", "bimodal_ablation/discretization"]),
]

OUT_COLUMNS = [
    "experiment",
    "axis",
    "dim",
    "architecture",
    "lambd",
    "budget",
    "realised_cost",
    "seed",
    "num_train_samples",
    "num_training_steps",
    "base_channels",
    "cond_dim",
    "dst_interpolation",
    "xi_max",
    "c_xi_max",
    "B_tilde",
    "mean_prediction_error",
    "mean_residual_error",
    "mean_prediction_error_super",
]

STEPS_RE = re.compile(r"_steps_(\d+)")
KMAX_RE = re.compile(r"kmax_(\d+)")


def load_source(tag, roots):
    rows = []
    for root in roots:
        for p in Path(root).rglob("result.csv"):
            df = pd.read_csv(p)
            s = str(p)
            m = STEPS_RE.search(s)
            df["num_training_steps"] = int(m.group(1)) if m else 20000
            km = KMAX_RE.search(s)
            df["kmax"] = int(km.group(1)) if km else pd.NA
            df["experiment"] = tag
            df["dst_interpolation"] = tag == "dst"
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args()

    root = Path(args.results_root)
    parts = []
    for tag, roots in SOURCES:
        d = load_source(tag, [root / r for r in roots])
        if d.empty:
            print(f"  {tag:15s} — no rows found")
            continue
        parts.append(d)
        print(f"  {tag:15s} {len(d):6d} rows")
    if not parts:
        sys.exit(f"No result.csv files found under {root}/")

    df = pd.concat(parts, ignore_index=True)

    # Backfill the architecture columns onto rows written before they existed.
    for col, default in LEGACY_COLUMN_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default).astype(int)
    if "mean_prediction_error_super" not in df.columns:
        df["mean_prediction_error_super"] = pd.NA

    # xi_max: kmax for endpoint rows, else the standard menu top per (axis, dim).
    def xi_max_row(r):
        if r["experiment"] == "endpoint" and pd.notna(r["kmax"]):
            return int(r["kmax"])
        return XI_MAX_FID[(r["axis"], int(r["dim"]))]

    df["xi_max"] = df.apply(xi_max_row, axis=1)
    df["c_xi_max"] = [
        x if a == "truncation" else x ** int(dim)
        for a, dim, x in zip(df["axis"], df["dim"], df["xi_max"])
    ]
    df["B_tilde"] = df["budget"] / df["c_xi_max"]

    df = (
        df[OUT_COLUMNS]
        .sort_values(
            ["experiment", "axis", "dim", "architecture", "budget", "lambd", "seed"]
        )
        .reset_index(drop=True)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}  ({len(df)} rows, {len(df.columns)} columns)")
    print("\nrows per experiment:")
    print(df["experiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
