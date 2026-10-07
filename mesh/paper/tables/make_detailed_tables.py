"""Detailed result tables of the appendix.

One row per (dimension or architecture, B~): optimal lambda*, training-set sizes and test
errors of MF at lambda* vs the HF-only baseline, and their relative difference.
  - plate and airfoil results from paper/results (evaluate.py output);
  - Poisson: HF baseline = the max-lambda cell per budget;
  - optima in the saturated HF regime (lambda* >= lambda_sat = 15) are reported as ">= 15"
    (HF-equivalent) instead of the degenerate grid value.

Outputs (this directory): {poisson,elasticity,hyperelasticity}_{truncation/solver,
discretization}.tex and airfoil_{volume,surface,cl,cd}.tex
Usage:  python make_detailed_tables.py
"""
import os.path as osp

import numpy as np
import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
RESULTS = osp.abspath(osp.join(HERE, "..", "results"))
ERR = "val/rel_l2_loss_fields_avg"
LAM_SAT = 15.0  # universal saturation threshold (recipe appendix)

PLATE_TABLES = [  # (csv, problem display name, label stem)
    ("linear_elasticity_solver_truncation.csv",
     "linear elasticity", "elasticity_solver", "solver error"),
    ("linear_elasticity_discretization.csv",
     "linear elasticity", "elasticity_discretization", "discretization error"),
    ("hyperelasticity_solver_truncation.csv",
     "hyperelasticity", "hyperelasticity_solver", "solver error"),
    ("hyperelasticity_discretization.csv",
     "hyperelasticity", "hyperelasticity_discretization", "discretization error"),
]
POISSON_BUDGETS = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]
AIRFOIL_CSV = "airfoil.csv"
L2H = r"\textbf{Rel.\ $\mathbf{L_2}$ error}"      # volume/surface = rel-L2 field error
COEFFH = r"\textbf{Rel.\ error}"                  # Cl/Cd = scalar |c_pred-c_gt|/|c_gt|
AIRFOIL_METRICS = [  # (err_col, label stem, error_type for caption, column header)
    ("val/volume_rel_l2_loss_fields_avg", "airfoil_volume", "volume field error", L2H),
    ("val/surface_rel_l2_loss_fields_avg", "airfoil_surface", "surface field error", L2H),
    ("val/rel_err_cl_avg", "airfoil_cl", "lift coefficient $C_l$ error", COEFFH),
    ("val/rel_err_cd_avg", "airfoil_cd", "drag coefficient $C_d$ error", COEFFH),
]
ARCH_ORDER = ["unet", "fno"]
ARCH_MAP = {"fno": "FNO", "unet": "U-Net"}
DIM_MAP = {1: "1D", 2: "2D", 3: "3D"}

CAPTION = (
    "Summary of the {problem} experiment results ({error_type}). "
    "Multi-fidelity (MF) uses the optimal $\\lambda^*$; "
    "High-fidelity (HF) is the high-fidelity-only baseline. "
    "$M_\\mathrm{{train}}$ denotes the number of training samples; "
    "$\\Delta (\\%)$ denotes the relative change in number of training samples "
    "and error compared to the high-fidelity baseline. "
    "Optima in the saturated high-fidelity regime are reported as "
    "$\\lambda^* \\geq 15$ (HF-equivalent)."
)


def fmt_lam(lam):
    return r"$\geq 15$" if lam >= LAM_SAT else f"{lam:.2f}"


def row_cells(row):
    return [
        f"{int(round(row['budget'])):,}",
        fmt_lam(row["lambda_star"]),
        f"{row['n_mf']:,}",
        f"{row['n_hf']:,}",
        f"{row['delta_n']:+.1f}",
        f"{row['err_mf']:.3f}",
        f"{row['err_hf']:.3f}",
        f"{row['delta_err']:+.1f}",
    ]


def add_deltas(summary):
    summary["delta_n"] = (summary.n_mf - summary.n_hf) / summary.n_hf * 100
    summary["delta_err"] = (summary.err_mf - summary.err_hf) / summary.err_hf * 100
    return summary


# --------------------------------------------------- plate / airfoil (Arch) tables
def compute_summary_plate(df, err_col=ERR):
    """Per (arch, budget) row: lambda* = argmin of err_col over the swept mixtures,
    with MF/HF sizes and errors. err_col lets airfoil reuse this per metric."""
    df = df.copy()
    df.loc[df.model_name == "ABUPT", "model_name"] = "AB-UPT"
    rows = []
    for model in sorted(df.model_name.unique()):
        mdf = df[df.model_name == model]
        for budget in sorted(mdf.b_tilde.unique()):
            bdf = mdf[mdf.b_tilde == budget]
            hf, mf_cands = bdf[bdf.hf_only], bdf[~bdf.hf_only]
            if hf.empty or mf_cands.empty:
                continue
            mean_by_lam = mf_cands.groupby("lambd")[err_col].mean()
            lam_star = mean_by_lam.idxmin()
            mf = mf_cands[mf_cands.lambd == lam_star]
            rows.append(dict(
                arch=model, budget=budget, lambda_star=lam_star,
                n_mf=int(round(mf.n_samples_realized.mean())),
                n_hf=int(round(hf.n_samples_realized.mean())),
                err_mf=mf[err_col].mean(), err_hf=hf[err_col].mean()))
    return add_deltas(pd.DataFrame(rows))


def latex_table_plate(summary, caption, label,
                      err_header=r"\textbf{Rel.\ $\mathbf{L_2}$ error}"):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Arch}} & \multirow{2}{*}{$\bm{\tilde{B}}$} & "
        r"\multirow{2}{*}{$\bm{\lambda^*}$} & "
        r"\multicolumn{3}{c}{$\mathbf{M_{\textbf{train}}}$} & "
        r"\multicolumn{3}{c}{" + err_header + r"} \\",
        r"\cmidrule(lr){4-6} \cmidrule(lr){7-9}",
        r" & & & \textbf{MF} & \textbf{HF} & $\bm{\Delta}$\textbf{(\%)} "
        r"& \textbf{MF} & \textbf{HF} & $\bm{\Delta}$\textbf{(\%)} \\",
        r"\midrule",
    ]
    archs = sorted(summary.arch.unique())
    for a_idx, arch in enumerate(archs):
        adf = summary[summary.arch == arch].reset_index(drop=True)
        for i, row in adf.iterrows():
            cell = (rf"\multirow{{{len(adf)}}}{{*}}{{{arch}}}" if i == 0 else "")
            lines.append(" & ".join([cell] + row_cells(row)) + r" \\")
        if a_idx < len(archs) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# -------------------------------------------------------------- poisson tables
def compute_summary_poisson(df, axis):
    rows = []
    sub = df[df.axis == axis]
    for dim in [1, 2, 3]:
        for arch in ARCH_ORDER:
            combo = sub[(sub.dim == dim) & (sub.architecture == arch)]
            if combo.empty:
                continue
            for budget in POISSON_BUDGETS:
                bdf = combo[np.isclose(combo.B_tilde, budget)]
                if bdf.empty:
                    continue
                mean_by_lam = bdf.groupby("lambd").mean_prediction_error.mean()
                lam_star = mean_by_lam.idxmin()
                lam_hf = bdf.lambd.max()  # HF baseline = max-lambda cell
                mf = bdf[bdf.lambd == lam_star]
                hf = bdf[bdf.lambd == lam_hf]
                rows.append(dict(
                    dim=dim, arch=arch, budget=budget, lambda_star=lam_star,
                    n_mf=int(round(mf.num_train_samples.mean())),
                    n_hf=int(round(hf.num_train_samples.mean())),
                    err_mf=mf.mean_prediction_error.mean(),
                    err_hf=hf.mean_prediction_error.mean()))
    return add_deltas(pd.DataFrame(rows))


def latex_table_poisson(summary, caption, label):
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{" + caption + "}", r"\label{tab:" + label + "}",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dim}} & \multirow{2}{*}{\textbf{Arch}} & "
        r"\multirow{2}{*}{$\bm{\tilde{B}}$} & \multirow{2}{*}{$\bm{\lambda^*}$} & "
        r"\multicolumn{3}{c}{$\mathbf{M_{\textbf{train}}}$} & "
        r"\multicolumn{3}{c}{\textbf{Rel.\ $\mathbf{L_2}$ error}} \\",
        r"\cmidrule(lr){5-7} \cmidrule(lr){8-10}",
        r" & & & & \textbf{MF} & \textbf{HF} & $\bm{\Delta}$\textbf{(\%)} "
        r"& \textbf{MF} & \textbf{HF} & $\bm{\Delta}$\textbf{(\%)} \\",
        r"\midrule",
    ]
    dims = sorted(summary.dim.unique())
    for d_idx, dim in enumerate(dims):
        ddf = summary[summary.dim == dim]
        archs = [a for a in ARCH_ORDER if a in ddf.arch.values]
        first_in_dim = True
        for a_idx, arch in enumerate(archs):
            adf = ddf[ddf.arch == arch].reset_index(drop=True)
            for i, row in adf.iterrows():
                dim_cell = (rf"\multirow{{{len(ddf)}}}{{*}}{{{DIM_MAP[dim]}}}"
                            if first_in_dim else "")
                arch_cell = (rf"\multirow{{{len(adf)}}}{{*}}{{{ARCH_MAP[arch]}}}"
                             if i == 0 else "")
                first_in_dim = False
                lines.append(" & ".join([dim_cell, arch_cell] + row_cells(row))
                             + r" \\")
            if a_idx < len(archs) - 1:
                lines.append(r"\cmidrule(lr){2-10}")
        if d_idx < len(dims) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    for csv, problem, label, error_type in PLATE_TABLES:
        df = pd.read_csv(f"{RESULTS}/{csv}")
        summary = compute_summary_plate(df)
        print(f"=== {label} ===")
        print(summary.to_string(index=False))
        tex = latex_table_plate(
            summary, CAPTION.format(problem=problem, error_type=error_type), label)
        with open(f"{HERE}/{label}.tex", "w") as f:
            f.write(tex + "\n")
        print(f"wrote {label}.tex\n")

    p = pd.read_csv(f"{RESULTS}/poisson.csv")
    for axis, label, error_type in [
            ("truncation", "poisson_truncation", "solver error"),
            ("discretization", "poisson_discretization", "discretization error")]:
        summary = compute_summary_poisson(p, axis)
        print(f"=== {label} ===")
        print(summary.to_string(index=False))
        tex = latex_table_poisson(
            summary, CAPTION.format(problem="Poisson", error_type=error_type), label)
        with open(f"{HERE}/{label}.tex", "w") as f:
            f.write(tex + "\n")
        print(f"wrote {label}.tex\n")

    # airfoil: 4 metric tables (Arch layout), gated on the coeff re-eval columns
    af_path = f"{RESULTS}/{AIRFOIL_CSV}"
    af = pd.read_csv(af_path) if osp.exists(af_path) else pd.DataFrame()
    if not af.empty and AIRFOIL_METRICS[0][0] in af.columns:
        for err_col, label, error_type, err_header in AIRFOIL_METRICS:
            summary = compute_summary_plate(af, err_col=err_col)
            print(f"=== {label} ===")
            print(summary.to_string(index=False))
            tex = latex_table_plate(
                summary, CAPTION.format(problem="airfoil", error_type=error_type), label,
                err_header=err_header)
            with open(f"{HERE}/{label}.tex", "w") as f:
                f.write(tex + "\n")
            print(f"wrote {label}.tex\n")
    else:
        print("NOTE: airfoil tables (volume/surface/cl/cd) skipped -- columns "
              f"'{AIRFOIL_METRICS[0][0]}' etc. not yet present.")


if __name__ == "__main__":
    main()
