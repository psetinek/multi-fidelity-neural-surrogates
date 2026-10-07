"""Regret tables from recipe_results.csv.

1. regret_table_main.tex -- compact: rows = datasets, columns = axes, cells =
   median (mean) held-out regret in %, pooled over architectures and budgets.
2. regret_table_full_{solver,discretization}.tex -- per-setting tables, one per
   fidelity axis: rows = dataset x architecture, columns = pilot window, regret
   at each successive held-out budget (ladder rungs above the pilot; absolute B~
   differs per row), and savings retention. Collapse-in-pilot rows (recipe
   prescribes HF-only deployment) are dagger-marked; negative regrets = the HF
   deploy beats the swept oracle. lambda_pred/deploy per holdout live in the
   CSV only.

Usage:  python make_regret_table.py   (after recipe.py)
"""
import os.path as osp

import pandas as pd

HERE = osp.dirname(osp.abspath(__file__))
CSV = osp.join(HERE, "recipe_results.csv")
OUT = osp.join(HERE, "regret_table_main.tex")

PROBLEMS = {  # setting-name prefix -> display name (row order)
    "poisson1d": "Poisson 1D",
    "poisson2d": "Poisson 2D",
    "poisson3d": "Poisson 3D",
    "lin": "Linear elasticity",
    "hyp": "Hyperelasticity",
    "airfoil": "Airfoil",
}
AXES = {"solv": "Solver error", "disc": "Discretization error"}
REGRET_COLS = ["regret_1", "regret_2", "regret_3", "regret_4"]


def parse_setting(s):
    parts = s.split("_")
    return parts[0], parts[1]  # (problem_key, axis_key)


def main():
    df = pd.read_csv(CSV)
    df[["problem", "axis"]] = df.setting.apply(
        lambda s: pd.Series(parse_setting(s)))
    # floor regrets (and retention) at zero for presentation: deployments that beat
    # the swept oracle (collapse-in-pilot HF deploys) count as zero regret
    for c in REGRET_COLS + ["retention"]:
        df[c] = df[c].clip(lower=0)

    N_H = 3  # 1st..3rd held-out budget in the overview (4th only in the appendix)

    for agg_name in ("median", "mean"):
        def agg(sub, i):
            vals = sub[f"regret_{i}"].dropna() * 100
            return getattr(vals, agg_name)() if len(vals) else None

        cells = []  # numeric (None = no data)
        for pkey, pname in PROBLEMS.items():
            sub = df[df.problem == pkey]
            cells.append([pname] + [agg(sub[sub.axis == a], i)
                                    for a in AXES for i in range(1, N_H + 1)])
        # Average row = column-wise mean of the displayed dataset cells
        # (zeros count, missing "--" entries do not)
        avg = ["Average"]
        for j in range(1, 1 + 2 * N_H):
            col = [c[j] for c in cells if c[j] is not None]
            avg.append(sum(col) / len(col) if col else None)
        cells.append(avg)
        rows = [[c[0]] + [("--" if v is None else f"{v:.1f}") for v in c[1:]]
                for c in cells]

        print(f"--- {agg_name} ---")
        for r in rows:
            print(f"{r[0]:20s} | solver {' '.join(f'{c:>5s}' for c in r[1:4])} | "
                  f"disc {' '.join(f'{c:>5s}' for c in r[4:7])}")

        ords = [r"\textbf{1\textsuperscript{st}}", r"\textbf{2\textsuperscript{nd}}",
                r"\textbf{3\textsuperscript{rd}}"][:N_H]
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            rf"\caption{{{agg_name.capitalize()} held-out regret (\%) of the "
            r"extrapolation recipe per dataset and fidelity axis, at the first "
            rf"three held-out budgets (ascending; {agg_name}s taken over "
            r"architectures).}",
            r"\label{tab:regret_overview}",
            rf"\begin{{tabular}}{{l{'r' * N_H}{'r' * N_H}}}",
            r"\toprule",
            r"\multirow{2}{*}{\textbf{Dataset}} & "
            rf"\multicolumn{{{N_H}}}{{c}}{{\textbf{{Solver error}}}} & "
            rf"\multicolumn{{{N_H}}}{{c}}{{\textbf{{Discretization error}}}} \\",
            rf"\cmidrule(lr){{2-{1 + N_H}}} \cmidrule(lr){{{2 + N_H}-{1 + 2 * N_H}}}",
            rf" & {' & '.join(ords)} & {' & '.join(ords)} \\",
            r"\midrule",
        ]
        for r in rows:
            if r[0] == "Average":
                lines.append(r"\midrule")
            lines.append(" & ".join(r) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        out = osp.join(HERE, f"regret_table_main_{agg_name}.tex")
        with open(out, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {out}\n")

    for akey in AXES:
        write_axis_table(df, akey)


def write_axis_table(df, akey):
    """One verbose per-setting table per fidelity axis: for each held-out budget the
    predicted lambda, the deployed cell (nearest swept lambda, or HF via the
    saturation clause), and the regret."""
    out = osp.join(HERE, f"regret_table_full_{AXES[akey].split()[0].lower()}.tex")
    model_names = {"AB-UPT": "AB-UPT", "Transolver": "Transolver",
                   "fno": "FNO", "unet": "U-Net"}
    sub_all = df[df.axis == akey]
    # recipe exports at most 4 held-out budgets (regret_1..4)
    n_h = min(4, max(len(eval(h)) for h in sub_all.heldout))  # noqa: S307 -- own CSV

    def fmt_regret(r, i):
        v = r[f"regret_{i}"]
        return "--" if pd.isna(v) else f"{v * 100:.1f}"

    axis_word = AXES[akey].split()[0].lower()
    ordinals = [r"\textbf{1\textsuperscript{st}}", r"\textbf{2\textsuperscript{nd}}",
                r"\textbf{3\textsuperscript{rd}}", r"\textbf{4\textsuperscript{th}}"][:n_h]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Per-setting recipe results on the {axis_word}-error axis: "
        r"regret at each held-out budget (ladder rungs above the calibration "
        r"window, in ascending order).}",
        rf"\label{{tab:regret_{axis_word}}}",
        rf"\begin{{tabular}}{{llc{'r' * n_h}}}",
        r"\toprule",
        r"\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Arch}} & "
        r"\multirow{2}{*}{\textbf{Calibration} $\bm{\tilde{B}}$} & "
        rf"\multicolumn{{{n_h}}}{{c}}{{\textbf{{Regret at held-out}} "
        r"$\bm{\tilde{B}}$ \textbf{(\%)}} \\",
        rf"\cmidrule(lr){{4-{3 + n_h}}}",
        rf" & & & {' & '.join(ordinals)} \\",
        r"\midrule",
    ]
    for pkey, pname in PROBLEMS.items():
        sub = sub_all[sub_all.problem == pkey].sort_values("setting")
        if sub.empty:
            continue
        first = True
        for _, r in sub.iterrows():
            model = model_names.get(r.setting.split("_", 2)[2],
                                    r.setting.split("_", 2)[2])
            dagger = r"$^\dagger$" if r.collapse_in_pilot else ""
            cells = [fmt_regret(r, i) for i in range(1, n_h + 1)]
            dcell = (rf"\multirow{{{len(sub)}}}{{*}}{{{pname}}}" if first else "")
            lines.append(f"{dcell} & {model}{dagger} & "
                         f"{r.pilot.strip('[]')} & " + " & ".join(cells) + r" \\")
            first = False
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}"]
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
