"""Extrapolation recipe for the optimal mixture lambda*(B~), evaluated on the paper runs.

Applied to the 8 settings {linear elasticity, hyperelasticity} x {solver, discretization} x
{Transolver, AB-UPT} (plus Poisson and airfoil where available), read from paper/results:

  1. lambda* estimates: 3-point parabolic vertex (argmin +-1 cell, quadratic on the seed-mean
     error); argmins at the grid edge are censored and never used as fit points.
  2. Warm-up screen: slide the 3-budget pilot window up while the pilot shows an LF-edge
     argmin or a net drift towards LF (>= 2 held-out budgets required above the window).
  3. Saturation censoring: a pilot vertex >= lambda_sat (= 15) is a one-sided observation;
     >= 1 censored pilot budget -> collapse -> HF-only everywhere.
  4. Fit: weighted least squares lambda* = a + b log2(B~) on the pilot vertices, weights
     1 / max(SE, 0.25)^2 (SE from a 300-draw seed bootstrap of the whole vertex procedure).
  5. Deployment at the held-out budgets: prediction >= lambda_sat -> HF-only; otherwise the
     nearest swept lambda cell. Scored as regret against the best swept cell.

Budget ladders: solver pilot {100, 200, 400} -> held out {800, 1600, 3200, 6400};
discretization pilot {100, 200, 400} -> held out {800, 1600, 3200}.

Outputs (this directory): recipe_results.csv (read by make_regret_table.py and overhead.py),
printed summary.
Usage:  python recipe.py
"""
import os.path as osp
import zlib

import numpy as np
import pandas as pd

OUT = osp.dirname(osp.abspath(__file__))
RESULTS = osp.join(OUT, "..", "results")
ERR = "val/rel_l2_loss_fields_avg"
K = 3
N_BOOT = 300
LAM_SAT = {"solver": 15.0, "discretization": 15.0}  # universal threshold; well above the >=95%-HF-spend point on all menus, results insensitive in [10,15]
THRESHOLD = 0.1  # savings eps*, consistent with figures/results
EDGE_SATCENS = "--no-edge-satcens" not in __import__("sys").argv  # recipe amendment

EXPS = {
    ("Lin. elast.", "solver"): "linear_elasticity_solver_truncation",
    ("Lin. elast.", "discretization"): "linear_elasticity_discretization",
    ("Hyperelast.", "solver"): "hyperelasticity_solver_truncation",
    ("Hyperelast.", "discretization"): "hyperelasticity_discretization",
}


def load_cells():
    """-> tidy cells df: setting, group, model, axis, b_tilde, lambd, err, hf_only."""
    frames = []
    for (group, axis), exp in EXPS.items():
        df = pd.read_csv(f"{RESULTS}/{exp}.csv")
        df.loc[df.model_name == "ABUPT", "model_name"] = "AB-UPT"
        for model in ["Transolver", "AB-UPT"]:
            sub = df[df.model_name == model].copy()
            sub["setting"] = f"{'lin' if 'Lin' in group else 'hyp'}_" \
                             f"{'solv' if axis == 'solver' else 'disc'}_{model}"
            sub["group"], sub["model"], sub["axis"] = group, model, axis
            sub["err"] = sub[ERR]
            sub["hf_mode"] = "flagged"
            frames.append(sub[["setting", "group", "model", "axis", "b_tilde",
                               "lambd", "err", "hf_only", "seed", "hf_mode"]])
    # airfoil (discretization axis = mesh refinement; real HF-only baselines). Uses the
    # VOLUME field error (the airfoil's primary field, matching the savings choice); the
    # combined field-avg is surface-dominated and pins the low-budget optimum at the LF
    # grid edge. Ladder 50..800 -> pilot {50,100,200}, held-out {400,800}.
    af_path = f"{RESULTS}/airfoil.csv"
    AF_ERR = "val/volume_rel_l2_loss_fields_avg"
    if osp.exists(af_path):
        af = pd.read_csv(af_path)
        af.loc[af.model_name == "ABUPT", "model_name"] = "AB-UPT"
        for model in ["Transolver", "AB-UPT"]:
            sub = af[af.model_name == model].copy()
            sub["setting"] = f"airfoil_disc_{model}"
            sub["group"], sub["model"], sub["axis"] = "Airfoil", model, "discretization"
            sub["err"] = sub[AF_ERR]
            sub["hf_mode"] = "flagged"
            frames.append(sub[["setting", "group", "model", "axis", "b_tilde",
                               "lambd", "err", "hf_only", "seed", "hf_mode"]])

    # poisson (grid models; HF baseline = max-lambda cell per budget)
    p = pd.read_csv(f"{RESULTS}/poisson.csv")
    # two small 2D budgets with a different normalization are excluded, as in the paper
    p = p[~(np.isclose(p.B_tilde, 5.42534722) & (p.dim == 2))]
    p = p[~(np.isclose(p.B_tilde, 14.28571429) & (p.dim == 2))]
    for dim in (1, 2, 3):
        for ax, axis in [("discretization", "discretization"), ("truncation", "solver")]:
            for arch in (["unet", "fno"] if dim < 3 else ["unet"]):
                d = p[(p.dim == dim) & (p.axis == ax) & (p.architecture == arch)].copy()
                if d.empty:
                    continue
                d["setting"] = f"poisson{dim}d_{axis[:4]}_{arch}"
                d["group"], d["model"] = f"Poisson {dim}D", {"unet": "UNet", "fno": "FNO"}[arch]
                d["axis"], d["hf_only"] = axis, False
                d["err"] = d.mean_prediction_error
                d = d.rename(columns={"B_tilde": "b_tilde"})
                d["hf_mode"] = "max_lambda"
                frames.append(d[["setting", "group", "model", "axis", "b_tilde",
                                 "lambd", "err", "hf_only", "seed", "hf_mode"]])
    out = pd.concat(frames, ignore_index=True)
    out["hf_mode"] = out.hf_mode.fillna("flagged")
    return out


def vertex(lams, errs):
    """3-point parabolic vertex around the argmin; None if argmin on edge."""
    i0 = int(np.argmin(errs))
    if i0 in (0, len(lams) - 1):
        return None
    x = np.asarray(lams[i0 - 1:i0 + 2], dtype=float)
    y = np.asarray(errs[i0 - 1:i0 + 2], dtype=float)
    c2, c1, _ = np.polyfit(x, y, 2)
    if c2 <= 0:
        return float(lams[i0])
    return float(np.clip(-c1 / (2 * c2), x.min(), x.max()))


def find_btilde_at_error(btildes, errors, threshold):
    order = np.argsort(btildes)
    bt, er = np.asarray(btildes)[order], np.asarray(errors)[order]
    for i in range(len(er) - 1):
        if er[i] >= threshold >= er[i + 1]:
            t = (np.log10(threshold) - np.log10(er[i])) / (np.log10(er[i + 1]) - np.log10(er[i]))
            return 10 ** (np.log10(bt[i]) + t * (np.log10(bt[i + 1]) - np.log10(bt[i])))
    return np.nan


class Setting:
    def __init__(self, name, df):
        self.name = name
        self.group, self.model, self.axis = df.group.iloc[0], df.model.iloc[0], df.axis.iloc[0]
        self.hf_mode = df.hf_mode.iloc[0] if "hf_mode" in df.columns else "flagged"
        self.lam_sat = LAM_SAT[self.axis]
        # savings eps*: airfoil volume errors are much lower than the plate/poisson
        # fields, so its curves never reach the global 0.1 -> use the figure's 2%
        self.threshold = 0.02 if self.group == "Airfoil" else THRESHOLD
        mf = df[~df.hf_only]
        self.cell_err = {b: g.droplevel(0).sort_index()
                         for b, g in mf.groupby(["b_tilde", "lambd"]).err.mean().groupby(level=0)}
        self.cell_seeds = {(b, l): g.err.values for (b, l), g in mf.groupby(["b_tilde", "lambd"])}
        self.budgets = sorted(self.cell_err)
        self.oracle_lam = {b: self.cell_err[b].idxmin() for b in self.budgets}
        self.oracle_err = {b: self.cell_err[b].min() for b in self.budgets}
        self.edge = {b: self.cell_err[b].idxmin() in
                     (self.cell_err[b].index.min(), self.cell_err[b].index.max())
                     for b in self.budgets}
        # an argmin AT a grid edge that lies in the
        # saturated regime (lambda_edge >= lambda_sat) is the same one-sided censored
        # observation as a vertex >= lambda_sat -- the edge cell IS the HF plateau.
        self.edge_sat = {b: (self.cell_err[b].idxmin() == self.cell_err[b].index.max()
                             and self.cell_err[b].index.max() >= self.lam_sat)
                         for b in self.budgets}
        self.vert = {b: vertex(self.cell_err[b].index.values, self.cell_err[b].values)
                     for b in self.budgets if len(self.cell_err[b]) >= 3}
        if self.hf_mode == "max_lambda":
            self.hf_curve = pd.Series({b: self.cell_err[b].iloc[-1] for b in sorted(self.cell_err)})
        else:
            self.hf_curve = df[df.hf_only].groupby("b_tilde").err.mean()

        # pilot window (ladder is already 2x-spaced)
        eligible = [b for b in self.budgets if len(self.cell_err[b]) >= 3]
        ladder = []
        for b in eligible:
            if not ladder or b >= 1.9 * ladder[-1]:
                ladder.append(b)
        chosen, self.window_stable = None, True
        min_heldout = 2 if len(ladder) > K + 1 else 1
        for i in range(len(ladder) - K + 1):
            win = ladder[i:i + K]
            if len([b for b in eligible if b > win[-1]]) < min_heldout:
                break
            lams = [self.oracle_lam[b] for b in win]
            left_edge = any(self.oracle_lam[b] == self.cell_err[b].index.min() for b in win)
            if not left_edge and lams[-1] >= lams[0]:
                chosen = i
                break
        if chosen is None:
            chosen, self.window_stable = 0, False
        self.calib = ladder[chosen:chosen + K]
        self.heldout = [b for b in eligible if b > self.calib[-1]]

    def vertex_se(self, b):
        s = self.cell_err[b]
        lams = s.index.values
        seed_sets = [self.cell_seeds[(b, l)] for l in lams]
        rng = np.random.default_rng(zlib.crc32(f"{self.name}|{b:.4f}|vertex".encode()))
        draws = []
        for _ in range(N_BOOT):
            means = np.array([v[rng.integers(0, len(v), len(v))].mean() for v in seed_sets])
            v = vertex(lams, means)
            if v is not None:
                draws.append(v)
        return float(np.std(draws)) if len(draws) > 10 else 1.0

    def fit(self):
        pts = [(b, self.vert[b], self.vertex_se(b)) for b in self.calib
               if not self.edge[b] and self.vert.get(b) is not None]
        self.sat_censored = [p for p in pts if p[1] >= self.lam_sat]
        self.collapse = (len(self.sat_censored) > 0
                         or (EDGE_SATCENS and any(self.edge_sat[b] for b in self.calib)))
        pts = [p for p in pts if p[1] < self.lam_sat]
        self.fit_pts = pts
        if self.collapse:
            return np.nan, np.nan
        if len(pts) < 2:
            return None
        x = np.log2([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        wts = 1.0 / np.maximum([p[2] for p in pts], 0.25) ** 2
        b_slope, a = np.polyfit(x, y, 1, w=np.sqrt(wts))
        return a, b_slope

    def line_boot(self):
        rng = np.random.default_rng(zlib.crc32(f"{self.name}|lineboot".encode()))
        out = []
        for _ in range(N_BOOT):
            pts = []
            for b in self.calib:
                s = self.cell_err[b]
                lams = s.index.values
                means = np.array([self.cell_seeds[(b, l)][rng.integers(
                    0, len(self.cell_seeds[(b, l)]), len(self.cell_seeds[(b, l)]))].mean()
                    for l in lams])
                v = vertex(lams, means)
                if v is not None and v < self.lam_sat:
                    pts.append((b, v))
            if len(pts) < 2:
                continue
            xs = np.log2([p[0] for p in pts])
            ys = [p[1] for p in pts]
            b_s, a_s = np.polyfit(xs, ys, 1)
            if b_s < 0:
                a_s, b_s = float(np.mean(ys)), 0.0
            out.append((a_s, b_s))
        return np.array(out)

    def deploy_err(self, b, lam_pred):
        if lam_pred >= self.lam_sat:
            if self.hf_mode == "max_lambda":
                return self.cell_err[b].iloc[-1], "HF(sat)"
            hf = self.hf_curve
            if b in hf.index:
                return hf.loc[b], "HF(sat)"
            lo, hi = hf[hf.index <= b], hf[hf.index >= b]
            if len(lo) and len(hi) and lo.index[-1] != hi.index[0]:
                t = (np.log10(b) - np.log10(lo.index[-1])) / (np.log10(hi.index[0]) - np.log10(lo.index[-1]))
                return 10 ** ((1 - t) * np.log10(lo.iloc[-1]) + t * np.log10(hi.iloc[0])), "HF(sat)"
            return (lo.iloc[-1] if len(lo) else hi.iloc[0]), "HF(sat)"
        s = self.cell_err[b]
        snap = s.index.values[np.argmin(np.abs(s.index.values - lam_pred))]
        return s[snap], f"cell {snap:.2f}"

    def savings(self, override=None):
        mf = {b: self.oracle_err[b] for b in self.budgets}
        if override:
            mf.update(override)
        b_mf = find_btilde_at_error(list(mf), list(mf.values()), self.threshold)
        b_hf = find_btilde_at_error(self.hf_curve.index.values, self.hf_curve.values, self.threshold)
        return b_hf / b_mf


def main():
    cells = load_cells()
    settings = {name: Setting(name, g) for name, g in cells.groupby("setting")}
    rows = []
    for name, s in sorted(settings.items()):
        ab = s.fit()
        if ab is None:
            print(f"{name}: <2 fit points, skipped")
            continue
        a, b_slope = ab
        preds, deploys, regrets = {}, {}, []
        for b in s.heldout:
            lam_p = np.inf if s.collapse else a + b_slope * np.log2(b)
            e, how = s.deploy_err(b, lam_p)
            preds[b], deploys[b] = lam_p, (e, how)
            regrets.append(e / s.oracle_err[b] - 1.0)
        S_oracle = s.savings()
        S_rule = s.savings({b: deploys[b][0] for b in s.heldout})
        retention = ((S_rule - 1) / (S_oracle - 1)
                     if np.isfinite(S_rule) and np.isfinite(S_oracle) else np.nan)
        rows.append(dict(
            setting=name, group=s.group, model=s.model, axis=s.axis,
            slope=b_slope, intercept=a, stable=s.window_stable,
            collapse_in_pilot=s.collapse, pilot=str([int(x) for x in s.calib]),
            **{f"regret_{i+1}": (regrets[i] if i < len(regrets) else np.nan) for i in range(4)},
            heldout=str([int(x) for x in s.heldout]),
            **{f"lambda_pred_{i+1}": (preds[s.heldout[i]] if i < len(s.heldout)
                                      else np.nan) for i in range(4)},
            **{f"deploy_{i+1}": (deploys[s.heldout[i]][1] if i < len(s.heldout)
                                 else "") for i in range(4)},
            mean_regret3=float(np.mean(regrets[:3])),
            n_sat_deploys=sum(1 for b in s.heldout if deploys[b][1] == "HF(sat)"),
            S_oracle=S_oracle, S_rule=S_rule, retention=retention))

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT}/recipe_results.csv", index=False)
    print(res[["setting", "slope", "pilot", "stable", "collapse_in_pilot", "regret_1",
               "regret_2", "regret_3", "regret_4", "S_oracle", "S_rule", "retention",
               "n_sat_deploys"]].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print(f"\nmedians: regret_1 {res.regret_1.median():.1%}, regret_2 {res.regret_2.median():.1%}, "
          f"regret_3 {res.regret_3.median():.1%}, regret_4 {res.regret_4.median():.1%} "
          f"| mean(first3) {res.mean_regret3.mean():.1%} "
          f"| retention=1.0 in {(res.retention > 0.999).sum()}/{len(res)}")
    for axis in ["solver", "discretization"]:
        g = res[res.axis == axis]
        print(f"{axis:15s}: median regret_1..3 "
              f"{g.regret_1.median():.1%}/{g.regret_2.median():.1%}/{g.regret_3.median():.1%} "
              f"| mean(first3) {g.mean_regret3.mean():.1%}")

    print(f"\nwrote {OUT}/recipe_results.csv")


if __name__ == "__main__":
    main()
