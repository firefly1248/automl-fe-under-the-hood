"""Run-to-run noise of the time-limited tools (plan section 8, noise measurement).

`python -m afe.analysis.noise` reads the repeat runs of the same units (unit key `rep` = 1, 2; `make exp02-noise` and
`make exp06-noise`) next to the original run (run 0) and writes
- tidy/noise_runs.csv: per exp, dataset, arm, fold and run the test loss (log loss or RMSE) and wall seconds;
- tidy/noise_arms.csv: per exp, dataset and arm the mean loss of each run, the spread between the runs
  ((max - min) / mean of the run means) and the mean absolute relative difference between runs on the same fold;
- tidy/noise_contrasts.csv: per contrast and dataset the mean effect (relative improvement of the arm over its
  reference) in each run, the spread of the three effects in percentage points, the fold-bootstrap 95% CI of the
  original run next to it, and whether all runs lie inside that CI. Where the reference was repeated too (exp06) each
  run uses its own reference run; where it was not (exp02: A_def, A, D0) every run uses the original reference.
Only folds present in all three runs enter. The fold CI covers the variation between folds; the spread covers the
variation between repeated runs of one unit, which the CI does not.
"""

import numpy as np
import pandas as pd
import yaml

from afe.analysis import exp02, exp06
from afe.paths import RESULTS

RUNS = (0, 1, 2)
ARM_COLS = ["exp", "dataset", "arm", "n_folds", "n_runs", *[f"loss_run{r}" for r in RUNS], "spread_rel", "fold_diff_rel"]
CONTRAST_COLS = ["exp", "contrast", "dataset", "n_folds", "n_runs", *[f"effect_run{r}" for r in RUNS], "spread_pp", "ci_lo",
                 "ci_hi", "ci_width_pp", "inside_ci"]
CONTRASTS = {  # exp -> contrast name -> (arm, reference); the effect is (ref - arm) / ref
    "exp02": {c: exp02.CONTRASTS[c] for c in ("D-A_def", "D-A", "D-D0", "E-D0")},
    "exp06": {c: exp06.CONTRASTS[c] for c in ("lama-ag", "h2o-ag")},
}
MODULES = {"exp02": exp02, "exp06": exp06}
REFERENCES = {"exp02": ["A", "A_def", "D0"], "exp06": []}  # references of the contrasts that were not repeated: run 0 only


def config(exp: str) -> dict:
    return yaml.safe_load((RESULTS.parent / "experiments" / MODULES[exp].EXP / "config.yaml").read_text())["noise"]


def runs(exp: str) -> pd.DataFrame:
    """Loss per unit of the original run (0) and the repeats, for the noise datasets and arms only."""
    mod = MODULES[exp]
    cfg = config(exp)
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    rows = []
    for noise in (False, True):
        for r in exp02.records(mod.EXP, noise=noise):
            u = r["unit"]
            if r["status"] != "ok" or u["dataset"] not in cfg["datasets"] or u["arm"] not in cfg["arms"] + REFERENCES[exp]:
                continue
            task = ds.loc[u["dataset"], "task"]
            rows.append({"exp": exp, "dataset": u["dataset"], "task": task, "arm": u["arm"], "fold": u["fold"],
                         "run": u.get("rep", 0), "loss": r["result"][exp02.METRIC[task]], "wall_s": r["wall_s"]})
    return pd.DataFrame(rows)


def arms(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (exp, dataset, arm), g in df.groupby(["exp", "dataset", "arm"]):
        if arm not in config(exp)["arms"]:
            continue
        w = g.pivot(index="fold", columns="run", values="loss").dropna()
        w = w[[r for r in RUNS if r in w]]
        if w.shape[1] < 2 or w.empty:
            continue
        means = w.mean()
        pair = [(w[a] - w[b]).abs() / ((w[a] + w[b]) / 2) for i, a in enumerate(w) for b in list(w)[i + 1:]]
        out.append({"exp": exp, "dataset": dataset, "arm": arm, "n_folds": len(w), "n_runs": w.shape[1],
                    **{f"loss_run{r}": means.get(r, np.nan) for r in RUNS},
                    "spread_rel": (means.max() - means.min()) / means.mean(),
                    "fold_diff_rel": float(np.mean([p.mean() for p in pair]))})
    return pd.DataFrame(out, columns=ARM_COLS)


def effect_per_run(wide: pd.DataFrame, arm: str, ref: str, run: int) -> pd.Series:
    """Per-fold relative improvement of `arm` over `ref` in `run` (the reference of run 0 if `ref` was not repeated)."""
    nan = pd.Series(np.nan, index=wide.index)
    a = wide[(arm, run)] if (arm, run) in wide else nan
    b = wide[(ref, run)] if (ref, run) in wide else wide[(ref, 0)]
    return (b - a) / b


def contrasts(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for exp, pairs in CONTRASTS.items():
        per = pd.read_csv(RESULTS / "tidy" / f"{exp}_contrasts.csv")
        for name, (arm, ref) in pairs.items():
            for dataset in sorted(df[(df.exp == exp) & (df.arm == arm)].dataset.unique()):
                g = df[(df.exp == exp) & (df.dataset == dataset)]
                wide = g.pivot_table(index="fold", columns=["arm", "run"], values="loss")
                if (ref, 0) not in wide:
                    continue
                eff = pd.DataFrame({r: effect_per_run(wide, arm, ref, r) for r in RUNS}).dropna(axis=1, how="all").dropna()
                if eff.shape[1] < 2 or eff.empty:
                    continue
                means = eff.mean()
                ci = per[(per.contrast == name) & (per.dataset == dataset)].iloc[0]
                out.append({"exp": exp, "contrast": name, "dataset": dataset, "n_folds": len(eff), "n_runs": eff.shape[1],
                            **{f"effect_run{r}": means.get(r, np.nan) for r in RUNS},
                            "spread_pp": 100 * (means.max() - means.min()),
                            "ci_lo": ci.ci_lo, "ci_hi": ci.ci_hi, "ci_width_pp": 100 * (ci.ci_hi - ci.ci_lo),
                            "inside_ci": bool(((means >= ci.ci_lo) & (means <= ci.ci_hi)).all())})
    return pd.DataFrame(out, columns=CONTRAST_COLS)


def main() -> None:
    df = pd.concat([runs("exp02"), runs("exp06")], ignore_index=True)
    tidy = RESULTS / "tidy"
    df.sort_values(["exp", "dataset", "arm", "fold", "run"]).round(6).to_csv(tidy / "noise_runs.csv", index=False)
    arms(df).round(6).to_csv(tidy / "noise_arms.csv", index=False)
    contrasts(df).round(6).to_csv(tidy / "noise_contrasts.csv", index=False)
    print(f"noise: {int((df.run > 0).sum())} repeat units, {df.run.nunique()} runs")


if __name__ == "__main__":
    main()
