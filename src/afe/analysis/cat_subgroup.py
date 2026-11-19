"""skrub_tv and ag_default on the primary datasets that have categorical columns (2026-11-19).

The median over all 50 datasets is dominated by numeric-only datasets where both arms leave the table unchanged (68% of
the `skrub_tv` units are exactly 0), so the headline is a count on the datasets with at least one categorical column
(`n_categorical > 0` in datasets.csv: 28 of the 50 primary ones), per arm and downstream model.

`python -m afe.analysis.cat_subgroup` reads tidy/exp04.parquet and writes
- tidy/exp04_cat_effects.csv: per arm, model and primary dataset (all 50), over the splits where the arm and base are both ok
  (complete cases): the effect in the task's primary metric as in `exp04_effects.csv` (AUC difference of the means for binary,
  relative log loss or RMSE reduction of the means for multiclass and regression, positive favours the arm) with a paired
  bootstrap 95% CI over the splits;
- tidy/exp04_cat_summary.csv: per arm and model over the datasets with categorical columns, for `task` all and each task type:
  `n_better` (sign of the mean), `n_clear_better` and `n_clear_worse` (the CI excludes zero), and the secondary medians
  over those datasets and over all 50 (`median_cat`, `median_all`; per task type only, the units differ between types).
One random stream over (arm, model, dataset) in sorted order: a CI depends on the position, append new arms at the end.
"""

import numpy as np
import pandas as pd

from afe.paths import RESULTS

ARMS = ["ag_default", "skrub_tv"]
N_BOOT = 10_000


METRIC = {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}


def effect(arm: np.ndarray, base: np.ndarray, task: str):
    """AUC difference of the means (binary) or relative reduction of the mean loss (multiclass, regression); vectorised over
    the last axis, so a bootstrap sample matrix gives one effect per row."""
    a, b = arm.mean(axis=-1), base.mean(axis=-1)
    return a - b if task == "binary" else (b - a) / b


def effect_ci(arm: np.ndarray, base: np.ndarray, task: str, rng) -> tuple[float, float, float]:
    """Point estimate and paired bootstrap 95% CI over the splits."""
    idx = rng.integers(0, len(arm), size=(N_BOOT, len(arm)))
    boot = effect(arm[idx], base[idx], task)
    return float(effect(arm, base, task)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def effects(exp04: pd.DataFrame, has_cat: dict[str, bool]) -> pd.DataFrame:
    d = exp04[exp04.arm.isin(ARMS) & (exp04.status == "ok") & (exp04.base_status == "ok") & exp04.primary]
    rng = np.random.default_rng(0)
    rows = []
    for (arm, model, dataset), g in d.groupby(["arm", "model", "dataset"], observed=True, sort=True):
        task = g.task.iloc[0]
        m = METRIC[task]
        e, lo, hi = effect_ci(g[m].to_numpy(), g[f"base_{m}"].to_numpy(), task, rng)
        rows.append({"arm": arm, "model": model, "dataset": dataset, "task": task, "has_cat": has_cat[dataset],
                     "n_splits": len(g), "effect": e, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def summary(eff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, model), g in eff.groupby(["arm", "model"]):
        for task in ["all", "binary", "multiclass", "regression"]:
            t = g if task == "all" else g[g.task == task]
            c = t[t.has_cat]
            row = {"arm": arm, "model": model, "task": task, "n_datasets": len(c), "n_better": int((c.effect > 0).sum()),
                   "n_clear_better": int((c.ci_lo > 0).sum()), "n_clear_worse": int((c.ci_hi < 0).sum())}
            if task != "all":  # medians within a task type only: the effect units differ between types
                row |= {"median_cat": c.effect.median(), "n_all": len(t), "median_all": t.effect.median()}
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    exp04 = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    eff = effects(exp04, (ds.n_categorical > 0).to_dict())
    eff.round(6).to_csv(RESULTS / "tidy" / "exp04_cat_effects.csv", index=False)
    s = summary(eff)
    s.round(6).to_csv(RESULTS / "tidy" / "exp04_cat_summary.csv", index=False)
    print(s[s.task == "all"].to_string(index=False))


if __name__ == "__main__":
    main()
