"""exp04 effects and summaries (plan sections 7.5, 7.6).

`python -m afe.analysis.stats` reads results/tidy/exp04.parquet and writes
- exp04_effects.csv: per (dataset, arm, model, effect, failure rule): the effect on per-dataset means,
  the Nadeau-Bengio corrected t-test verdict (descriptive win / tie / loss) and its p;
- exp04_summary.csv: per (task, model, arm, effect, rule) over primary datasets: median effect with a
  bootstrap 95% CI over datasets, Wilcoxon signed-rank p with Holm within the family
  (task, model, primary effect, rule), share of wins and losses among the n_datasets datasets that have data
  (a dataset with no complete case under complete_cases is not in the denominator), coverage (the mean over all
  listed datasets, a dataset with no complete case counts as 0).

Effects (per-dataset means over the splits; positive favours the arm):
- auc_diff: mean(arm AUC) - mean(base AUC) (binary);
- logloss_rel: (base - arm) / base on mean log loss (binary, multiclass); logloss_diff: base - arm;
- rmse_rel: (base - arm) / base on mean RMSE (regression).

Failure rules (plan 7.6): base_fallback scores a non-ok unit with the base metric of its split;
complete_cases keeps only splits where the arm is ok and drops the dataset from that arm's comparison
when it has none (not_applicable units count as fallback in base_fallback and never as failures).
"""

import numpy as np
import pandas as pd
from scipy import stats as sp

from afe.paths import RESULTS
from afe.run import store

ALPHA = 0.05
N_BOOT = 10_000
EFFECTS = {  # effect -> (metric column, task types, relative)
    "auc_diff": ("auc", ("binary",), False),
    "logloss_rel": ("logloss", ("binary", "multiclass"), True),
    "logloss_diff": ("logloss", ("binary", "multiclass"), False),
    "rmse_rel": ("rmse", ("regression",), True),
}
PRIMARY = {"binary": ("auc_diff", "logloss_rel"), "multiclass": ("logloss_rel",), "regression": ("rmse_rel",)}
LOWER_BETTER = {"auc": False, "logloss": True, "rmse": True}


def nb_test(diffs: np.ndarray, ratio: float) -> tuple[float, float]:
    """Nadeau-Bengio corrected t-test on per-split differences; returns (t, p). All-zero diffs: (0, 1)."""
    j = len(diffs)
    if j < 2 or np.allclose(diffs, 0):
        return 0.0, 1.0
    var = diffs.var(ddof=1)
    if var == 0:
        return (np.inf if diffs.mean() > 0 else -np.inf), 0.0
    t = diffs.mean() / np.sqrt((1 / j + ratio) * var)
    return float(t), float(2 * sp.t.sf(abs(t), df=j - 1))


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values."""
    n, order = len(p), np.argsort(p)
    out = np.empty(n)
    out[order] = np.minimum(np.maximum.accumulate((n - np.arange(n)) * p[order]), 1)
    return out


def effects_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, arm, model, effect, rule). Splits with no base score are dropped."""
    df = df[df.base_status == "ok"]
    rows = []
    for (dataset, arm, model), g in df[df.arm != "base"].groupby(["dataset", "arm", "model"], observed=True):
        task = g.task.iloc[0]
        for effect, (metric, tasks, relative) in EFFECTS.items():
            if task not in tasks:
                continue
            ok = g.status == "ok"
            base_m, arm_m = g[f"base_{metric}"].to_numpy(), g[metric].to_numpy()
            for rule in ("base_fallback", "complete_cases"):
                if rule == "base_fallback":
                    a, b = np.where(ok, arm_m, base_m), base_m
                else:
                    a, b = arm_m[ok], base_m[ok]
                if len(a) == 0:
                    rows.append({"dataset": dataset, "task": task, "arm": arm, "model": model,
                                 "effect": effect, "rule": rule, "n_splits": 0, "value": np.nan,
                                 "p_nb": np.nan, "verdict": "no_data", "coverage": 0.0})
                    continue
                am, bm = a.mean(), b.mean()
                value = (bm - am) / bm if relative else (am - bm if metric == "auc" else bm - am)
                diffs = (b - a) if LOWER_BETTER[metric] else (a - b)  # positive favours the arm
                n_tr = g.base_n_train.to_numpy(), g.base_n_test.to_numpy()
                ratio = float(np.mean(n_tr[1] / n_tr[0]))
                t, p = nb_test(diffs, ratio)
                verdict = ("win" if diffs.mean() > 0 else "loss") if p < ALPHA else "tie"
                rows.append({"dataset": dataset, "task": task, "arm": arm, "model": model,
                             "effect": effect, "rule": rule, "n_splits": len(a), "value": value,
                             "p_nb": p, "verdict": verdict, "coverage": ok.mean()})
    return pd.DataFrame(rows)


def boot_ci(values: np.ndarray, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(values, size=(N_BOOT, len(values)), replace=True), axis=1)
    return float(np.quantile(med, 0.025)), float(np.quantile(med, 0.975))


def summary_table(eff: pd.DataFrame, primary_only: bool = True) -> pd.DataFrame:
    """Median effect with bootstrap CI over primary datasets, Wilcoxon p (Holm within family), shares."""
    rows = []
    for (task, model, effect, rule), g in eff.groupby(["task", "model", "effect", "rule"], observed=True):
        for arm, ga in g.groupby("arm", observed=True):
            v = ga.value.dropna().to_numpy()
            if len(v) == 0:
                continue
            lo, hi = boot_ci(v)
            w = sp.wilcoxon(v, zero_method="pratt").pvalue if np.any(v != 0) else 1.0
            has = ga[ga.verdict != "no_data"]  # shares are over the datasets with data (n_datasets), not all listed
            rows.append({"task": task, "model": model, "effect": effect, "rule": rule, "arm": arm,
                         "n_datasets": len(v), "median_effect": float(np.median(v)), "ci_lo": lo, "ci_hi": hi,
                         "wilcoxon_p": float(w), "share_win": (has.verdict == "win").mean(),
                         "share_loss": (has.verdict == "loss").mean(), "coverage": ga.coverage.mean()})
    out = pd.DataFrame(rows)
    out["in_family"] = [r.effect in PRIMARY[r.task] for r in out.itertuples()]
    out["wilcoxon_p_holm"] = np.nan
    for _, idx in out[out.in_family].groupby(["task", "model", "effect", "rule"], observed=True).groups.items():
        out.loc[idx, "wilcoxon_p_holm"] = holm(out.loc[idx, "wilcoxon_p"].to_numpy())
    return out


def main() -> None:
    df = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    eff = effects_table(df[df.primary])
    eff.to_csv(RESULTS / "tidy" / "exp04_effects.csv", index=False)
    summ = summary_table(eff)
    summ.to_csv(RESULTS / "tidy" / "exp04_summary.csv", index=False)
    print(f"effects: {len(eff)} rows, summary: {len(summ)} rows")


if __name__ == "__main__":
    main()
