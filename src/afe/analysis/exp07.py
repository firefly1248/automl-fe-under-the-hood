"""exp07 CatBoost ablation: do external categorical features help CatBoost less than LightGBM because CatBoost builds
its own combinations? (protocol change A4, 2026-11-17).

`python -m afe.analysis.exp07` reads the exp07 records (CatBoost with `max_ctr_complexity=1`, same cached generator
output and seeds as exp04) and the exp04 CatBoost fits with the default setting, and writes
- tidy/exp07.parquet: one row per exp07 unit (dataset, repeat, fold, arm, status, wall time, test loss);
- tidy/exp07_effects.csv: per arm and dataset, over the folds present in all four fits (default and `ctr=1`, arm and base):
  the effect of the arm over `base` with the default CatBoost, with `ctr=1`, and their paired difference
  (`ctr1 - default`), each as the mean improvement in the task's primary metric (AUC difference for binary, relative log loss
  reduction for multiclass, relative RMSE reduction for regression; as in exp04) with a fold-bootstrap 95% CI;
- tidy/exp07_summary.csv: per arm over datasets `n_better` (sign of the mean difference), `n_clear_better` and
  `n_clear_worse` (the fold CI excludes zero); for `base` the same on `ctr1` over default CatBoost (negative: CatBoost loses
  without its own combinations). No median: the datasets' effects are in different units.
Folds are the 10 outer folds (2 repeats of 5) as units; complete cases only. The bootstrap uses one random stream over arms
and datasets in the order below, so a CI depends on the position: append new arms at the end.
"""

import numpy as np
import pandas as pd

from afe.analysis import exp02
from afe.paths import RESULTS

EXP = "exp07_ctr_ablation"
ARMS = ["ag_tabprep_keepcat", "lama_adv", "openfe"]  # base is the reference
METRIC = {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}  # the primary metric of each task type (exp04)
N_BOOT = 10_000


def tidy() -> pd.DataFrame:
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    rows = []
    for r in exp02.records(EXP):
        u, res = r["unit"], r.get("result") or {}
        task = ds.loc[u["dataset"], "task"]
        rows.append({"dataset": u["dataset"], "task": task, "repeat": u["repeat"], "fold": u["fold"], "arm": u["arm"],
                     "status": r["status"], "wall_s": r["wall_s"], "loss": (res.get("test") or {}).get(METRIC[task])})
    df = pd.DataFrame(rows, columns=["dataset", "task", "repeat", "fold", "arm", "status", "wall_s", "loss"])
    return df.sort_values(["dataset", "arm", "repeat", "fold"]).reset_index(drop=True)


def gain(task: str, ref, new):
    """Improvement of `new` over `ref`: the AUC difference for binary, the relative loss reduction otherwise."""
    return new - ref if task == "binary" else (ref - new) / ref


def default_fits(exp04: pd.DataFrame, datasets: list[str]) -> pd.DataFrame:
    """The exp04 CatBoost units of the ablation arms and datasets: test loss with the default setting."""
    d = exp04[(exp04.model == "catboost") & exp04.dataset.isin(datasets) & (exp04.status == "ok")
              & exp04.arm.isin(["base", *ARMS])].copy()
    d["loss"] = [getattr(r, METRIC[r.task]) for r in d.itertuples()]
    return d[["dataset", "task", "repeat", "fold", "arm", "loss"]]


def effects(df: pd.DataFrame, default: pd.DataFrame) -> pd.DataFrame:
    """Per arm and dataset the three effects with fold-bootstrap CIs; `df` is `tidy()`, `default` is `default_fits()`."""
    ok = df[df.status == "ok"]
    key = ["dataset", "task", "repeat", "fold"]
    d = default.pivot_table(index=key, columns="arm", values="loss").add_suffix("_default")
    c = ok.pivot_table(index=key, columns="arm", values="loss").add_suffix("_ctr1")
    w = d.join(c, how="inner").reset_index()
    rng = np.random.default_rng(0)
    rows = []

    def add(arm: str, dataset: str, task: str, d_default, d_ctr1, diff):
        row = {"arm": arm, "dataset": dataset, "task": task, "n_folds": len(diff)}
        for name, v in (("default", d_default), ("ctr1", d_ctr1), ("diff", diff)):
            if v is None:
                continue
            boot = rng.choice(v, size=(N_BOOT, len(v)), replace=True).mean(axis=1)
            row |= {f"effect_{name}": v.mean(), f"ci_lo_{name}": np.quantile(boot, 0.025),
                    f"ci_hi_{name}": np.quantile(boot, 0.975)}
        rows.append(row)

    for dataset, g in w.groupby("dataset"):
        task = g.task.iloc[0]
        # base: relative improvement of ctr=1 over the default CatBoost (negative: it loses without combinations)
        gb = g.dropna(subset=["base_default", "base_ctr1"])
        if len(gb):
            v = gain(task, gb.base_default, gb.base_ctr1).to_numpy()
            add("base", dataset, task, None, None, v)
        for arm in ARMS:
            if f"{arm}_default" not in g or f"{arm}_ctr1" not in g:
                continue  # the arm has no fit of one kind on this dataset
            ga = g.dropna(subset=["base_default", "base_ctr1", f"{arm}_default", f"{arm}_ctr1"])
            if not len(ga):
                continue
            e_default = gain(task, ga.base_default, ga[f"{arm}_default"]).to_numpy()
            e_ctr1 = gain(task, ga.base_ctr1, ga[f"{arm}_ctr1"]).to_numpy()
            add(arm, dataset, task, e_default, e_ctr1, e_ctr1 - e_default)
    return pd.DataFrame(rows)


def summary(eff: pd.DataFrame) -> pd.DataFrame:
    g = eff.groupby("arm")
    s = g.effect_diff.agg(n_datasets="size", n_better=lambda v: int((v > 0).sum())).reset_index()
    s["n_clear_better"] = g.ci_lo_diff.agg(lambda v: int((v > 0).sum())).to_numpy()
    s["n_clear_worse"] = g.ci_hi_diff.agg(lambda v: int((v < 0).sum())).to_numpy()
    return s


def main() -> None:
    df = tidy()
    if df.empty:
        return print("exp07: no records yet (make exp07)")
    exp04 = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    eff = effects(df, default_fits(exp04, sorted(df.dataset.unique())))
    tidy_dir = RESULTS / "tidy"
    df.to_parquet(tidy_dir / "exp07.parquet", index=False)
    eff.round(6).to_csv(tidy_dir / "exp07_effects.csv", index=False)
    summary(eff).round(6).to_csv(tidy_dir / "exp07_summary.csv", index=False)
    print(f"exp07: {len(df)} units, {int((df.status == 'ok').sum())} ok, {len(eff)} arm-dataset rows")


if __name__ == "__main__":
    main()
