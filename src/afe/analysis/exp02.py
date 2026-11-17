"""exp02 tidy table, contrasts and figure (plan section 8).

`python -m afe.analysis.exp02` reads the exp02 run records and writes
- tidy/exp02.parquet: one row per unit (dataset, fold, arm, status, test metrics, models trained, fit time);
- tidy/exp02_models.csv: per unit and trained model, its stack level, validation score and fit time;
- tidy/exp02_contrasts.csv: per dataset and contrast the mean relative improvement over folds (positive
  favours the first arm) with a fold bootstrap 95% CI, on the eval metric (log loss or RMSE);
- tidy/exp02_contrasts_summary.csv: per task type `n_better` (sign of the mean effect), `n_clear_better` and
  `n_clear_worse` (fold CI excludes zero) of `n_datasets`, and the median (secondary, descriptive);
- tidy/exp02_arm_means.csv and exp02_arm_wall.csv: per dataset and arm the mean and sd of the metric over folds and the
  mean wall seconds; per arm the wall seconds over all units;
- tidy/exp02_rounds.csv: median boosting rounds of the single-model arms per dataset;
- figures/exp02/contrasts.{svg,png,csv} (the main contrasts) and single_model.* (hyperparameters vs features of `_r13`).
Only folds where both arms of a contrast are ok enter it.
"""

import gzip
import json
import re

import numpy as np
import pandas as pd

from afe.analysis.figures import save
from afe.paths import RESULTS
from afe.run import store

EXP = "exp02_autogluon_ablation"
METRIC = {"binary": "logloss", "multiclass": "logloss", "regression": "rmse"}
CONTRASTS = {  # name -> (arm, reference); improvement of arm over reference
    "B-A": ("B", "A"), "D-A": ("D", "A"), "D-D0": ("D", "D0"), "E-D0": ("E", "D0"), "B-B2": ("B", "B2"),
    # single-model diagnostics (worker_single.py): hyperparameters and features of `_r13` told apart
    "D-A_def": ("D", "A_def"), "A-A_def": ("A", "A_def"), "A_et0-A": ("A_et0", "A"), "B-B_def": ("B", "B_def"),
    "B_et0-B": ("B_et0", "B"), "B_def-A_def": ("B_def", "A_def"), "B_et0-A_et0": ("B_et0", "A_et0"),
    # worker_hp.py, elevators, kr-vs-kp and socmob only: one group of `_r13` values back to default, against A_et0
    **{f"A_et0_no{g}-A_et0": (f"A_et0_no{g}", "A_et0") for g in ("samp", "cap", "reg", "cat")},
    "A_et0_noreg-A_def": ("A_et0_noreg", "A_def"), "A_et0_nocat-A_def": ("A_et0_nocat", "A_def"),
}
MAIN = ["B-A", "D-A", "D-D0", "E-D0", "B-B2"]
SINGLE = ["A-A_def", "A_et0-A", "B_def-A_def", "B_et0-A_et0", "B-B_def", "B_et0-B"]  # B-A is in MAIN
N_BOOT = 10_000


def records(exp=EXP):
    """Loose records first, then the packed ones that are not already loose (a pack keeps copies of both)."""
    raw = RESULTS / "raw" / exp
    seen = set()
    for p in sorted((raw / "fit").glob("*.json")):
        seen.add(p.stem)
        yield json.loads(p.read_text())
    gz = raw / "fit.jsonl.gz"
    if gz.exists():
        for line in gzip.open(gz, "rt"):
            r = json.loads(line)
            if r["hash"] not in seen:
                yield r


def level(model: str) -> int:
    """Stack level from the AutoGluon model name (`..._L2`); AutoGluon's own `stack_level` field is empty here."""
    m = re.search(r"_L(\d+)$", model)
    return int(m.group(1)) if m else 1


def tidy() -> tuple[pd.DataFrame, pd.DataFrame]:
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    rows, models = [], []
    for r in records():
        u, res = r["unit"], r.get("result") or {}
        row = {"dataset": u["dataset"], "task": ds.loc[u["dataset"], "task"], "fold": u["fold"], "arm": u["arm"],
               "status": r["status"], "wall_s": r["wall_s"], "peak_rss_mb": r["peak_rss_mb"]}
        if res:
            m = res["models"]
            row |= {"logloss": res.get("logloss"), "rmse": res.get("rmse"), "auc": res.get("auc"),
                    "n_features_in": res["n_features_in"], "best_model": res["best_model"], "n_models": len(m),
                    "n_stack_levels": len({level(k) for k in m if not k.startswith("WeightedEnsemble")}),
                    "sum_fit_s": sum(v["fit_time"] for v in m.values())}
            models += [{"dataset": u["dataset"], "fold": u["fold"], "arm": u["arm"], "model": k,
                        "stack_level": level(k), "val_score": v["val_score"], "fit_time": v["fit_time"],
                        "num_features": v["num_features"],
                        "num_boost_round": (v.get("hyperparameters_fit") or {}).get("num_boost_round")}
                       for k, v in m.items()]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["dataset", "arm", "fold"]).reset_index(drop=True)
    return df, pd.DataFrame(models)


def contrasts(df: pd.DataFrame, pairs=CONTRASTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df.status == "ok"].copy()
    ok["loss"] = [getattr(r, METRIC[r.task]) for r in ok.itertuples()]
    wide = ok.pivot_table(index=["dataset", "task", "fold"], columns="arm", values="loss").reset_index()
    rng = np.random.default_rng(0)  # one stream over the dict: a CI depends on the contrast's position, append only
    rows = []
    for name, (arm, ref) in pairs.items():
        if arm not in wide or ref not in wide:
            continue
        for (dataset, task), g in wide.groupby(["dataset", "task"]):
            d = ((g[ref] - g[arm]) / g[ref]).dropna().to_numpy()  # relative improvement of arm over ref
            if len(d) == 0:
                continue
            boot = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(axis=1)
            rows.append({"contrast": name, "dataset": dataset, "task": task, "n_folds": len(d),
                         "effect": d.mean(), "ci_lo": np.quantile(boot, 0.025), "ci_hi": np.quantile(boot, 0.975)})
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    per = pd.DataFrame(rows)
    g = per.groupby(["contrast", "task"])
    summ = g.effect.agg(median="median", n_datasets="size", n_better=lambda v: int((v > 0).sum())).reset_index()
    # n_better is the sign of the mean effect; the clear counts need the fold CI to exclude zero
    summ["n_clear_better"] = g.ci_lo.agg(lambda v: int((v > 0).sum())).to_numpy()
    summ["n_clear_worse"] = g.ci_hi.agg(lambda v: int((v < 0).sum())).to_numpy()
    return per, summ


def figure(per: pd.DataFrame, exp="exp02", pairs=MAIN, name="contrasts") -> None:
    import matplotlib.pyplot as plt

    names = [c for c in pairs if c in set(per.contrast)]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 6), sharey=True)
    axes = np.atleast_1d(axes)
    order = per.drop_duplicates("dataset").sort_values(["task", "dataset"]).dataset.tolist()
    for ax, contrast in zip(axes, names):
        g = per[per.contrast == contrast].set_index("dataset").reindex(order)
        y = np.arange(len(order))
        ax.hlines(y, g.ci_lo, g.ci_hi, color="0.6")
        ax.scatter(g.effect, y, s=22, zorder=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(contrast)
        ax.set_xlabel("relative improvement")
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, exp, name, per[per.contrast.isin(names)])


def arm_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and arm: mean and sd over folds of the eval metric, mean wall seconds (the README numbers)."""
    ok = df[df.status == "ok"].copy()
    ok["loss"] = [getattr(r, METRIC[r.task]) for r in ok.itertuples()]
    return (ok.groupby(["dataset", "task", "arm"]).agg(n_folds=("fold", "size"), loss_mean=("loss", "mean"),
                                                       loss_sd=("loss", "std"), wall_s_mean=("wall_s", "mean"))
            .reset_index().round(6))


def rounds(models: pd.DataFrame) -> pd.DataFrame:
    """Median boosting rounds of the single-model arms per dataset (AutoGluon caps LightGBM at 10,000)."""
    arms = ["A", "B", "B2", "A_def", "B_def", "A_et0", "B_et0", "A_et0_noreg"]
    m = models[models.arm.isin(arms) & models.num_boost_round.notna()]
    return m.groupby(["dataset", "arm"]).num_boost_round.median().reset_index(name="median_rounds")


def arm_wall(df: pd.DataFrame) -> pd.DataFrame:
    """Per arm over all its ok units: mean, min and max wall seconds."""
    ok = df[df.status == "ok"]
    return (ok.groupby("arm").agg(n_units=("fold", "size"), wall_s_mean=("wall_s", "mean"), wall_s_min=("wall_s", "min"),
                                  wall_s_max=("wall_s", "max")).reset_index().round(3))


def main() -> None:
    df, models = tidy()
    store.write_parquet(RESULTS / "tidy" / "exp02.parquet", df)
    models.to_csv(RESULTS / "tidy" / "exp02_models.csv", index=False)
    per, summ = contrasts(df)
    per.to_csv(RESULTS / "tidy" / "exp02_contrasts.csv", index=False)
    summ = summ[~summ.contrast.str.startswith("A_et0_no")]  # hyperparameter groups ran on 3 datasets: no medians
    summ.to_csv(RESULTS / "tidy" / "exp02_contrasts_summary.csv", index=False)
    arm_means(df).to_csv(RESULTS / "tidy" / "exp02_arm_means.csv", index=False)
    arm_wall(df).to_csv(RESULTS / "tidy" / "exp02_arm_wall.csv", index=False)
    rounds(models).to_csv(RESULTS / "tidy" / "exp02_rounds.csv", index=False)
    if len(per):
        figure(per)
        figure(per, pairs=SINGLE, name="single_model")
    print(f"exp02: {len(df)} units, {int((df.status == 'ok').sum())} ok, {len(per)} contrast rows")


if __name__ == "__main__":
    main()
