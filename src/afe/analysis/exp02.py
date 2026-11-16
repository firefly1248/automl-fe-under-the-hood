"""exp02 tidy table, contrasts and figure (plan section 8).

`python -m afe.analysis.exp02` reads the exp02 run records and writes
- tidy/exp02.parquet: one row per unit (dataset, fold, arm, status, test metrics, models trained, fit time);
- tidy/exp02_models.csv: per unit and trained model, its stack level, validation score and fit time;
- tidy/exp02_contrasts.csv: per dataset and contrast the mean relative improvement over folds (positive
  favours the first arm) with a fold bootstrap 95% CI, on the eval metric (log loss or RMSE);
- tidy/exp02_contrasts_summary.csv: median over datasets within task type (4 per type, descriptive);
- figures/exp02/contrasts.{svg,png,csv}.
Only folds where both arms of a contrast are ok enter it.
"""

import gzip
import json

import numpy as np
import pandas as pd

from afe.analysis.figures import save
from afe.paths import RESULTS
from afe.run import store

EXP = "exp02_autogluon_ablation"
METRIC = {"binary": "logloss", "multiclass": "logloss", "regression": "rmse"}
CONTRASTS = {  # name -> (arm, reference); improvement of arm over reference
    "B-A": ("B", "A"), "D-A": ("D", "A"), "D-D0": ("D", "D0"), "E-D0": ("E", "D0"), "B-B2": ("B", "B2"),
}
N_BOOT = 10_000


def records(exp=EXP):
    raw = RESULTS / "raw" / exp
    for p in sorted((raw / "fit").glob("*.json")):
        yield json.loads(p.read_text())
    gz = raw / "fit.jsonl.gz"
    if gz.exists():
        for line in gzip.open(gz, "rt"):
            yield json.loads(line)


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
                    "n_stack_levels": len({v["stack_level"] for v in m.values() if v["stack_level"]}),
                    "sum_fit_s": sum(v["fit_time"] for v in m.values())}
            models += [{"dataset": u["dataset"], "fold": u["fold"], "arm": u["arm"], "model": k,
                        "stack_level": v["stack_level"], "val_score": v["val_score"], "fit_time": v["fit_time"],
                        "num_features": v["num_features"]} for k, v in m.items()]
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["dataset", "arm", "fold"]).reset_index(drop=True)
    return df, pd.DataFrame(models)


def contrasts(df: pd.DataFrame, pairs=CONTRASTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = df[df.status == "ok"].copy()
    ok["loss"] = [getattr(r, METRIC[r.task]) for r in ok.itertuples()]
    wide = ok.pivot_table(index=["dataset", "task", "fold"], columns="arm", values="loss").reset_index()
    rng = np.random.default_rng(0)
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
    summ = (per.groupby(["contrast", "task"]).effect
            .agg(median="median", n_datasets="size", n_better=lambda v: int((v > 0).sum())).reset_index())
    return per, summ


def figure(per: pd.DataFrame, exp="exp02", pairs=CONTRASTS) -> None:
    import matplotlib.pyplot as plt

    names = [c for c in pairs if c in set(per.contrast)]
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 6), sharey=True)
    axes = np.atleast_1d(axes)
    order = per.drop_duplicates("dataset").sort_values(["task", "dataset"]).dataset.tolist()
    for ax, name in zip(axes, names):
        g = per[per.contrast == name].set_index("dataset").reindex(order)
        y = np.arange(len(order))
        ax.hlines(y, g.ci_lo, g.ci_hi, color="0.6")
        ax.scatter(g.effect, y, s=22, zorder=3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(name)
        ax.set_xlabel("relative improvement")
    axes[0].set_yticks(np.arange(len(order)), order)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, exp, "contrasts", per)


def main() -> None:
    df, models = tidy()
    store.write_parquet(RESULTS / "tidy" / "exp02.parquet", df)
    models.to_csv(RESULTS / "tidy" / "exp02_models.csv", index=False)
    per, summ = contrasts(df)
    per.to_csv(RESULTS / "tidy" / "exp02_contrasts.csv", index=False)
    summ.to_csv(RESULTS / "tidy" / "exp02_contrasts_summary.csv", index=False)
    if len(per):
        figure(per)
    print(f"exp02: {len(df)} units, {int((df.status == 'ok').sum())} ok, {len(per)} contrast rows")


if __name__ == "__main__":
    main()
