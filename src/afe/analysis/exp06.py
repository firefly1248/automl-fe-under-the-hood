"""exp06 tidy table, contrasts and figure (plan section 8, optional).

`python -m afe.analysis.exp06` reads the exp06 run records and writes
- tidy/exp06.parquet: one row per unit (dataset, fold, arm, status, test metrics, best model, wall time);
- tidy/exp06_contrasts.csv and exp06_contrasts_summary.csv: as exp02, on log loss or RMSE: each tool against
  AutoGluon `best_quality`, and H2O with target encoding against H2O without;
- tidy/exp06_lama_diag.csv: pol and socmob, mean RMSE over the 5 folds per LAMA variant next to AutoGluon, H2O, the
  exp02 `A_def` (same folds and outer training rows, less its early-stopping holdout) and the exp04 base LightGBM and CatBoost (same folds, 68% of rows);
- tidy/exp06_arm_means.csv and exp06_arm_wall.csv: per dataset and arm the mean and sd of the metric over folds, wall
  seconds and columns used by LAMA; per arm the wall seconds over all units;
- figures/exp06/contrasts.{svg,png,csv} (the three tool contrasts; the LAMA diagnostic contrasts are in the CSVs).
"""

import pandas as pd

from afe.analysis import exp02
from afe.paths import RESULTS
from afe.run import store

EXP = "exp06_automl_end_to_end"
CONTRASTS = {"lama-ag": ("lama", "ag_best"), "h2o-ag": ("h2o", "ag_best"), "h2o_te-h2o": ("h2o_te", "h2o")}
DIAG = {"lama_num-lama": ("lama_num", "lama"), "lama_full-lama": ("lama_full", "lama"),
        "lama_num_full-lama": ("lama_num_full", "lama"), "lama_num_full-lama_num": ("lama_num_full", "lama_num")}


def tidy() -> pd.DataFrame:
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    rows = []
    for r in exp02.records(EXP):
        u, res = r["unit"], r.get("result") or {}
        rows.append({"dataset": u["dataset"], "task": ds.loc[u["dataset"], "task"], "fold": u["fold"],
                     "arm": u["arm"], "status": r["status"], "wall_s": r["wall_s"],
                     "peak_rss_mb": r["peak_rss_mb"], "logloss": res.get("logloss"), "rmse": res.get("rmse"),
                     "auc": res.get("auc"), "best_model": res.get("best_model"),
                     "n_columns_used": res.get("n_columns_used")})
    return pd.DataFrame(rows).sort_values(["dataset", "arm", "fold"]).reset_index(drop=True)


def lama_diag(df: pd.DataFrame) -> pd.DataFrame:
    """Mean RMSE per dataset and arm on pol and socmob. References: exp02 `A_def` (default LightGBM through
    AutoGluon, same folds and outer training rows, less AutoGluon's early-stopping holdout) and the exp04 base GBMs (repeat 0, same folds, but
    fit on 68% of the rows)."""
    keep = ["pol", "socmob"]
    d = df[df.dataset.isin(keep) & (df.status == "ok")]
    out = d.groupby(["dataset", "arm"]).agg(rmse=("rmse", "mean"), wall_s=("wall_s", "mean"), n_folds=("fold", "size"))
    a = exp02.tidy()[0]
    a = a[a.dataset.isin(keep) & (a.arm == "A_def") & (a.status == "ok")]
    a = a.groupby(["dataset", "arm"]).agg(rmse=("rmse", "mean"), wall_s=("wall_s", "mean"), n_folds=("fold", "size"))
    a.index = a.index.set_levels(["exp02_A_def"], level=1)
    b = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    b = b[b.dataset.isin(keep) & (b.arm == "base") & (b.repeat == 0)]
    base = b.groupby(["dataset", "model"]).agg(rmse=("rmse", "mean"), wall_s=("fit_wall_s", "mean"), n_folds=("fold", "size"))
    base.index = base.index.set_levels("exp04_base_" + base.index.levels[1] + "_68pct", level=1).set_names(["dataset", "arm"])
    return pd.concat([out, a, base]).reset_index().sort_values(["dataset", "rmse"])


def arm_means(df: pd.DataFrame) -> pd.DataFrame:
    """Per dataset and arm: mean and sd over folds of the eval metric and mean wall seconds."""
    ok = df[df.status == "ok"].copy()
    ok["loss"] = [getattr(r, exp02.METRIC[r.task]) for r in ok.itertuples()]
    return (ok.groupby(["dataset", "task", "arm"]).agg(n_folds=("fold", "size"), loss_mean=("loss", "mean"),
                                                       loss_sd=("loss", "std"), wall_s_mean=("wall_s", "mean"),
                                                       wall_s_min=("wall_s", "min"), wall_s_max=("wall_s", "max"),
                                                       n_columns_used_min=("n_columns_used", "min"),
                                                       n_columns_used_max=("n_columns_used", "max"))
            .reset_index().round(6))


def main() -> None:
    df = tidy()
    store.write_parquet(RESULTS / "tidy" / "exp06.parquet", df)
    per, summ = exp02.contrasts(df, CONTRASTS | DIAG)
    per.to_csv(RESULTS / "tidy" / "exp06_contrasts.csv", index=False)
    summ = summ[~summ.contrast.isin(DIAG)]  # diagnostics ran on 2 datasets: no medians
    summ.to_csv(RESULTS / "tidy" / "exp06_contrasts_summary.csv", index=False)
    lama_diag(df).round(4).to_csv(RESULTS / "tidy" / "exp06_lama_diag.csv", index=False)
    arm_means(df).to_csv(RESULTS / "tidy" / "exp06_arm_means.csv", index=False)
    exp02.arm_wall(df).to_csv(RESULTS / "tidy" / "exp06_arm_wall.csv", index=False)
    if len(per):
        exp02.figure(per, "exp06", CONTRASTS)
    print(f"exp06: {len(df)} units, {int((df.status == 'ok').sum())} ok, {len(per)} contrast rows")


if __name__ == "__main__":
    main()
