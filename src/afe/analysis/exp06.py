"""exp06 tidy table, contrasts and figure (plan section 8, optional).

`python -m afe.analysis.exp06` reads the exp06 run records and writes
- tidy/exp06.parquet: one row per unit (dataset, fold, arm, status, test metrics, best model, wall time);
- tidy/exp06_contrasts.csv and exp06_contrasts_summary.csv: as exp02, on log loss or RMSE: each tool against
  AutoGluon `best_quality`, and H2O with target encoding against H2O without;
- figures/exp06/contrasts.{svg,png,csv}.
"""

import pandas as pd

from afe.analysis import exp02
from afe.paths import RESULTS
from afe.run import store

EXP = "exp06_automl_end_to_end"
CONTRASTS = {"lama-ag": ("lama", "ag_best"), "h2o-ag": ("h2o", "ag_best"), "h2o_te-h2o": ("h2o_te", "h2o")}


def tidy() -> pd.DataFrame:
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    rows = []
    for r in exp02.records(EXP):
        u, res = r["unit"], r.get("result") or {}
        rows.append({"dataset": u["dataset"], "task": ds.loc[u["dataset"], "task"], "fold": u["fold"],
                     "arm": u["arm"], "status": r["status"], "wall_s": r["wall_s"],
                     "peak_rss_mb": r["peak_rss_mb"], "logloss": res.get("logloss"), "rmse": res.get("rmse"),
                     "auc": res.get("auc"), "best_model": res.get("best_model")})
    return pd.DataFrame(rows).sort_values(["dataset", "arm", "fold"]).reset_index(drop=True)


def main() -> None:
    df = tidy()
    store.write_parquet(RESULTS / "tidy" / "exp06.parquet", df)
    per, summ = exp02.contrasts(df, CONTRASTS)
    per.to_csv(RESULTS / "tidy" / "exp06_contrasts.csv", index=False)
    summ.to_csv(RESULTS / "tidy" / "exp06_contrasts_summary.csv", index=False)
    if len(per):
        exp02.figure(per, "exp06", CONTRASTS)
    print(f"exp06: {len(df)} units, {int((df.status == 'ok').sum())} ok, {len(per)} contrast rows")


if __name__ == "__main__":
    main()
