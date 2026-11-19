"""exp09 type diagnostic (2026-11-19). Run in the core env after `make exp09`: `python experiments/exp09_numcast/diagnose.py`.

For the five diagnostic datasets (repeat 0, fold 0 fit rows), what each categorical column of the raw frame becomes before it
reaches LightGBM and CatBoost, per arm. A column reaches CatBoost as a categorical feature (`cat_features`) and LightGBM as a
categorical feature exactly when its dtype is a pandas category; any other numeric dtype is a plain number to both
(`afe.models.downstream`; until 2026-11-19 LightGBM skipped ordered categoricals, which are all of them in these frames,
`exp09_ordered.csv`). The arm outputs are the cached exp04 generator outputs (`ag_default`, `skrub_tv`) and the exp09 one
(`base_numcast`), so the table shows what the fits saw.

Writes results/tidy/
- exp09_columns.csv: one row per categorical column (a non-numeric, non-date, non-bool dtype of the raw frame): the number of
  levels in the fit rows, whether every level parses as a number (`numeric_like`), and the type under `base` (the base view),
  `ag_default`, `skrub_tv` (the kind skrub assigned, number of output columns and their dtype) and `base_numcast`;
- exp09_dtypes.csv: per dataset and arm the downstream input width and how many columns are categorical (pandas category: CatBoost
  `cat_features`, LightGBM categorical) or numeric;
- exp09_ordered.csv: for every primary dataset with a categorical column, how many categorical columns of the raw frame are ordered
  (LightGBM's `categorical_feature="auto"` skips ordered pandas categoricals, `lightgbm/basic.py` `cat_cols_not_ordered`;
  the downstream now makes them unordered).
"""

import copy
from pathlib import Path

import pandas as pd
import yaml

from afe.data import splits as S
from afe.data.loaders import load
from afe.generators.base import BaseView
from afe.paths import RESULTS, ROOT
from afe.run import cli, store
from afe.run.unit import digest

CFG4 = yaml.safe_load((ROOT / "experiments" / "exp04_open_benchmark" / "config.yaml").read_text())
CFG9 = yaml.safe_load((ROOT / "experiments" / "exp09_numcast" / "config.yaml").read_text())
DATASETS = CFG9["stages"]["run"]["datasets"]


def cached(cfg: dict, arm: str, dataset: str) -> tuple[pd.DataFrame, dict]:
    """The fit-row generator output and the generator record of repeat 0, fold 0."""
    cfg = copy.deepcopy(cfg)
    cfg["stages"]["diag"] = {"datasets": [dataset], "repeats": [0], "folds": [0], "arms": [arm]}
    (g, _), = cli.plan(cfg, "diag")
    h = digest(g)
    rec = store.read(store.record_path(g["exp"], "gen", h))
    assert rec["status"] == "ok", (dataset, arm)
    return pd.read_parquet(store.cache_dir(g["exp"], h) / "train.parquet"), rec["result"]


def kind(s: pd.Series) -> str:
    return "categorical" if isinstance(s.dtype, pd.CategoricalDtype) else f"numeric ({s.dtype})"


def main() -> None:
    cols, dtypes = [], []
    for ds in DATASETS:
        X, _, _ = load(ds)
        fit = S.rows(S.load(ds), 0, 0)[0]
        Xf = X.iloc[fit]
        view = BaseView().fit(Xf)
        base = view.transform(Xf)
        ag, _ = cached(CFG4, "ag_default", ds)
        sk, sk_rec = cached(CFG4, "skrub_tv", ds)
        nc, nc_rec = cached(CFG9, "base_numcast", ds)
        assert list(nc.columns) == list(base.columns)
        # a skrub output belongs to the longest input column name it starts with (`col` or `col_<level or number>`)
        owner = {o: max((c for c in Xf.columns if o == c or o.startswith(c + "_")), key=len) for o in sk.columns}
        for c, lv in view.levels.items():
            lvs = Xf[c].dropna().astype(str).unique()
            outs = [o for o, w in owner.items() if w == c]
            cols.append({"dataset": ds, "column": c, "raw_dtype": str(Xf[c].dtype), "n_levels": len(lvs),
                         "numeric_like": bool(len(lvs) and pd.to_numeric(pd.Series(lvs), errors="coerce").notna().all()),
                         "ordered_in_base": bool(base[c].cat.ordered), "base": kind(base[c]), "ag_default": kind(ag[c]) if c in ag else "dropped",
                         "skrub_kind": sk_rec["describe"]["column_to_kind"].get(c, "dropped"), "skrub_n_outputs": len(outs),
                         "skrub_dtype": ",".join(sorted({str(sk[o].dtype) for o in outs})),
                         "base_numcast": kind(nc[c])})
        for arm, out in (("base", base), ("ag_default", ag), ("skrub_tv", sk), ("base_numcast", nc)):
            cat = [c for c in out.columns if isinstance(out[c].dtype, pd.CategoricalDtype)]
            dtypes.append({"dataset": ds, "arm": arm, "n_columns_in": X.shape[1], "n_columns_out": out.shape[1],
                           "n_categorical": len(cat), "n_numeric": out.shape[1] - len(cat)})
    ds = pd.read_csv(RESULTS / "datasets.csv")
    ordered = []
    for key in ds[ds.primary & (ds.n_categorical > 0)].key:
        Xk, _, _ = load(key)
        cats = [c for c in Xk.columns if isinstance(Xk[c].dtype, pd.CategoricalDtype)]
        ordered.append({"dataset": key, "n_category_columns": len(cats), "n_ordered": sum(Xk[c].cat.ordered for c in cats)})
    tidy = RESULTS / "tidy"
    pd.DataFrame(ordered).to_csv(tidy / "exp09_ordered.csv", index=False)
    pd.DataFrame(cols).to_csv(tidy / "exp09_columns.csv", index=False)
    pd.DataFrame(dtypes).to_csv(tidy / "exp09_dtypes.csv", index=False)
    print(pd.DataFrame(dtypes).to_string(index=False))


if __name__ == "__main__":
    main()
