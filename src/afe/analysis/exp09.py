"""exp09 (2026-11-19): do `ag_default` and `skrub_tv` help CatBoost on fps_benchmark, auction_verification and
Brazilian_houses because they turn number-like categorical columns into numbers? `base_numcast` is `base` with exactly that
cast and nothing else. `base_codes` is `base` with every categorical column as integer codes.

`python -m afe.analysis.exp09` reads the exp09 records and the exp04 table and writes tidy/exp09_effects.csv: per dataset,
downstream model and arm (`ag_default`, `skrub_tv`, `base_numcast`), over the splits where the arm and `base` are ok, the mean
effect over `base` in the task's primary metric, defined as in `exp04_effects.csv` and `afe.analysis.cat_subgroup` (AUC
difference of the means for binary, relative log loss or RMSE reduction of the means otherwise), with a paired bootstrap 95%
CI over the splits. One random stream, sorted order, append arms.

It also reads the by-hand fps_benchmark probe `tidy/exp09_onehot_probe.csv` (`experiments/exp09_numcast/probe_onehot.py`: per split
the CatBoost RMSE of `base`, of `base` with `one_hot_max_size=255` and of `base_codes`) and writes tidy/exp09_onehot_probe_effects.csv:
for the `onehot255` arm the effect over `base` (reduction of the mean RMSE, as above) with its CI (random stream seeded 0), and the
smallest and largest per-split RMSE of the arm and of `base`. `base_codes` is not repeated here: its effect and CI are in exp09_effects.csv.
"""

import numpy as np
import pandas as pd
import yaml

from afe.analysis.cat_subgroup import METRIC, effect_ci
from afe.paths import RESULTS, ROOT
from afe.run import cli

CFG = yaml.safe_load((ROOT / "experiments" / "exp09_numcast" / "config.yaml").read_text())
ARMS = ["ag_default", "skrub_tv", "base_numcast", "base_codes"]
METRICS = ["auc", "logloss", "rmse"]


def tidy() -> pd.DataFrame:
    rows = []
    for g, grec, fits in cli.records(CFG, "run"):
        for f, frec in fits:
            if frec and frec["status"] == "ok":
                rows.append({"dataset": g["dataset"], "repeat": g["repeat"], "fold": g["fold"], "arm": f["arm"],
                             "model": f["model"], **{m: frec["result"]["test"].get(m) for m in METRICS}})
    return pd.DataFrame(rows)


def effects(new: pd.DataFrame, exp04: pd.DataFrame) -> pd.DataFrame:
    key = ["dataset", "repeat", "fold", "model"]
    ok = exp04[exp04.dataset.isin(CFG["stages"]["run"]["datasets"]) & (exp04.status == "ok")]
    base = ok[ok.arm == "base"][[*key, "task", *METRICS]].rename(columns={m: f"base_{m}" for m in METRICS})
    arms = pd.concat([ok[ok.arm.isin(["ag_default", "skrub_tv"])][[*key, "arm", *METRICS]], new])
    d = arms.merge(base, on=key)
    rng = np.random.default_rng(0)
    rows = []
    for (dataset, model, arm), g in d.sort_values(["dataset", "model", "arm"]).groupby(["dataset", "model", "arm"], sort=False):
        task = g.task.iloc[0]
        m = METRIC[task]
        e, lo, hi = effect_ci(g[m].to_numpy(), g[f"base_{m}"].to_numpy(), task, rng)
        rows.append({"dataset": dataset, "task": task, "model": model, "arm": arm, "n_splits": len(g), "effect": e,
                     "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def onehot_probe_effects(probe: pd.DataFrame) -> pd.DataFrame:
    e, lo, hi = effect_ci(probe["onehot255"].to_numpy(), probe["base"].to_numpy(), "regression", np.random.default_rng(0))
    return pd.DataFrame([{"arm": "onehot255", "n_splits": len(probe), "effect": e, "ci_lo": lo, "ci_hi": hi,
                          "rmse_min": probe.onehot255.min(), "rmse_max": probe.onehot255.max(),
                          "base_rmse_min": probe.base.min(), "base_rmse_max": probe.base.max()}])


def main() -> None:
    new = tidy()
    if new.empty:
        return print("exp09: no records yet (make exp09)")
    eff = effects(new, pd.read_parquet(RESULTS / "tidy" / "exp04.parquet"))
    eff.round(6).to_csv(RESULTS / "tidy" / "exp09_effects.csv", index=False)
    onehot_probe_effects(pd.read_csv(RESULTS / "tidy" / "exp09_onehot_probe.csv")).round(6).to_csv(
        RESULTS / "tidy" / "exp09_onehot_probe_effects.csv", index=False)
    print(f"exp09: {len(new)} ok fit units, {len(eff)} rows")


if __name__ == "__main__":
    main()
