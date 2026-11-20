"""exp05 check 3, kick on LightGBM (2026-11-20), core env, run by hand: `python experiments/exp05_leakage_checks/kick_missing_level.py`.

OpenFE's own transform gives a missing `Combine` pair the code -1, which `openfe_transductive` turns into the level "-1.0"; the
inductive `openfe` arm leaves it missing. LightGBM treats a level differently from missing (CatBoost does not: its downstream code
turns missing into one level in both arms). This refits the five kick units of `openfe_transductive` on LightGBM from the stored
caches twice, as recorded and with "-1.0" removed (back to missing), and compares with the recorded inductive `openfe` AUC.
Writes results/tidy/exp05_transductive_kick_folds.csv (per fold) and exp05_transductive_kick_refit.csv (one row: `effect` is the
mean AUC gain of the transductive arm with the level set back to missing, `effect_as_recorded` with the level, each with a
fold bootstrap 95% CI, 10,000 draws, seed 0, as in `report.py`). The as-recorded refit must reproduce the records.
"""

import numpy as np
import pandas as pd
import yaml

from afe.generators.base import BaseView, combine
from afe.models.downstream import fit_score
from afe.paths import RESULTS, ROOT
from afe.run import cli, store
from afe.run.worker import _data

DATASET, MODEL, METRIC = "kick", "lgbm", "auc"
TRANSDUCTIVE = yaml.safe_load((ROOT / "experiments" / "exp05_leakage_checks" / "transductive.yaml").read_text())
EXP04 = yaml.safe_load((ROOT / "experiments" / "exp04_open_benchmark" / "config.yaml").read_text())


def refit(g, gr, f, drop_level: bool) -> float:
    X, y, meta, idx = _data(g)
    view = BaseView().fit(X.iloc[idx["train"]])
    parts = {}
    for split, rows in idx.items():
        new = pd.read_parquet(store.cache_dir("exp05_transductive", gr["hash"]) / f"{split}.parquet")
        if drop_level:
            for c in new.columns:
                if isinstance(new[c].dtype, pd.CategoricalDtype) and "-1.0" in new[c].cat.categories:
                    new[c] = new[c].cat.remove_categories(["-1.0"])
        new.index = X.index[rows]
        parts[split] = (combine(view.transform(X.iloc[rows]), new), y.iloc[rows])
    return fit_score(f["model"], f["model_params"], f["seed"], meta["task"], int(y.nunique()), *parts.values())["test"][METRIC]


def interval(gains: np.ndarray) -> tuple[float, float]:
    boot = np.random.default_rng(0).choice(gains, size=(10_000, len(gains)), replace=True).mean(axis=1)
    return float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    trans = {g["fold"]: (g, gr, fits) for g, gr, fits in cli.records(TRANSDUCTIVE, "run") if g["dataset"] == DATASET}
    ind = {g["fold"]: fits for g, gr, fits in cli.records(EXP04, "pilot")
           if g["arm"] == "openfe" and g["repeat"] == 0 and g["dataset"] == DATASET}
    rows = []
    for fold in sorted(trans):
        g, gr, fits = trans[fold]
        f, rec = next((f, r) for f, r in fits if f["model"] == MODEL)
        inductive = next(r for f2, r in ind[fold] if f2["model"] == MODEL)["result"]["test"][METRIC]
        recorded = rec["result"]["test"][METRIC]
        as_recorded, no_level = refit(g, gr, f, False), refit(g, gr, f, True)
        assert abs(as_recorded - recorded) < 1e-9, (fold, as_recorded, recorded)
        rows.append({"fold": fold, "inductive": inductive, "transductive_recorded": recorded, "transductive_level_missing": no_level})
    d = pd.DataFrame(rows)
    d["gain_as_recorded"], d["gain_level_missing"] = d.transductive_recorded - d.inductive, d.transductive_level_missing - d.inductive
    d.round(6).to_csv(RESULTS / "tidy" / "exp05_transductive_kick_folds.csv", index=False)
    lo, hi = interval(d.gain_level_missing.to_numpy())
    lo_r, hi_r = interval(d.gain_as_recorded.to_numpy())
    one = pd.DataFrame([{"dataset": DATASET, "model": MODEL, "metric": METRIC, "n_folds": len(d), "effect": d.gain_level_missing.mean(),
                         "ci_lo": lo, "ci_hi": hi, "effect_as_recorded": d.gain_as_recorded.mean(), "ci_lo_as_recorded": lo_r,
                         "ci_hi_as_recorded": hi_r}])
    one.round(6).to_csv(RESULTS / "tidy" / "exp05_transductive_kick_refit.csv", index=False)
    print(d.round(5).to_string(index=False), "\n", one.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
