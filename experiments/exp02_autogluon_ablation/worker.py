"""One exp02 unit (autogluon env): `python worker.py <task.json>`.

Each arm fits a TabularPredictor on the full outer train (fit and inner rows) and validates as AutoGluon
does by default (holdout or bagging); the test fold is scored with the exp04 metrics. Arms (plan section 8):
- A:  `GBM` with the `_r13` hyperparameters and no `ag.prep_params`, no bagging;
- B:  `GBM_PREP` `_r13`, no bagging (B - A = new features plus TE replacing native categoricals);
- B2: `GBM_PREP` `_r21` (the portfolio's own no-prep control);
- C:  `medium_quality`;
- D:  `best_quality` with `dynamic_stacking=False` (2023 portfolio, stacking; DyStack off so D and D0
      both fit models for the full 600 s);
- D0: `best_quality` with `num_stack_levels=0` and `dynamic_stacking=False` (separates stacking from portfolio);
- E:  `best_quality_v150` (2025-12-18 portfolio with `GBM_PREP`, no stacking, early-stop callback).
"""

import json
import sys
from pathlib import Path

import numpy as np
from autogluon.tabular import TabularPredictor

from afe.data import splits as S
from afe.data.loaders import encode_y, load
from afe.generators.ag_tabprep import prep_config
from afe.models.downstream import metrics

LABEL = "__label__"


def arm_kwargs(arm: str) -> dict:
    if arm in ("A", "B", "B2"):
        cfg = prep_config("_r21" if arm == "B2" else "_r13")
        if arm == "A":
            cfg = {k: v for k, v in cfg.items() if not k.startswith("ag.prep_params")}
        return {"hyperparameters": {("GBM" if arm == "A" else "GBM_PREP"): [cfg]}}
    return {
        "C": {"presets": "medium_quality"},
        "D": {"presets": "best_quality", "dynamic_stacking": False},
        "D0": {"presets": "best_quality", "dynamic_stacking": False, "num_stack_levels": 0},
        "E": {"presets": "best_quality_v150"},
    }[arm]


def main(task: dict) -> dict:
    from afe.envcheck import ENVS
    from afe.machine import versions

    unit = task["unit"]
    X, y, meta = load(unit["dataset"])
    y = encode_y(y, meta["task"])
    fit, inner, test = S.rows(S.load(unit["dataset"]), unit["repeat"], unit["fold"])
    rows = np.r_[fit, inner]
    train = X.iloc[rows].copy()
    train[LABEL] = y.iloc[rows].to_numpy()
    pred = TabularPredictor(label=LABEL, problem_type=meta["task"], eval_metric=task["eval_metric"][meta["task"]],
                            path=task["ag_path"], verbosity=2)
    pred.fit(train, time_limit=task["time_limit"], num_cpus=task["num_cpus"], **arm_kwargs(unit["arm"]))
    if meta["task"] == "regression":
        p, k = pred.predict(X.iloc[test]).to_numpy(), 0
    else:
        pp = pred.predict_proba(X.iloc[test])
        p, k = pp[sorted(pp.columns)].to_numpy(), int(y.nunique())
    info = pred.info()["model_info"]
    models = {m: {"val_score": i.get("val_score"), "fit_time": i.get("fit_time"), "stack_level": i.get("stack_level"),
                  "num_features": len(i.get("features") or []), "hyperparameters": i.get("hyperparameters"),
                  "hyperparameters_fit": i.get("hyperparameters_fit")} for m, i in info.items()}
    return {
        **metrics(meta["task"], y.iloc[test], p, k),
        "eval_metric": task["eval_metric"][meta["task"]],
        "n_rows": {"train": len(rows), "test": len(test)},
        "n_features_in": X.shape[1],
        "best_model": pred.model_best,
        "models": models,
        "versions": versions(ENVS["autogluon"][1] + ["numpy", "pandas", "scikit-learn"]),
    }


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(main(task), default=str))
