"""exp06 `ag_best` unit (autogluon env): AutoGluon `best_quality` as shipped, 600 s."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor

from afe.data import splits as S
from afe.data.loaders import encode_y, load
from afe.models.downstream import metrics

LABEL = "__label__"


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
    pred.fit(train, time_limit=task["time_limit"], num_cpus=task["num_cpus"], presets="best_quality")
    # fastai sets pandas chained_assignment to "raise" when its model is fitted; AutoGluon's XGBoost preprocess
    # assigns into a slice of its own, which raises. Assignments behave the same as with "warn".
    pd.options.mode.chained_assignment = None
    if meta["task"] == "regression":
        p, k = pred.predict(X.iloc[test]).to_numpy(), 0
    else:
        pp = pred.predict_proba(X.iloc[test])
        p, k = pp[sorted(pp.columns)].to_numpy(), int(y.nunique())
    lb = pred.leaderboard(display=False)
    return {
        **metrics(meta["task"], y.iloc[test], p, k),
        "n_rows": {"train": len(rows), "test": len(test)},
        "best_model": pred.model_best,
        "models": {r.model: {"val_score": r.score_val, "fit_time": r.fit_time, "stack_level": r.stack_level}
                   for r in lb.itertuples()},
        "versions": versions(ENVS["autogluon"][1] + ["numpy", "pandas", "scikit-learn"]),
    }


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(main(task), default=str))
