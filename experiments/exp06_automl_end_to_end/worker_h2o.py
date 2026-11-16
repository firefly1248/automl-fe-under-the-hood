"""exp06 `h2o` and `h2o_te` units (h2o env): H2O AutoML, 600 s, optional target-encoding preprocessing."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")

import h2o
import numpy as np
from h2o.automl import H2OAutoML

from afe.data import splits as S
from afe.data.loaders import encode_y, load
from afe.models.downstream import metrics


def main(task: dict) -> dict:
    from afe.envcheck import ENVS
    from afe.machine import versions

    unit = task["unit"]
    X, y, meta = load(unit["dataset"])
    y = encode_y(y, meta["task"])
    fit, inner, test = S.rows(S.load(unit["dataset"]), unit["repeat"], unit["fold"])
    rows = np.r_[fit, inner]
    h2o.init(nthreads=task["num_cpus"], max_mem_size=f"{task['mem_gb']}G", verbose=False)
    try:
        train = X.iloc[rows].copy()
        train["__label__"] = y.iloc[rows].to_numpy()
        tr = h2o.H2OFrame(train)
        te = h2o.H2OFrame(X.iloc[test])
        if meta["task"] != "regression":
            tr["__label__"] = tr["__label__"].asfactor()
        aml = H2OAutoML(max_runtime_secs=task["time_limit"], seed=S.tool_seed(unit["repeat"], unit["fold"]),
                        sort_metric="logloss" if meta["task"] != "regression" else "RMSE",
                        preprocessing=["target_encoding"] if unit["arm"] == "h2o_te" else None)
        aml.train(y="__label__", training_frame=tr)
        pred = aml.leader.predict(te).as_data_frame(use_multi_thread=True)
        if meta["task"] == "regression":
            p, k = pred["predict"].to_numpy(), 0
        else:
            k = int(y.nunique())
            p = pred[[f"p{i}" for i in range(k)]].to_numpy()
        lb = aml.leaderboard.as_data_frame(use_multi_thread=True)
        return {
            **metrics(meta["task"], y.iloc[test], p, k),
            "n_rows": {"train": len(rows), "test": len(test)},
            "best_model": str(aml.leader.model_id),
            "models": lb.head(30).to_dict("records"),
            "versions": versions(ENVS["h2o"][1] + ["numpy", "pandas", "scikit-learn"]),
        }
    finally:
        h2o.cluster().shutdown()


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(main(task), default=str))
