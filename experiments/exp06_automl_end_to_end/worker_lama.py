"""exp06 `lama` unit (lama env): default `TabularAutoML`, 600 s."""

import json
import sys
from pathlib import Path

import numpy as np

from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task

from afe.data import splits as S
from afe.data.loaders import load
from afe.models.downstream import metrics

LAMA_TASK = {"binary": "binary", "multiclass": "multiclass", "regression": "reg"}


def main(task: dict) -> dict:
    from afe.envcheck import ENVS
    from afe.machine import versions

    unit = task["unit"]
    X, y, meta = load(unit["dataset"])
    fit, inner, test = S.rows(S.load(unit["dataset"]), unit["repeat"], unit["fold"])
    rows = np.r_[fit, inner]
    train = X.iloc[rows].copy()
    train["__target__"] = y.iloc[rows].to_numpy()
    automl = TabularAutoML(task=Task(LAMA_TASK[meta["task"]]), timeout=task["time_limit"],
                           cpu_limit=task["num_cpus"])
    automl.fit_predict(train, roles={"target": "__target__"}, verbose=1)
    pred = automl.predict(X.iloc[test]).data
    if meta["task"] == "regression":
        yt, p, k = y.iloc[test].astype(float), pred[:, 0], 0
    else:
        mapping = automl.reader.class_mapping or {c: i for i, c in enumerate(sorted(y.unique()))}
        yt, k = y.iloc[test].map(mapping), len(mapping)
        p = np.c_[1 - pred[:, 0], pred[:, 0]] if pred.shape[1] == 1 else pred
    return {
        **metrics(meta["task"], yt, p, k),
        "n_rows": {"train": len(rows), "test": len(test)},
        "models": automl.create_model_str_desc(),
        "versions": versions(ENVS["lama"][1] + ["numpy", "pandas", "scikit-learn"]),
    }


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(main(task), default=str))
