"""exp06 LAMA diagnostic units (lama env), regression datasets only: where `lama` trails the base GBM.

Same unit, folds and scoring as `worker_lama.py`; the arms change one thing each:
- lama_num:      every column gets the Numeric role (category columns enter as integer codes), so the reader's
                 own role guessing, category encodings and column drops do not apply;
- lama_full:     `TabularUtilizedAutoML` (LightAutoML's preset that keeps training until the budget is spent)
                 with the reader's default roles;
- lama_num_full: both.
"""

import json
import sys
from pathlib import Path

import numpy as np

from lightautoml.automl.presets.tabular_presets import TabularAutoML, TabularUtilizedAutoML
from lightautoml.tasks import Task

from afe.data import splits as S
from afe.data.loaders import load
from afe.models.downstream import metrics

ARMS = {"lama_num": (TabularAutoML, True), "lama_full": (TabularUtilizedAutoML, False),
        "lama_num_full": (TabularUtilizedAutoML, True)}


def main(task: dict) -> dict:
    from afe.envcheck import ENVS
    from afe.machine import versions

    unit = task["unit"]
    X, y, meta = load(unit["dataset"])
    assert meta["task"] == "regression"
    cls, numeric = ARMS[unit["arm"]]
    if numeric:
        X = X.apply(lambda c: c.cat.codes.where(c.cat.codes >= 0) if c.dtype.name == "category" else c)
    fit, inner, test = S.rows(S.load(unit["dataset"]), unit["repeat"], unit["fold"])
    rows = np.r_[fit, inner]
    train = X.iloc[rows].copy()
    train["__target__"] = y.iloc[rows].to_numpy()
    roles = {"target": "__target__"} | ({"numeric": list(X.columns)} if numeric else {})
    automl = cls(task=Task("reg"), timeout=task["time_limit"], cpu_limit=task["num_cpus"])
    automl.fit_predict(train, roles=roles, verbose=1)
    pred = automl.predict(X.iloc[test]).data
    return {
        **metrics("regression", y.iloc[test].astype(float), pred[:, 0], 0),
        "n_rows": {"train": len(rows), "test": len(test)},
        "n_columns_in": X.shape[1],
        "n_columns_used": len(automl.reader.used_features) if hasattr(automl, "reader") else None,
        "models": automl.create_model_str_desc() if hasattr(automl, "create_model_str_desc") else None,
        "versions": versions(ENVS["lama"][1] + ["numpy", "pandas", "scikit-learn"]),
    }


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(main(task), default=str))
