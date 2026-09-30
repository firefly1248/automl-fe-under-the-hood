import numpy as np
import pandas as pd
import pytest

from afe.models.downstream import fit_score

PARAMS = {"lgbm": {"n_estimators": 50, "learning_rate": 0.1, "early_stopping": 10},
          "catboost": {"iterations": 50, "learning_rate": 0.1, "early_stopping": 10}}


def _data(task, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "c": pd.Categorical(rng.choice(list("xyz"), n))})
    s = X.a + (X.c == "x") * 1.0
    y = {"regression": s, "binary": (s > 0.3).astype(int), "multiclass": pd.cut(s, 3, labels=False)}[task]
    return X, pd.Series(y)


@pytest.mark.parametrize("model", ["lgbm", "catboost"])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_fit_score(model, task):
    X, y = _data(task)
    k = 0 if task == "regression" else int(y.nunique())
    parts = [(X.iloc[a:b], y.iloc[a:b]) for a, b in ((0, 250), (250, 320), (320, 400))]
    res = fit_score(model, PARAMS[model], 0, task, k, *parts)
    key = {"regression": "r2", "binary": "auc", "multiclass": "accuracy"}[task]
    assert res["test"][key] > 0.5 and res["best_iteration"] >= 0


def test_missing_class_in_fit_rows():
    X, y = _data("multiclass")
    fit = y != 2  # class 2 absent from fit rows
    res = fit_score("lgbm", PARAMS["lgbm"], 0, "multiclass", 3, (X[fit], y[fit]), (X[fit], y[fit]), (X, y))
    assert np.isfinite(res["test"]["logloss"])
