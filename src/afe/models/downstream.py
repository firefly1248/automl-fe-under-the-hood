"""Downstream LightGBM and CatBoost (plan section 7.3), identical for every arm. Runs in the core env.

Trains on fit rows, early-stops on inner, scores on test. Untuned defaults by design.
"""

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, r2_score, roc_auc_score


def _full_proba(classes, p: np.ndarray, k: int) -> np.ndarray:
    """Probabilities for all k classes (a class missing from the fit rows gets 0)."""
    out = np.zeros((len(p), k))
    out[:, np.asarray(classes, dtype=int)] = p
    return out


def metrics(task: str, y: pd.Series, pred: np.ndarray, k: int) -> dict:
    if task == "regression":
        return {"rmse": float(np.sqrt(mean_squared_error(y, pred))), "r2": float(r2_score(y, pred))}
    out = {"logloss": float(log_loss(y, pred, labels=list(range(k))))}
    if task == "binary":
        out["auc"] = float(roc_auc_score(y, pred[:, 1]))
    else:
        out["accuracy"] = float(accuracy_score(y, pred.argmax(axis=1)))
    return out


def _lgbm(params, seed, task, k, X_fit, y_fit, X_inner, y_inner):
    import lightgbm as lgb

    names = {c: f"f{i}" for i, c in enumerate(X_fit.columns)}  # LightGBM rejects JSON characters in names
    kw = {n: v for n, v in params.items() if n != "early_stopping"}
    cls = lgb.LGBMRegressor if task == "regression" else lgb.LGBMClassifier
    m = cls(**kw, random_state=seed, n_jobs=1, verbose=-1)
    m.fit(X_fit.rename(columns=names), y_fit, eval_X=X_inner.rename(columns=names), eval_y=y_inner,
          callbacks=[lgb.early_stopping(params["early_stopping"], verbose=False)])

    def predict(X):
        X = X.rename(columns=names)
        return m.predict(X) if task == "regression" else _full_proba(m.classes_, m.predict_proba(X), k)

    return predict, int(m.best_iteration_)


def _catboost(params, seed, task, k, X_fit, y_fit, X_inner, y_inner):
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    cat = [c for c in X_fit.columns if isinstance(X_fit[c].dtype, pd.CategoricalDtype)]

    def pool(X, y=None):
        X = X.astype({c: object for c in cat})
        for c in cat:
            X[c] = X[c].where(X[c].notna(), "__nan__").astype(str)
        return Pool(X, y, cat_features=cat)

    loss = {"regression": "RMSE", "binary": "Logloss", "multiclass": "MultiClass"}[task]
    cls = CatBoostRegressor if task == "regression" else CatBoostClassifier
    m = cls(iterations=params["iterations"], learning_rate=params["learning_rate"], loss_function=loss,
            eval_metric=loss, od_type="Iter", od_wait=params["early_stopping"], use_best_model=True,
            random_seed=seed, thread_count=1, verbose=0, allow_writing_files=False)
    m.fit(pool(X_fit, y_fit), eval_set=pool(X_inner, y_inner))

    def predict(X):
        return m.predict(pool(X)) if task == "regression" else _full_proba(m.classes_, m.predict_proba(pool(X)), k)

    return predict, int(m.get_best_iteration())


MODELS = {"lgbm": _lgbm, "catboost": _catboost}


def fit_score(model: str, params: dict, seed: int, task: str, k: int, fit, inner, test) -> dict:
    """fit/inner/test: (X, y) pairs. k: number of classes in the full dataset (0 for regression)."""
    t0 = time.perf_counter()
    predict, best = MODELS[model](params, seed, task, k, *fit, *inner)
    fit_s = time.perf_counter() - t0
    inner_m = metrics(task, inner[1], predict(inner[0]), k)
    return {
        "test": metrics(task, test[1], predict(test[0]), k),
        "inner": inner_m["rmse" if task == "regression" else "logloss"],
        "best_iteration": best,
        "fit_s": round(fit_s, 3),
    }
