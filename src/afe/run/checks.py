"""Sanity check on a finished AutoML unit, applied by the exp02 and exp06 orchestrators (not by the workers, whose
files are hashed into the unit records).

A bagged AutoGluon fit in a directory outside the project made every Ray-based model (LightGBM, CatBoost, XGBoost,
neural nets) fail at `ray.init`; AutoGluon logged a warning, skipped them, and the unit ended "ok" with random forests
and extra trees only (10 cmc units of exp02 arms D and D0). A unit whose result lists no boosted-tree or neural model is
therefore marked failed.
"""

import re

FAMILY = re.compile(r"LightGBM|LGBM|CatBoost|XGBoost|NeuralNet|DeepLearning|^GBM|NN_|Torch", re.IGNORECASE)


def trained_names(models) -> list[str]:
    """Model names from the three result formats: AutoGluon dict, H2O list of dicts, LightAutoML description text."""
    if isinstance(models, dict):
        return list(models)
    if isinstance(models, list):
        return [m["model_id"] for m in models]
    return [str(models)]


def has_boosted_or_nn(models) -> bool:
    return any(FAMILY.search(n) for n in trained_names(models))


def verdict(result: dict | None) -> str | None:
    """None if fine, else the error text for a failed unit."""
    if result and not has_boosted_or_nn(result.get("models")):
        return "no GBM, CatBoost, XGBoost or neural-net model in the result"
    return None
