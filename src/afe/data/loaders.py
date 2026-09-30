"""Load datasets cached by `make data`. Pandas only, so every env can use it."""

import json

import pandas as pd

from afe.paths import CACHE, RESULTS, ROOT  # noqa: F401  (re-exported)


def load(key: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    meta = json.loads((CACHE / "data" / f"{key}.json").read_text())
    X = pd.read_parquet(CACHE / "data" / f"{key}.parquet")
    y = X.pop(meta["target"])
    return X, y, meta


def encode_y(y: pd.Series, task: str) -> pd.Series:
    """Classification: integer codes of the sorted labels; regression: float."""
    if task == "regression":
        return y.astype(float)
    return pd.Series(pd.factorize(y, sort=True)[0], index=y.index)
