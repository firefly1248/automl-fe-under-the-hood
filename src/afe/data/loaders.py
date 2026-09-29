"""Load datasets cached by `make data`. Pandas only, so every env can use it."""

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
CACHE = Path(os.environ.get("AFE_CACHE", "~/.cache/afe")).expanduser()


def load(key: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    meta = json.loads((CACHE / "data" / f"{key}.json").read_text())
    X = pd.read_parquet(CACHE / "data" / f"{key}.parquet")
    y = X.pop(meta["target"])
    return X, y, meta
