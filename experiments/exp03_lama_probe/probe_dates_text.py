"""exp03 gap probe: how default `TabularAutoML` 0.4.2 treats datetime and text columns.

Tiny synthetic frame (3,000 rows). Four probes, one CSV (`results/tidy/exp03_dates_text.csv`, columns
probe, item, value):
- reader_roles: the role the default `Reader` gives each column;
- fit_default: `TabularAutoML` default pipelines (lgb -> LGBAdvancedPipeline, linear_l2 -> LinearFeatures),
  the features each model received that come from the date and text columns, and what the selector kept;
- fit_manual_roles: the same with a manual `text` role and a `base_date` datetime role;
- nlp_extras: what happens to `TabularNLPAutoML` when the `nlp` extras are not installed.

Run in the lama env: `uv run --frozen --project envs/lama python experiments/exp03_lama_probe/probe_dates_text.py`.
"""

import os

os.environ["OMP_NUM_THREADS"] = "2"  # before torch/LightGBM load (two libomp copies on macOS)

import importlib.util
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.dataset.roles import DatetimeRole
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task

from afe.paths import RESULTS

N, SEED, CPU, TIMEOUT = 3000, 0, 2, 60
TARGET = "__target__"
WORDS = "the a cat dog market price good bad service late fast slow order item arrived quality broken great".split()
rows = []


def add(probe, item, value):
    rows.append({"probe": probe, "item": item, "value": str(value)})
    print(probe, item, value, flush=True)


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    start = pd.Timestamp("2020-01-01")
    d1 = start + pd.to_timedelta(rng.integers(0, 1460, N), unit="D")
    d2 = start + pd.to_timedelta(rng.integers(0, 1460, N), unit="D")
    text = [" ".join(rng.choice(WORDS, rng.integers(6, 11))) for _ in range(N)]
    num = rng.normal(size=N)
    winter = np.isin(d1.month, [11, 12, 1]).astype(float)
    good = np.array(["good" in t.split() for t in text], dtype=float)
    logit = 1.5 * winter + 1.5 * good + 0.5 * num - 1.0
    y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "num": num,
        "dt_str": d1.strftime("%Y-%m-%d"),  # ISO string
        "dt_ts": pd.Series(d2),  # datetime64[ns]
        "date_int": d1.strftime("%Y%m%d").astype(int),  # YYYYMMDD integer
        "cat": rng.choice(list("abcdef"), N),
        "text_long": text,  # about 8 words, nearly all values distinct
        TARGET: y,
    })


def derived(features, cols) -> list:
    return sorted(f for f in features if any(f.endswith(f"__{c}") or f == c or f"__{c}__" in f for c in cols))


def fit(df, roles, label):
    automl = TabularAutoML(task=Task("binary"), cpu_limit=CPU, timeout=TIMEOUT,
                           general_params={"use_algos": [["lgb", "linear_l2"]]})
    try:
        automl.fit_predict(df, roles=roles, verbose=0)
    except Exception as e:  # recorded as a result
        add(label, "exception", f"{type(e).__name__}: {str(e)[:300]}")
        return
    add(label, "reader_roles", "; ".join(f"{c}={r!r}" for c, r in automl.reader.roles.items()))
    add(label, "reader_dropped", sorted(c for c in automl.reader.dropped_features if c != TARGET))
    for pipe in automl.levels[0]:
        sel = getattr(pipe, "pre_selection", None)
        if sel is not None and sel.is_fitted:
            add(label, "selector_selected", sorted(sel.selected_features))
            add(label, "selector_dropped", sorted(sel.dropped_features))
        for algo in pipe.ml_algos:
            feats = list(algo.features)
            add(label, f"{algo.name}.n_features", len(feats))
            add(label, f"{algo.name}.from_date_text_columns",
                derived(feats, ["dt_str", "dt_ts", "date_int", "text_long"]) or "none")
            add(label, f"{algo.name}.all_features", "; ".join(sorted(feats)))


def main() -> None:
    df = frame()
    reader = PandasToPandasReader(Task("binary"), **TabularAutoML(task=Task("binary")).reader_params | {"n_jobs": 1})
    reader.fit_read(df, roles={"target": TARGET})
    for c, r in reader.roles.items():
        add("reader_roles", c, repr(r))
    add("reader_roles", "dropped", sorted(c for c in reader.dropped_features if c != TARGET))

    fit(df, {"target": TARGET}, "fit_default")
    fit(df, {"target": TARGET, "text": ["text_long"],
             DatetimeRole(np.datetime64, base_date=True): ["dt_str"], "datetime": ["dt_ts"]}, "fit_manual_roles")

    nlp = ["transformers", "nltk", "fasttext", "gensim", "sentence_transformers"]
    add("nlp_extras", "installed", {m: importlib.util.find_spec(m) is not None for m in nlp})
    logging.disable(logging.CRITICAL)
    try:
        from lightautoml.automl.presets.text_presets import TabularNLPAutoML  # noqa: F401
        add("nlp_extras", "import TabularNLPAutoML", "ok")
    except Exception as e:
        add("nlp_extras", "import TabularNLPAutoML", f"{type(e).__name__}: {str(e)[:200]}")

    out = RESULTS / "tidy" / "exp03_dates_text.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
