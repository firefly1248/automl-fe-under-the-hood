"""exp06 gap probe: what H2O AutoML 3.46 does to dates, strings and categoricals.

Tiny synthetic frame (3,000 rows, binary target) with a pandas datetime column, text, a free-text column and
categoricals of 4, 20 to 25 and 60 levels. Two `H2OAutoML` runs (default, then
`preprocessing=["target_encoding"]`), 45 s each, `nthreads=2`. Writes `results/tidy/exp06_h2o_probe.csv`
(columns probe, item, value):
- frame_types: the column types H2O guesses on upload (`H2OFrame` from the pandas frame);
- <run>: for every model of the leaderboard, `categorical_encoding` (default / input / actual from the model
  parameters; "absent" when the algorithm has no such parameter) and the predictor columns it was trained on;
- <run>.te_model: the parameters the AutoML target-encoding step used (actual value, default in brackets) and the
  columns it encoded.

Run in the h2o env: `JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run --frozen --project envs/h2o python
experiments/exp06_automl_end_to_end/probe_h2o.py`.
"""

import os

os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")

import re

import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML

from afe.paths import RESULTS

N, SEED, RUNTIME = 3000, 0, 45
rows = []


def add(probe, item, value):
    rows.append({"probe": probe, "item": item, "value": str(value)})
    print(probe, item, value, flush=True)


def frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"num": rng.normal(size=N)})
    for k in (4, 20, 21, 22, 23, 24, 25, 60):
        df[f"cat_{k}"] = rng.choice([f"c{i}" for i in range(k)], N)
    df["dt"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 1460, N), unit="D")
    words = ["a", "b", "good", "bad", "c", "d", "e"]
    df["text_free"] = [" ".join(rng.choice(words, 8)) for _ in range(N)]  # nearly all values distinct
    df["text_few"] = rng.choice([" ".join(rng.choice(words, 8)) for _ in range(5)], N)  # 5 distinct sentences
    df["y"] = (rng.random(N) < 0.5).astype(int)
    return df


def run(name, train, **kw):
    aml = H2OAutoML(max_runtime_secs=RUNTIME, seed=SEED, **kw)
    aml.train(y="y", training_frame=train)
    ids = aml.leaderboard.as_data_frame(use_multi_thread=True).model_id.tolist()
    for mid in ids:
        m = h2o.get_model(mid)
        label = re.sub(r"_AutoML_\d+_\d{8}_\d{6}", "", mid)
        enc = m.params.get("categorical_encoding")
        enc = f"default={enc['default']}, input={enc['input']}, actual={enc['actual']}" if enc else "absent"
        preds = [c for c in m._model_json["output"]["names"] if c not in ("y", "y_te_fold")]
        add(name, f"{label}.categorical_encoding", f"{m.algo}: {enc}")
        add(name, f"{label}.predictors", "; ".join(preds))
    for key in h2o.ls().iloc[:, 0].tolist():
        if key.startswith("TargetEncoding") and not key.endswith(("_preprocessor",)) and "_encodings_" not in key:
            te = h2o.get_model(key)
            p = te.params
            for a in ("blending", "inflection_point", "smoothing", "noise", "data_leakage_handling",
                      "keep_original_categorical_columns", "seed"):
                add(f"{name}.te_model", a, f"{p[a]['actual']} (default {p[a]['default']})")
            add(f"{name}.te_model", "fold_column", p["fold_column"]["actual"]["column_name"])
            add(f"{name}.te_model", "columns_encoded",
                "; ".join(c for c in te._model_json["output"]["names"] if c not in ("y", "y_te_fold")))


def main() -> None:
    h2o.init(nthreads=2, max_mem_size="2G", ip="127.0.0.1", port=54433, verbose=False)
    try:
        df = frame()
        train = h2o.H2OFrame(df)
        train["y"] = train["y"].asfactor()
        for c, t in train.types.items():
            add("frame_types", c, t)
        run("default", train)
        h2o.remove_all()
        train = h2o.H2OFrame(df)
        train["y"] = train["y"].asfactor()
        run("target_encoding", train, preprocessing=["target_encoding"])
    finally:
        h2o.cluster().shutdown()
    out = RESULTS / "tidy" / "exp06_h2o_probe.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out)


if __name__ == "__main__":
    main()
