"""Child process for one unit: `python -m afe.run.worker <task.json>`.

gen stage (arm env): fit the generator on fit rows, transform inner and test, write the cache.
fit stage (core env): build the downstream input, train, score. The parent writes the record.
"""

import json
import sys
import time
from pathlib import Path

import pandas as pd

from afe.data import splits as S
from afe.data.loaders import encode_y, load
from afe.generators import ARMS, make
from afe.generators.base import BaseView, combine, finalize
from afe.run import store


def _data(unit: dict):
    X, y, meta = load(unit["dataset"])
    y = encode_y(y, meta["task"])
    fit, inner, test = S.rows(S.load(unit["dataset"]), unit["repeat"], unit["fold"])
    return X, y, meta, {"train": fit, "inner": inner, "test": test}


def gen(task: dict) -> dict:
    from afe.envcheck import ENVS
    from afe.machine import versions

    unit = task["unit"]
    X, y, meta, idx = _data(unit)
    g = make(unit["arm"], **unit["arm_params"])
    seed = S.tool_seed(unit["repeat"], unit["fold"])
    times, n_inf, outs = {}, 0, {}
    t0 = time.perf_counter()
    outs["train"] = g.fit_transform(X.iloc[idx["train"]], y.iloc[idx["train"]], meta["task"], seed)
    times["fit_s"] = time.perf_counter() - t0
    for s in ("inner", "test"):
        t0 = time.perf_counter()
        outs[s] = g.transform(X.iloc[idx[s]])
        times[f"transform_{s}_s"] = time.perf_counter() - t0
    for s, out in outs.items():
        out, n = finalize(out)
        n_inf += n
        store.write_parquet(Path(task["cache"]) / f"{s}.parquet", out)
    return {
        "n_features_in": X.shape[1],
        "n_features_out": outs["train"].shape[1],
        "replaces_input": g.replaces_input,
        "n_inf_replaced": n_inf,
        **{k: round(v, 3) for k, v in times.items()},
        "describe": g.describe(),
        "versions": versions(ENVS[ARMS[unit["arm"]][2]][1] + ["numpy", "pandas", "scikit-learn"]),
    }


def fit(task: dict) -> dict:
    from afe.models.downstream import fit_score

    unit, g = task["unit"], task["gen"]
    X, y, meta, idx = _data(task["gen_unit"])
    view = BaseView().fit(X.iloc[idx["train"]])
    parts = {}
    for s, rows in idx.items():
        new = pd.read_parquet(Path(task["cache"]) / f"{s}.parquet")
        if g["replaces_input"]:
            Xs = new
        else:
            new.index = X.index[rows]
            Xs = combine(view.transform(X.iloc[rows]), new)
        parts[s] = (Xs, y.iloc[rows])
    k = 0 if meta["task"] == "regression" else int(y.nunique())
    res = fit_score(unit["model"], unit["model_params"], unit["seed"], meta["task"], k,
                    parts["train"], parts["inner"], parts["test"])
    return {**res, "n_features": parts["train"][0].shape[1], "n_rows": {s: len(r) for s, r in idx.items()}}


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    result = {"gen": gen, "fit": fit}[task["stage"]](task)
    Path(task["out"]).write_text(json.dumps(result, default=str))
