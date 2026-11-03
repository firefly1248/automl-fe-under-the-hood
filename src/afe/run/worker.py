"""Child process for one unit: `python -m afe.run.worker <task.json>`.

gen stage (arm env): fit the generator on fit rows, transform inner and test, write the cache.
fit stage (core env): build the downstream input, train, score. The parent writes the record.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from afe.data import splits as S
from afe.data.loaders import encode_y, load
from afe.generators import ARMS, make
from afe.generators.base import BaseView, combine, finalize
from afe.run import store


def _data(unit: dict, permute_y: bool = False):
    X, y, meta = load(unit["dataset"])
    y = encode_y(y, meta["task"])
    fit, inner, test = S.rows(S.load(unit["dataset"]), unit["repeat"], unit["fold"])
    if permute_y:  # exp05 check 1: noise labels on fit and inner rows (early stopping too); test keeps true labels
        rows = np.r_[fit, inner]
        rng = np.random.default_rng(S.tool_seed(unit["repeat"], unit["fold"]))
        y = y.copy()
        y.iloc[rows] = rng.permutation(y.iloc[rows].to_numpy())
    return X, y, meta, {"train": fit, "inner": inner, "test": test}


def gen(task: dict) -> dict:
    from afe.envcheck import ENVS
    from afe.machine import versions

    unit = task["unit"]
    X, y, meta, idx = _data(unit, task["permute_y"])
    g = make(unit["arm"], **unit["arm_params"])
    if not g.applicable(X.iloc[idx["train"]]):
        return {"not_applicable": True}  # describe() needs a fitted generator
    seed = S.tool_seed(unit["repeat"], unit["fold"])
    times, n_inf, outs = {}, 0, {}
    t0 = time.perf_counter()
    if g.transductive:  # one call with inner and test rows together (plan section 6, openfe_transductive)
        n_inner = len(idx["inner"])
        outs["train"], other = g.fit_transform_all(X.iloc[idx["train"]], y.iloc[idx["train"]],
                                                   X.iloc[np.r_[idx["inner"], idx["test"]]], meta["task"], seed)
        outs["inner"], outs["test"] = other.iloc[:n_inner], other.iloc[n_inner:]
        times["fit_s"] = time.perf_counter() - t0
    else:
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
    X, y, meta, idx = _data(task["gen_unit"], task["permute_y"])
    view = BaseView().fit(X.iloc[idx["train"]])
    parts = {}
    for s, rows in idx.items():
        new = pd.read_parquet(Path(task["cache"]) / f"{s}.parquet")
        if unit["view"] == "keepcat":  # base view + only the columns the source arm's generators created
            new = new[g["describe"]["new_columns"]]
        if g["replaces_input"] and unit["view"] != "keepcat":
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
