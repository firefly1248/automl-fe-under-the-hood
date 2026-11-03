"""exp05 check 4 oracle: the downstream models on base view + the true planted feature, on the folds and
seeds of `planted.yaml`. Writes results/tidy/exp05_oracle.csv. Core env:

    uv run --project envs/core python experiments/exp05_leakage_checks/planted_oracle.py
"""

from pathlib import Path

import pandas as pd
import yaml

from afe.data import splits as S
from afe.data.loaders import load
from afe.generators.base import BaseView
from afe.models.downstream import fit_score
from afe.paths import CACHE, RESULTS

CFG = yaml.safe_load((Path(__file__).parent / "planted.yaml").read_text())

rows = []
for key in CFG["stages"]["run"]["datasets"]:
    X, y, meta = load(key)
    X = X.assign(oracle=pd.read_parquet(CACHE / "data" / f"{key}.oracle.parquet").oracle.to_numpy())
    for r in range(CFG["protocol"]["repeats"]):
        for k in range(CFG["protocol"]["n_folds"]):
            idx = dict(zip(("fit", "inner", "test"), S.rows(S.load(key), r, k)))
            view = BaseView().fit(X.iloc[idx["fit"]])
            parts = {s: (view.transform(X.iloc[i]), y.iloc[i].astype(float)) for s, i in idx.items()}
            for model, params in CFG["protocol"]["models"].items():
                res = fit_score(model, params, S.tool_seed(r, k), "regression", 0, parts["fit"], parts["inner"],
                                parts["test"])
                rows.append({"dataset": key, "repeat": r, "fold": k, "model": model, **res["test"]})
                print(key, k, model, res["test"], flush=True)
pd.DataFrame(rows).to_csv(RESULTS / "tidy" / "exp05_oracle.csv", index=False)
