"""exp03 step 3: default `TabularAutoML` fits on adult, diamonds and Amazon_employee_access.

For each dataset: OOF score of the blend, and per model the features it received and the selector's
selected/dropped features. Simple vs advanced pipeline is not reachable via config (step 1: the pipeline
is fixed per algo), so only the default run is recorded. Run in the lama env: `make exp03-step3`.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.tasks import Task
from sklearn.metrics import log_loss, mean_squared_error, r2_score, roc_auc_score

from afe.data.loaders import ROOT, load
from afe.machine import strip_home

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
TIDY = ROOT / "results" / "tidy"
RAW = ROOT / "results" / "raw" / "exp03"
DATASETS = ["adult", "diamonds", "Amazon_employee_access"]
LAMA_TASK = {"binary": "binary", "multiclass": "multiclass", "regression": "reg"}


def oof_metrics(task: str, y: pd.Series, oof: np.ndarray, reader) -> dict:
    ok = np.isfinite(oof).all(axis=1)
    if task == "regression":
        pred = oof[ok, 0]
        return {"oof_rows": int(ok.sum()), "rmse": float(np.sqrt(mean_squared_error(y[ok], pred))),
                "r2": float(r2_score(y[ok], pred))}
    mapping = reader.class_mapping or {c: i for i, c in enumerate(sorted(y.unique()))}
    yi = y.map(mapping).to_numpy()[ok]
    p1 = oof[ok, 0]
    return {"oof_rows": int(ok.sum()), "auc": float(roc_auc_score(yi, p1)),
            "logloss": float(log_loss(yi, np.c_[1 - p1, p1]))}


def selector_info(automl) -> dict:
    """Per level-0 pipeline: the selector's kept features and, when it ran, importances and dropped features."""
    out = {}
    for i, pipe in enumerate(automl.levels[0]):
        sel = getattr(pipe, "pre_selection", None)
        if sel is None or not sel.is_fitted:
            continue
        info = {"selected_features": list(sel.selected_features)}
        imp = getattr(sel, "mapped_importances", None)
        if imp is not None:
            imp = imp.sort_values(ascending=False)
            info["importances"] = {k: float(v) for k, v in imp.items()}
            info["dropped"] = [k for k, v in imp.items() if v <= 0]
        out[f"pipe_{i}"] = info
    return out


def model_rows(automl, dataset: str) -> list[dict]:
    """One row per fitted model: the (post-pipeline) features it received; LAMA keeps them on the algo."""
    rows = []
    for i, level in enumerate(automl.levels):
        for pipe in level:
            for algo in pipe.ml_algos:
                feats = list(algo.features)
                rows.append({"dataset": dataset, "level": i, "model": algo.name,
                             "n_features": len(feats), "features": "; ".join(sorted(feats)[:25])})
    return rows


def main() -> None:
    TIDY.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    cpu = psutil.cpu_count(logical=False) - 2
    oof_rows, m_rows, raw = [], [], {}
    for key in DATASETS:
        X, y, meta = load(key)
        df = X.copy()
        df["__target__"] = y
        automl = TabularAutoML(task=Task(LAMA_TASK[meta["task"]]), timeout=CFG["timeout"], cpu_limit=cpu)
        oof = automl.fit_predict(df, roles={"target": "__target__"}, verbose=1)
        m = oof_metrics(meta["task"], y, oof.data, automl.reader)
        oof_rows.append({"dataset": key, "task": meta["task"], "rows": len(df), "cols": X.shape[1],
                         "used_features": len(automl.collect_used_feats()), **m})
        m_rows += model_rows(automl, key)
        raw[key] = {"metrics": m, "used_features": sorted(automl.collect_used_feats()),
                    "roles": {c: repr(r) for c, r in automl.reader._roles.items()},
                    "selector": selector_info(automl)}
        print(key, m, flush=True)
    pd.DataFrame(oof_rows).to_csv(TIDY / "exp03_fit_oof.csv", index=False)
    pd.DataFrame(m_rows).to_csv(TIDY / "exp03_fit_models.csv", index=False)
    (RAW / "fit_probe.json").write_text(strip_home(json.dumps(raw, indent=1, default=str)))


if __name__ == "__main__":
    main()
