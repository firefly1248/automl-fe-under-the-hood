"""exp03 steps 1-2: LightAutoML default TabularAutoML config and what the Reader does to each column.

Run in the lama env: `make exp03`.
"""

import inspect
from pathlib import Path

import lightautoml.reader.base as rb
import numpy as np
import pandas as pd
import yaml
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBSimpleFeatures
from lightautoml.pipelines.features.linear_pipeline import LinearFeatures
from lightautoml.tasks import Task

from afe.data.loaders import ROOT, load
from afe.data.synthetic import hard_columns

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
TIDY = ROOT / "results" / "tidy"
LAMA_TASK = {"binary": "binary", "multiclass": "multiclass", "regression": "reg"}
# create_automl routes these algo keys to get_gbms / get_linear / get_rfs / get_nn (tabular_presets.py)
ALGO_MODEL = {"lgb": "gbm", "lgb_tuned": "gbm", "cb": "gbm", "cb_tuned": "gbm", "xgb": "gbm", "xgb_tuned": "gbm",
              "linear_l2": "linear_l2", "rf": "rf", "rf_tuned": "rf"}


def automl(task: str) -> TabularAutoML:
    return TabularAutoML(task=Task(LAMA_TASK[task]), timeout=CFG["timeout"], cpu_limit=CFG["cpu_limit"])


def defaults(cls) -> dict:
    return {p.name: repr(p.default) for p in inspect.signature(cls.__init__).parameters.values()
            if p.default is not p.empty}


def step1() -> None:
    out = {"lightautoml_version": __import__("lightautoml").__version__, "tasks": {}}
    rng = np.random.default_rng(CFG["seed"])
    dummy = pd.DataFrame(rng.normal(size=(1000, 5)), columns=list("abcde")).assign(target=rng.integers(0, 2, 1000))
    for task in LAMA_TASK:
        a = automl(task)
        a.infer_auto_params(dummy, target_col="target")
        algos = [algo for level in a.general_params["use_algos"] for algo in level]
        out["tasks"][task] = {
            "use_algos": a.general_params["use_algos"],
            "feature_pipeline_per_algo": {
                algo: type(a.get_feature_pipeline(ALGO_MODEL[algo])).__name__ for algo in algos if algo in ALGO_MODEL
            },
            "selection_params": a.selection_params,
            "selection_mode_meaning": {
                0: "no selection",
                1: "ImportanceCutoffSelector: LGBSimpleFeatures + LightGBM, gain importance, drop importance <= cutoff",
                2: "mode 1 plus NpIterativeFeatureSelector (permutation importance)",
            }[a.selection_params["mode"]],
            "gbm_pipeline_params": a.gbm_pipeline_params,
            "linear_pipeline_params": a.linear_pipeline_params,
            "reader_params": a.reader_params,
        }
    out["pipeline_defaults"] = {c.__name__: defaults(c) for c in (LGBSimpleFeatures, LGBAdvancedPipeline, LinearFeatures)}
    out["reader_class_defaults"] = defaults(rb.PandasToPandasReader)
    (TIDY / "exp03_config.yaml").write_text(yaml.safe_dump(out, sort_keys=False, default_flow_style=None))


captured: dict = {}


def spy(name: str) -> None:
    """Record what the reader's own rule functions return, without changing them."""
    fn = getattr(rb, name)

    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        captured[name] = res.copy()
        return res

    setattr(rb, name, wrapper)


def role_str(role) -> str:
    if role is None:
        return ""
    extra = [f"{k}={getattr(role, k)}" for k in ("encoding_type", "ordinal", "discretization") if getattr(role, k, None)]
    return role.name + (f"({', '.join(extra)})" if extra else "")


def fired(stat: pd.DataFrame | None, col: str) -> str:
    if stat is None or col not in stat.index:
        return ""
    return ",".join(c for c in stat.columns if "rule" in c and bool(stat.at[col, c])) or "none"


def roles_for(name: str, X: pd.DataFrame, y: pd.Series, task: str) -> list[dict]:
    captured.clear()
    params = {**automl(task).reader_params, "n_jobs": CFG["cpu_limit"]}
    reader = rb.PandasToPandasReader(Task(LAMA_TASK[task]), **params)
    reader.fit_read(X.assign(__target__=y.values), roles={"target": "__target__"})
    num, cat = captured.get("calc_encoding_rules"), captured.get("calc_category_rules")
    rows = []
    for col in X.columns:
        ok = reader._is_ok_feature(X[col])
        simple = reader._guess_role(X[col]) if ok else None
        final = reader.roles.get(col)
        num_rules, cat_rules = fired(num, col), fired(cat, col)
        if not ok:
            reason = "simple check: nan rate or constant rate >= reader limit"
        elif final is None:
            reason = "advanced: best encoding score below drop_score_co"
        elif num is not None and col in num.index:
            reason = f"numeric rules fired: {num_rules}" if num_rules != "none" else "no numeric rule fired -> category"
        elif cat is not None and col in cat.index:
            reason = f"category rules fired: {cat_rules}"
        else:
            reason = "simple guess kept"
        simple_name = simple.name if simple is not None else "Drop"
        final_name = final.name if final is not None else "Drop"
        rows.append(
            {
                "dataset": name,
                "column": col,
                "dtype": str(X[col].dtype),
                "nunique": int(X[col].nunique()),
                "simple_role": role_str(simple) or "Drop",
                "lama_role": role_str(final) or "Drop",
                "change": "" if simple_name == final_name else f"{simple_name} -> {final_name}",
                "reason": reason,
            }
        )
    for key, stat in (("numeric", num), ("category", cat)):
        if stat is not None:
            stat.assign(dataset=name, stat=key).to_csv(
                TIDY / "exp03_role_stats.csv", mode="a", header=not (TIDY / "exp03_role_stats.csv").exists()
            )
    return rows


def step2() -> None:
    (TIDY / "exp03_role_stats.csv").unlink(missing_ok=True)
    for name in ("calc_encoding_rules", "calc_category_rules"):
        spy(name)
    X, y, meta = hard_columns(n=CFG["n"], seed=CFG["seed"])
    rows = roles_for("hard_columns", X, y, meta["task"])
    for key in CFG["typed_datasets"]:
        X, y, meta = load(key)
        rows += roles_for(key, X, y, meta["task"])
    pd.DataFrame(rows).to_csv(TIDY / "exp03_roles.csv", index=False)


if __name__ == "__main__":
    TIDY.mkdir(parents=True, exist_ok=True)
    step1()
    step2()
