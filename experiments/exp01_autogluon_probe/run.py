"""exp01 steps 1, 2 and 4: what AutoGluon's default preprocessing and the TabPrep group do.

Run in the autogluon env: `make exp01` (step 3 is inventory.py).
"""

import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from autogluon.features.generators import AbstractFeatureGenerator, AutoMLPipelineFeatureGenerator

from afe.data.loaders import ROOT, load
from afe.data.synthetic import hard_columns
from afe.generators.ag_tabprep import build_group, build_single
from afe.machine import strip_home

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
TIDY = ROOT / "results" / "tidy"
RAW = ROOT / "results" / "raw" / "exp01"


def capture_logs() -> io.StringIO:
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger = logging.getLogger("autogluon")
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    return buf


def walk(gen: AbstractFeatureGenerator):
    yield gen
    for group in getattr(gen, "generators", []):
        for g in group:
            yield from walk(g)


def own_outputs(col: str, outs: list[str], inputs: list[str]) -> list[str]:
    """AutoGluon links every input of a generator to all its outputs; keep the outputs named after
    this column plus shared outputs not named after any input (e.g. `__nlp__.*` n-grams)."""
    def named(o: str, c: str) -> bool:
        return o == c or o.startswith(c + ".")
    return [o for o in outs if named(o, col) or not any(named(o, c) for c in inputs)]


def note(X: pd.DataFrame, col: str) -> str:
    dup = next((c for c in X.columns if c != col and X[c].equals(X[col])), None)
    if dup:
        return f"exact duplicate of {dup}"
    return "all values unique" if X[col].nunique(dropna=False) == len(X) else ""


def type_rows(gen, X: pd.DataFrame, Xt: pd.DataFrame, variant: str, seed: int) -> list[dict]:
    links = gen.get_feature_links()
    useless = set(gen._useless_features_in or [])
    unused = set(gen._feature_metadata_in_unused.get_features()) if gen._feature_metadata_in_unused else set()
    fm_in, fm_out = gen.feature_metadata_in, gen.feature_metadata
    inputs = list(X.columns)
    rows = []
    for col in inputs:
        outs = own_outputs(col, links.get(col, []), inputs)
        if not outs:
            action = "dropped"
        elif outs == [col]:
            action = "kept" if str(Xt[col].dtype) == str(X[col].dtype) else "retyped"
        else:
            action = "expanded"
        reason = "useless" if col in useless else "unused" if col in unused else ""
        known = col in fm_in.get_features()
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "input_col": col,
                "input_dtype": str(X[col].dtype),
                "ag_raw_in": fm_in.get_feature_type_raw(col) if known else "",
                "ag_special_in": ",".join(fm_in.get_feature_types_special(col)) if known else "",
                "action": action,
                "ag_drop_reason": reason,
                "note": note(X, col) if action == "dropped" else "",
                "n_outputs": len(outs),
                "output_cols": "; ".join(outs[:20]) + (" ..." if len(outs) > 20 else ""),
                "output_dtypes": "; ".join(sorted({str(Xt[o].dtype) for o in outs})),
                "ag_special_out": "; ".join(sorted({",".join(fm_out.get_feature_types_special(o)) for o in outs})),
            }
        )
    return rows


def step1_2() -> None:
    runs = [("base", {"n": CFG["n"]}, s) for s in CFG["seeds"]]
    runs += [(name, kw, CFG["seeds"][0]) for name, kw in CFG["variants"].items()]
    rows, meta = [], {}
    for variant, kw, seed in runs:
        X, y, _ = hard_columns(seed=seed, **kw)
        buf = capture_logs()
        gen = AutoMLPipelineFeatureGenerator(verbosity=3)
        Xt = gen.fit_transform(X, y)
        (RAW / f"log_{variant}_s{seed}.txt").write_text(strip_home(buf.getvalue()))
        rows += type_rows(gen, X, Xt, variant, seed)
        meta[f"{variant}_s{seed}"] = {
            "n_in": X.shape[1],
            "n_out": Xt.shape[1],
            "type_map_raw_out": gen.feature_metadata.type_map_raw,
            "type_group_map_special_out": {k: v for k, v in gen.feature_metadata.type_group_map_special.items() if v},
        }
    pd.DataFrame(rows).to_csv(TIDY / "exp01_types.csv", index=False)
    (TIDY / "exp01_feature_metadata.json").write_text(json.dumps(meta, indent=1))


def encode_y(y: pd.Series, task: str) -> tuple[pd.Series, str]:
    if task == "regression":
        return y.astype(float), "regression"
    codes = pd.Series(pd.factorize(y, sort=True)[0], index=y.index)
    return codes, "binary" if codes.nunique() == 2 else "multiclass"


def step4() -> None:
    seed = CFG["step4_seed"]
    frames = {"hard_columns": hard_columns(n=CFG["n"], seed=seed)}
    frames.update({k: load(k) for k in CFG["step4_datasets"]})
    rows = []
    for name, (X, y, meta) in frames.items():
        y, problem_type = encode_y(y, meta["task"])
        base = AutoMLPipelineFeatureGenerator(verbosity=0)
        X1 = base.fit_transform(X, y)
        for arm in ("arithmetic", "r13_group"):
            gen = (
                build_single("ArithmeticFeatureGenerator", problem_type, seed)
                if arm == "arithmetic"
                else build_group(problem_type, seed)
            )
            t0 = time.perf_counter()
            Xo = gen.fit_transform(X1, y, feature_metadata_in=base.feature_metadata)
            fit_s = time.perf_counter() - t0
            arith = next(g for g in walk(gen) if type(g).__name__ == "ArithmeticFeatureGenerator")
            new = [c for c in Xo.columns if c not in X1.columns]
            row = {
                "dataset": name,
                "arm": arm,
                "rows": len(X),
                "n_in": X1.shape[1],
                "n_out": Xo.shape[1],
                "n_new": len(new),
                "n_originals_kept": Xo.shape[1] - len(new),
                "n_category_out": int((Xo.dtypes == "category").sum()),
                "arith_base_cols": len(arith.used_base_cols),
                "arith_candidates": len(arith.new_feats),
                "arith_selection_method": arith.selection_method,
                "fit_transform_s": round(fit_s, 3),
                "new_examples": "; ".join(new[:10]),
            }
            if name == "hard_columns":
                ratio = (X["a"] / X["b"]).rank()
                rho = Xo[new].select_dtypes("number").rank().corrwith(ratio).abs()
                row["max_abs_rho_with_a_over_b"] = round(float(rho.max()), 5) if len(rho) else np.nan
                row["recovers_a_over_b"] = bool(len(rho) and rho.max() > CFG["recover_threshold"])
                row["best_match"] = rho.idxmax() if len(rho) else ""
            rows.append(row)
    pd.DataFrame(rows).to_csv(TIDY / "exp01_tabprep.csv", index=False)


if __name__ == "__main__":
    TIDY.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    step1_2()
    step4()
