"""exp08 (A4, 2026-11-17): what skrub's `TableVectorizer` with all defaults does to each column.

Run in the skrub env: `make exp08`. One fit on `hard_columns()` (as exp01) and on the typed datasets of the exp03 probe;
writes results/tidy/exp08_skrub_types.csv, one row per input column: dtype, nunique, the kind skrub assigned, the
processing steps, the outputs and the action (kept, retyped, expanded, dropped), and exp08_skrub_defaults.yaml with the
defaults read from the installed version.
"""

import inspect

import pandas as pd
import skrub
import yaml
from skrub import DatetimeEncoder, StringEncoder, TableVectorizer

from afe.data.loaders import ROOT, load
from afe.data.synthetic import hard_columns
from afe.generators.skrub_tv import SkrubTV

TIDY = ROOT / "results" / "tidy"
DATASETS = ["adult", "bank-marketing", "credit-g", "Amazon_employee_access", "Amazon_employee_access_int", "diamonds",
            "house_sales_42092"]  # the typed datasets of the exp03 probe


def defaults(cls) -> dict:
    return {p.name: repr(p.default) for p in inspect.signature(cls.__init__).parameters.values()
            if p.default is not p.empty}


def rows(name: str, X: pd.DataFrame, y: pd.Series, task: str) -> list[dict]:
    g = SkrubTV()
    out = g.fit_transform(X, y, task, 0)
    tv = g.tv
    res = []
    for col in X.columns:
        outs = tv.input_to_outputs_.get(col, [])
        kind = tv.column_to_kind_.get(col, "dropped")
        if not outs:
            action = "dropped"
        elif outs == [col]:
            action = "kept" if str(out[col].dtype) == str(X[col].dtype) else "retyped"
        else:
            action = "expanded"
        res.append({"dataset": name, "input_col": col, "input_dtype": str(X[col].dtype), "nunique": int(X[col].nunique()),
                    "skrub_kind": kind, "steps": " > ".join(map(str, tv.all_processing_steps_.get(col, []))),
                    "action": action, "n_outputs": len(outs), "output_cols": "; ".join(outs[:6]) + (" ..." if len(outs) > 6 else ""),
                    "output_dtypes": "; ".join(sorted({str(out[o].dtype) for o in outs}))})
    return res


def main() -> None:
    X, y, _ = hard_columns(n=5000, seed=0)
    table = rows("hard_columns", X, y, "binary")
    for key in DATASETS:
        X, y, meta = load(key)
        table += rows(key, X, y, meta["task"])
    pd.DataFrame(table).to_csv(TIDY / "exp08_skrub_types.csv", index=False)
    defaults_ = {"skrub_version": skrub.__version__, "TableVectorizer": defaults(TableVectorizer),
                 "StringEncoder": defaults(StringEncoder), "DatetimeEncoder": defaults(DatetimeEncoder)}
    (TIDY / "exp08_skrub_defaults.yaml").write_text(yaml.safe_dump(defaults_, sort_keys=False, width=110))
    print(f"exp08: {len(table)} columns")


if __name__ == "__main__":
    main()
