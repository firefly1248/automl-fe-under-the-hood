"""Contract test 7 (plan section 6), openfe env only:

    AFE_ENV=openfe uv run --project envs/openfe pytest tests/test_openfe_equivalence.py

On fit rows, every first-order operator of the inductive evaluator reproduces OpenFE's own `Node.calculate`
on the fit rows alone. On other rows (no OpenFE reference: its transform is transductive), column-wide
operators are checked against a direct pandas computation from fit rows, and fit rows passed through
`transform` get their `fit_transform` values. The kept features and gains on three pilot datasets equal the
pilot records (the first were written before the gain filter was rewritten); needs `make data` and the
committed results/raw/exp04_open_benchmark/gen.jsonl.gz.
"""

import gzip
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

if os.environ.get("AFE_ENV") != "openfe":
    pytest.skip("openfe env only", allow_module_level=True)

from openfe import get_candidate_features  # noqa: E402

from afe.data import splits as S  # noqa: E402
from afe.data.synthetic import hard_columns  # noqa: E402
from afe.generators.openfe_gen import AGG, ROW_WISE, Inductive, OpenFEInductive, prep  # noqa: E402
from afe.paths import RESULTS  # noqa: E402
from afe.run.worker import _data  # noqa: E402

COLS = ["num", "city_code", "a", "b", "cat_low", "cat_rare", "cat_binary"]
PILOT = ["concrete_compressive_strength", "cmc", "credit-g"]  # regression, multiclass, binary; seconds each
PILOT_GEN = RESULTS / "raw" / "exp04_open_benchmark" / "gen.jsonl.gz"


@pytest.fixture(scope="module")
def data():
    X, _, _ = hard_columns(n=600, seed=0)
    X = X[COLS]
    X.loc[X.sample(30, random_state=0).index, ["a", "cat_low"]] = np.nan  # NaN in values and keys
    Xp = prep(X)
    cats = [c for c in Xp.columns if not pd.api.types.is_numeric_dtype(Xp[c])]
    num = [c for c in Xp.columns if c not in cats and Xp[c].nunique() > 100]
    ordinal = [c for c in Xp.columns if c not in cats and c not in num]
    nodes = get_candidate_features(numerical_features=num, categorical_features=cats, ordinal_features=ordinal)
    return Xp.iloc[:400], Xp.iloc[400:], nodes


def _as_float(s: pd.Series) -> np.ndarray:
    return s.cat.codes.to_numpy(float) if isinstance(s.dtype, pd.CategoricalDtype) else s.to_numpy(float)


def test_7_fit_rows_match_openfe(data):
    fit, _, nodes = data
    ours = Inductive(nodes).fit_transform(fit)
    ops = set()
    for i, n in enumerate(nodes):
        # OpenFE's Combine is a category whose values are the factorize codes (-1.0 for NaN): compare values
        ref = deepcopy(n).calculate(fit, is_root=True).astype(float).to_numpy()
        np.testing.assert_allclose(_as_float(ours[i]), ref, rtol=1e-12, equal_nan=True, err_msg=n.name)
        ops.add(n.name)
    assert ops >= {"freq", "Combine", "CombineThenFreq", "GroupByThenMean", "GroupByThenStd", "GroupByThenRank",
                   "GroupByThenNUnique", "/", "log"}


def test_7_new_rows_use_fit_statistics(data):
    fit, new, nodes = data
    ind = Inductive([n for n in nodes if n.name not in ROW_WISE])
    ind.fit_transform(fit)
    out = ind.transform(new)
    for i, n in enumerate(ind.nodes):
        a, b = (new[ch.name] for ch in n.children) if len(n.children) == 2 else (new[n.children[0].name], None)
        fa = fit[n.children[0].name]
        if n.name == "freq":
            exp = a.map(fa.value_counts())
        elif n.name in ("Combine", "CombineThenFreq"):
            fk = fit[n.children[0].name].astype(str) + "_" + fit[n.children[1].name].astype(str)
            fk = fk.where(fit[n.children[0].name].notna() & fit[n.children[1].name].notna())
            k = (a.astype(str) + "_" + b.astype(str)).where(a.notna() & b.notna())
            exp = (pd.Series(pd.Categorical(k, categories=pd.Index(fk.dropna().unique())), index=new.index)
                   if n.name == "Combine" else k.map(fk.value_counts()))
        elif n.name == "GroupByThenRank":
            fb = fit[n.children[1].name]
            exp = pd.Series([np.nan if pd.isna(x) or k not in set(fb.dropna()) or fa[fb == k].isna().all() else
                             ((fa[fb == k] < x).sum() + 0.5 * ((fa[fb == k] == x).sum() + 1)) / fa[fb == k].notna().sum()
                             for x, k in zip(a, b)], index=new.index)
        else:
            exp = b.map(fa.groupby(fit[n.children[1].name]).agg(AGG[n.name]))
        np.testing.assert_allclose(_as_float(out[i]), _as_float(exp), rtol=1e-12, equal_nan=True, err_msg=n.name)


def test_7_fit_rows_through_transform_get_fit_values(data):
    fit, _, nodes = data
    ind = Inductive(nodes)
    ours = ind.fit_transform(fit)
    again = ind.transform(fit)
    for i, n in enumerate(nodes):
        np.testing.assert_allclose(_as_float(again[i]), _as_float(ours[i]), rtol=1e-12, equal_nan=True, err_msg=n.name)


def _pilot_record(key: str) -> dict:
    for line in gzip.open(PILOT_GEN, "rt"):
        r = json.loads(line)
        u = r["unit"]
        if (u["dataset"], u["arm"], u["repeat"], u["fold"]) == (key, "openfe", 0, 0) and r["status"] == "ok":
            return r
    pytest.skip(f"no pilot record for {key}")


@pytest.mark.parametrize("key", PILOT)
def test_7_kept_features_match_pilot(key):
    rec = _pilot_record(key)
    try:
        X, y, meta, idx = _data(rec["unit"])
    except FileNotFoundError:
        pytest.skip("needs make data")
    g = OpenFEInductive()
    g._search(X.iloc[idx["train"]], y.iloc[idx["train"]], meta["task"], S.tool_seed(0, 0))
    assert g.describe() == rec["result"]["describe"]
