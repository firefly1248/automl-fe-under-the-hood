"""Contract test 7 (plan section 6), openfe env only:

    AFE_ENV=openfe uv run --project envs/openfe pytest tests/test_openfe_equivalence.py

On fit rows, every first-order operator of the inductive evaluator reproduces OpenFE's own `Node.calculate`
on the fit rows alone. On other rows (no OpenFE reference: its transform is transductive), column-wide
operators are checked against a direct pandas computation from fit rows.
"""

import os
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

if os.environ.get("AFE_ENV") != "openfe":
    pytest.skip("openfe env only", allow_module_level=True)

from openfe import get_candidate_features  # noqa: E402

from afe.data.synthetic import hard_columns  # noqa: E402
from afe.generators.openfe_gen import ROW_WISE, Inductive, prep  # noqa: E402

COLS = ["num", "city_code", "a", "b", "cat_low", "cat_rare", "cat_binary"]


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
            exp = pd.Series([np.nan if pd.isna(x) or k not in set(fb.dropna()) else
                             ((fa[fb == k] < x).sum() + 0.5 * (fa[fb == k] == x).sum()) / fa[fb == k].notna().sum()
                             for x, k in zip(a, b)], index=new.index)
        else:
            stat = {"GroupByThenMin": "min", "GroupByThenMax": "max", "GroupByThenMean": "mean",
                    "GroupByThenMedian": "median", "GroupByThenStd": "std", "GroupByThenNUnique": "nunique"}[n.name]
            exp = b.map(fa.groupby(fit[n.children[1].name]).agg(stat))
        np.testing.assert_allclose(_as_float(out[i]), _as_float(exp), rtol=1e-12, equal_nan=True, err_msg=n.name)
