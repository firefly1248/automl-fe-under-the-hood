"""Generator contract (plan section 6), for the arms of the env named by AFE_ENV (default core).

    AFE_ENV=autogluon uv run --project envs/autogluon pytest tests/test_generators_contract.py

Runs on the synthetic frame and on one small real dataset (credit-g, if `make data` has cached it).
Test 7 (OpenFE operator equivalence) is in test_openfe_equivalence.py.
"""

import os

import numpy as np
import pandas as pd
import pytest

from afe.data.loaders import encode_y, load
from afe.data.synthetic import hard_columns
from afe.generators import ARMS, make
from afe.generators.base import finalize

ENV = os.environ.get("AFE_ENV", "core")
ENV_ARMS = [a for a, (_, _, env) in ARMS.items() if env == ENV]  # empty -> tests reported as skipped
REAL = "credit-g"


def _frames() -> dict:
    """name -> (X, y, task, categorical columns used for the unseen-category test)."""
    X, y, _ = hard_columns(n=800, seed=0)
    out = {"synthetic": (X, y, "binary", ["cat_low", "cat_rare", "cat_binary"])}
    try:
        Xr, yr, meta = load(REAL)
    except FileNotFoundError:
        return out
    cats = [c for c in Xr.columns if not pd.api.types.is_numeric_dtype(Xr[c])]
    out["real"] = (Xr, encode_y(yr, meta["task"]), meta["task"], cats)
    return out


FRAMES = _frames()
CASES = [pytest.param(a, f, id=f"{a}-{f}") for a in ENV_ARMS for f in FRAMES]
_FITTED: dict = {}


def _split(frame: str):
    X, y, task, cats = FRAMES[frame]
    n = int(len(X) * 0.75)
    return X.iloc[:n], y.iloc[:n], X.iloc[n:], task, cats


def _fit(arm: str, X, y, task, seed=0):
    g = make(arm)
    return g, g.fit_transform(X, y, task, seed)


def fitted(arm: str, frame: str):
    """One fit per (arm, frame), shared by the tests that do not need a fresh one."""
    if (arm, frame) not in _FITTED:
        Xf, yf, Xt, task, _ = _split(frame)
        _FITTED[arm, frame] = _fit(arm, Xf, yf, task)
    return _FITTED[arm, frame]


def assert_same(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Same columns and values; categoricals must have identical categories (LightGBM uses the codes)."""
    assert list(a.columns) == list(b.columns)
    for c in a.columns:
        if isinstance(a[c].dtype, pd.CategoricalDtype) or isinstance(b[c].dtype, pd.CategoricalDtype):
            assert a[c].dtype == b[c].dtype, f"{c}: categories differ"
            assert (a[c].cat.codes.to_numpy() == b[c].cat.codes.to_numpy()).all(), c
        else:
            np.testing.assert_allclose(a[c].to_numpy(float), b[c].to_numpy(float), rtol=1e-7, equal_nan=True,
                                       err_msg=c)


@pytest.mark.parametrize("arm,frame", CASES)
def test_1_shape_names_dtypes(arm, frame):
    Xf, _, Xt, _, _ = _split(frame)
    g, out = fitted(arm, frame)
    out_t = g.transform(Xt)
    assert list(out_t.columns) == list(out.columns)
    for o, X in ((out, Xf), (out_t, Xt)):
        assert o.index.equals(X.index) and o.columns.is_unique
        o, _ = finalize(o)
        assert all(pd.api.types.is_numeric_dtype(o[c]) or isinstance(o[c].dtype, pd.CategoricalDtype) for c in o)
        assert not np.isinf(o.select_dtypes("number").to_numpy(dtype=float)).any()


@pytest.mark.parametrize("arm,frame", CASES)
def test_2_deterministic(arm, frame):
    Xf, yf, _, task, _ = _split(frame)
    assert_same(_fit(arm, Xf, yf, task)[1], _fit(arm, Xf, yf, task)[1])


@pytest.mark.parametrize("arm,frame", CASES)
def test_3_isolation(arm, frame, tmp_path, monkeypatch):
    """Fold 1 alone vs fold 1 after fold 0 in the same process and temp root: no state carried over."""
    monkeypatch.chdir(tmp_path)
    Xf, yf, Xt, task, _ = _split(frame)
    n = len(Xf)
    f0, f1 = slice(0, n * 2 // 3), slice(n // 3, n)
    g1, alone = _fit(arm, Xf.iloc[f1], yf.iloc[f1], task, seed=1)
    alone_t = g1.transform(Xt)
    g0, _ = _fit(arm, Xf.iloc[f0], yf.iloc[f0], task, seed=0)
    g0.transform(Xt)
    g1b, after = _fit(arm, Xf.iloc[f1], yf.iloc[f1], task, seed=1)
    assert_same(alone, after)
    assert_same(alone_t, g1b.transform(Xt))


@pytest.mark.parametrize("arm,frame", CASES)
def test_4_batch_independence(arm, frame):
    _, _, Xt, _, _ = _split(frame)
    g, _ = fitted(arm, frame)
    full = g.transform(Xt)
    half = len(Xt) // 2
    assert_same(full, pd.concat([g.transform(Xt.iloc[:half]), g.transform(Xt.iloc[half:])]))
    rows = Xt.sample(20, random_state=0).index
    assert_same(full.loc[rows], pd.concat([g.transform(Xt.loc[[r]]) for r in rows]))


@pytest.mark.parametrize("arm,frame", CASES)
def test_5_unseen_categories(arm, frame):
    _, _, Xt, _, cats = _split(frame)
    g, _ = fitted(arm, frame)
    seen = g.transform(Xt)
    unseen = g.transform(Xt.assign(**{c: "never_seen" for c in cats}))
    changed = [c for c in seen.columns if not seen[c].equals(unseen[c])]
    if g.unseen_default is None:
        assert all(unseen[c].isna().all() for c in changed), "unseen categories must map to NaN"
    else:  # documented default: only check it is a usable value
        num = finalize(unseen[changed])[0].select_dtypes("number")
        assert not np.isinf(num.to_numpy(dtype=float)).any()


@pytest.mark.parametrize("arm,frame", [p for p in CASES if make(p.values[0]).target_encoding])
def test_6_target_encoding_is_out_of_fold(arm, frame):
    Xf, _, _, _, _ = _split(frame)
    g, out = fitted(arm, frame)
    assert not out.equals(g.transform(Xf))
