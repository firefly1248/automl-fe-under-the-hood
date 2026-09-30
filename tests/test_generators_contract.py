"""Generator contract (plan section 6), for the arms of the env named by AFE_ENV (default core).

    AFE_ENV=autogluon uv run --project envs/autogluon pytest tests/test_generators_contract.py
"""

import os

import numpy as np
import pandas as pd
import pytest

from afe.data.synthetic import hard_columns
from afe.generators import ARMS, make
from afe.generators.base import finalize

ENV = os.environ.get("AFE_ENV", "core")
ENV_ARMS = [a for a, (_, _, env) in ARMS.items() if env == ENV]  # empty -> tests reported as skipped


@pytest.fixture(scope="module")
def frame():
    X, y, _ = hard_columns(n=800, seed=0)
    return X.iloc[:600], y.iloc[:600], X.iloc[600:], y.iloc[600:]


def _fit(arm, X, y, seed=0):
    g = make(arm)
    return g, g.fit_transform(X, y, "binary", seed)


@pytest.mark.parametrize("arm", ENV_ARMS)
def test_shape_names_dtypes(arm, frame):
    Xf, yf, Xt, _ = frame
    g, out = _fit(arm, Xf, yf)
    for o, X in ((out, Xf), (g.transform(Xt), Xt)):
        assert o.index.equals(X.index) and o.columns.is_unique
        o, _ = finalize(o)
        assert all(pd.api.types.is_numeric_dtype(o[c]) or isinstance(o[c].dtype, pd.CategoricalDtype) for c in o)
        assert not np.isinf(o.select_dtypes("number").to_numpy(dtype=float)).any()


@pytest.mark.parametrize("arm", ENV_ARMS)
def test_deterministic(arm, frame):
    Xf, yf, _, _ = frame
    pd.testing.assert_frame_equal(_fit(arm, Xf, yf)[1], _fit(arm, Xf, yf)[1])


@pytest.mark.parametrize("arm", ENV_ARMS)
def test_test_rows_do_not_change_fit_rows(arm, frame):
    Xf, yf, Xt, _ = frame
    g1, out1 = _fit(arm, Xf, yf)
    g1.transform(Xt)
    noise = Xt.sample(frac=1, random_state=1).set_axis(Xt.index)
    g2, out2 = _fit(arm, Xf, yf)
    g2.transform(noise)
    pd.testing.assert_frame_equal(out1, out2)


@pytest.mark.parametrize("arm", ENV_ARMS)
def test_row_locality(arm, frame):
    Xf, yf, Xt, _ = frame
    g, _ = _fit(arm, Xf, yf)
    shuffled = Xt.sample(frac=1, random_state=2)
    pd.testing.assert_frame_equal(g.transform(Xt), g.transform(shuffled).loc[Xt.index])


@pytest.mark.parametrize("arm", ENV_ARMS)
def test_unseen_categories(arm, frame):
    Xf, yf, Xt, _ = frame
    g, _ = _fit(arm, Xf, yf)
    g.transform(Xt.assign(cat_low="never_seen", cat_rare="never_seen"))


@pytest.mark.parametrize("arm", [a for a in ENV_ARMS if getattr(make(a), "target_encoding", False)])
def test_target_encoding_is_out_of_fold(arm, frame):
    Xf, yf, _, _ = frame
    g, out = _fit(arm, Xf, yf)
    assert not out.equals(g.transform(Xf))
