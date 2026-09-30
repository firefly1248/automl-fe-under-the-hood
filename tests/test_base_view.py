import numpy as np
import pandas as pd

from afe.generators.base import BaseView, combine, finalize


def test_base_view_types_and_unseen():
    fit = pd.DataFrame({"c": ["a", "b", None], "d": pd.to_datetime(["2020-01-01", None, "2021-01-01"]),
                        "b": [True, False, True], "x": [1.0, 2.0, 3.0]})
    v = BaseView().fit(fit)
    out = v.transform(pd.DataFrame({"c": ["a", "zzz", None], "d": fit.d, "b": fit.b, "x": fit.x}))
    assert isinstance(out.c.dtype, pd.CategoricalDtype) and out.c.isna().tolist() == [False, True, True]
    assert out.d.dtype.kind in "if" and out.d.isna().tolist() == [False, True, False]
    assert out.b.dtype == np.int8 and out.x.dtype == float


def test_finalize_and_combine():
    out, n = finalize(pd.DataFrame({"r": [1.0, np.inf, -np.inf], "f": [True, False, True]}))
    assert n == 2 and out.r.isna().sum() == 2 and out.f.dtype == np.int8
    both = combine(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [2], "b": [3]}))
    assert list(both.columns) == ["a", "a__new", "b"]
