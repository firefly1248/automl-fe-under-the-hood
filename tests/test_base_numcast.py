"""`base_numcast` casts exactly the categorical columns whose levels all parse as numbers (exp09)."""

import numpy as np
import pandas as pd

from afe.generators import make


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "num_like": pd.Categorical(["1", "2", "10", "2", "1", "10"]),
        "word": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
        "mixed": pd.Categorical(["1", "x", "1", "x", "1", "x"]),
        "empty": pd.Series([None] * 6, dtype=object),
        "x": np.arange(6.0),
    })


def test_casts_only_number_like_categoricals():
    X, g = frame(), make("base_numcast")
    out = g.fit_transform(X, pd.Series(range(6)), "regression", 0)
    assert g.numeric == ["num_like"] and g.empty == ["empty"]
    assert out.num_like.dtype == float and out.num_like.tolist() == [1, 2, 10, 2, 1, 10]
    assert isinstance(out.word.dtype, pd.CategoricalDtype) and isinstance(out.mixed.dtype, pd.CategoricalDtype)
    assert out["empty"].isna().all() and out["empty"].dtype == float and out.x.tolist() == X.x.tolist()
    seen = g.transform(X.assign(num_like=pd.Categorical(["3"] * 6)))  # a level absent from the fit rows -> NaN, as the base view
    assert seen.num_like.isna().all()


def test_base_codes_gives_every_categorical_as_a_number():
    X, g = frame(), make("base_codes")
    out = g.fit_transform(X, pd.Series(range(6)), "regression", 0)
    assert g.view.levels.keys() == {"num_like", "word", "mixed", "empty"}
    assert out.num_like.tolist() == [0, 2, 1, 2, 0, 1]  # levels sort as text: 1, 10, 2 and out.word.tolist() == [0, 1, 0, 1, 0, 1]
    assert out["empty"].isna().all() and all(out[c].dtype == float for c in out) and out.x.tolist() == X.x.tolist()
    assert g.transform(X.assign(word=pd.Categorical(["c"] * 6))).word.isna().all()  # unseen -> NaN
