import numpy as np
import pandas as pd

from afe.data import splits as S


def _y(n=1000, seed=0):
    return pd.Series(np.random.default_rng(seed).choice([0, 1, 2], n, p=[0.7, 0.25, 0.05]))


def test_folds_partition_and_inner():
    y = _y()
    sp = S.make(y, "multiclass")
    seen_test = np.zeros(len(y), int)
    for k in range(S.N_FOLDS):
        fit, inner, test = S.rows(sp, 0, k)
        assert len(set(fit) | set(inner) | set(test)) == len(y)
        assert not (set(fit) & set(inner)) and not (set(fit) & set(test)) and not (set(inner) & set(test))
        seen_test[test] += 1
        assert abs(len(inner) / (len(fit) + len(inner)) - S.INNER_SHARE) < 0.01
        for c in range(3):  # stratified: class shares match within one row per fold
            assert abs((y.iloc[test] == c).sum() - (y == c).sum() / S.N_FOLDS) <= 1
    assert (seen_test == 1).all()


def test_deterministic_and_regression():
    y = _y()
    assert S.make(y, "binary").equals(S.make(y, "binary"))
    yr = pd.Series(np.random.default_rng(1).normal(size=503))
    sp = S.make(yr, "regression")
    assert sorted(sp.outer_fold.value_counts()) == [100, 100, 101, 101, 101]
