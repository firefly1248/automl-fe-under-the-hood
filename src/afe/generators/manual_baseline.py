"""`manual_baseline` arm: a few hand-written features, every statistic fitted on fit rows (plan section 6).

- frequency encoding (share of fit rows) of every categorical column; NaN is its own level;
- top-5 numeric columns by mutual information with y: a/b and b/a (|b| < 1e-12 -> NaN) and a*b per pair;
- key = categorical column with the highest mutual information with y (none: decile bins of the top-MI
  numeric column); per top-5 numeric column: group mean and x - group mean (unseen key -> global mean).
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from afe.generators.base import FeatureGenerator

TOP = 5
EPS = 1e-12
NAN_KEY = "__nan__"


def _is_num(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _key(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), NAN_KEY).astype(str)


class ManualBaseline(FeatureGenerator):
    name = "manual_baseline"
    env = "core"
    unseen_default = "frequency 0; group mean of an unseen key = global mean"

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        num = [c for c in X_fit.columns if _is_num(X_fit[c])]
        cat = [c for c in X_fit.columns if not _is_num(X_fit[c]) and not pd.api.types.is_bool_dtype(X_fit[c])]
        mi = mutual_info_regression if task == "regression" else mutual_info_classif
        self.medians = X_fit[num].median()
        num_mi = pd.Series(mi(X_fit[num].fillna(self.medians), y_fit, random_state=seed), index=num) if num else pd.Series(dtype=float)
        self.top = num_mi.sort_values(ascending=False, kind="stable").index[:TOP].tolist()
        self.freq = {c: _key(X_fit[c]).value_counts(normalize=True) for c in cat}
        if cat:
            codes = np.column_stack([pd.factorize(_key(X_fit[c]))[0] for c in cat])
            cat_mi = pd.Series(mi(codes, y_fit, discrete_features=True, random_state=seed), index=cat)
            self.key_col, self.bins = cat_mi.sort_values(ascending=False, kind="stable").index[0], None
        elif self.top:
            self.key_col = self.top[0]
            self.bins = np.unique(X_fit[self.key_col].quantile(np.linspace(0, 1, 11)).to_numpy())
        else:
            self.key_col, self.bins = None, None
        if self.key_col:
            key = self._group_key(X_fit)
            self.group_mean = {c: X_fit[c].groupby(key).mean() for c in self.top}
            self.global_mean = {c: X_fit[c].mean() for c in self.top}
        return self.transform(X_fit)

    def _group_key(self, X) -> pd.Series:
        if self.bins is None:
            return _key(X[self.key_col])
        b = np.clip(np.searchsorted(self.bins, X[self.key_col].to_numpy(), side="right") - 1, 0, len(self.bins) - 2)
        return _key(pd.Series(np.where(X[self.key_col].isna(), np.nan, b), index=X.index))

    def transform(self, X) -> pd.DataFrame:
        out = {f"freq__{c}": _key(X[c]).map(f).fillna(0.0).astype(float) for c, f in self.freq.items()}
        for a, b in combinations(self.top, 2):
            xa, xb = X[a].astype(float), X[b].astype(float)
            out[f"{a}__div__{b}"] = xa / xb.where(xb.abs() >= EPS)
            out[f"{b}__div__{a}"] = xb / xa.where(xa.abs() >= EPS)
            out[f"{a}__mul__{b}"] = xa * xb
        if self.key_col:
            key = self._group_key(X)
            for c in self.top:
                gm = key.map(self.group_mean[c]).astype(float).fillna(self.global_mean[c])
                out[f"{c}__gmean__{self.key_col}"] = gm
                out[f"{c}__minus_gmean__{self.key_col}"] = X[c].astype(float) - gm
        return pd.DataFrame(out, index=X.index)

    def describe(self) -> dict:
        return {"top_numeric": self.top, "key": self.key_col, "key_binned": self.bins is not None,
                "frequency_encoded": list(self.freq)}
