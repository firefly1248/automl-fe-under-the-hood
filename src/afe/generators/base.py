"""Generator interface and the shared input view (plan section 6). numpy/pandas only."""

import numpy as np
import pandas as pd


class FeatureGenerator:
    name: str
    env: str
    transductive: bool = False
    replaces_input: bool = False  # True: downstream uses only this output (e.g. ag_default)
    target_encoding: bool = False  # True: contract test 6 checks that fit-row encodings are out-of-fold
    unseen_default: str | None = None  # documented non-NaN output for unseen categories; None: NaN

    def applicable(self, X: pd.DataFrame) -> bool:
        """False if the tool cannot run on this input; the unit is recorded as `not_applicable`."""
        return True

    def fit_transform(self, X_fit: pd.DataFrame, y_fit: pd.Series, task: str, seed: int) -> pd.DataFrame:
        """The ONLY source of features for the rows it was fitted on. Supervised encodings are out-of-fold."""
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """For rows not used in fit (inner validation and test). Same columns as fit_transform."""
        raise NotImplementedError

    def describe(self) -> dict:
        """Created features, ops used, wrapper decisions."""
        return {}


class BaseView:
    """object/string -> category (levels from fit rows, unseen -> NaN); datetime -> int64 ns; bool -> int8."""

    def fit(self, X: pd.DataFrame) -> "BaseView":
        self.levels = {
            c: X[c].astype("category").cat.categories
            for c in X.columns
            if not pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_datetime64_any_dtype(X[c])
            and not pd.api.types.is_bool_dtype(X[c])
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = {}
        for c in X.columns:
            s = X[c]
            if c in self.levels:
                s = pd.Categorical(s.where(s.isin(self.levels[c])), categories=self.levels[c])
            elif pd.api.types.is_datetime64_any_dtype(s):
                s = s.astype("int64").where(s.notna())
            elif pd.api.types.is_bool_dtype(s):
                s = s.astype(np.int8)
            out[c] = s
        return pd.DataFrame(out, index=X.index)


def finalize(out: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Generator output as the downstream sees it: bool -> int8, inf -> NaN. Returns (frame, n_inf)."""
    out = out.astype({c: np.int8 for c in out.columns if pd.api.types.is_bool_dtype(out[c])})
    num = out.select_dtypes("number").columns
    inf = np.isinf(out[num].to_numpy(dtype=float)) if len(num) else np.zeros((len(out), 0), bool)
    if inf.any():
        out[num] = out[num].mask(pd.DataFrame(inf, index=out.index, columns=num))
    return out, int(inf.sum())


def combine(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """base_view + new columns; new names that clash with base get a `__new` suffix."""
    new = new.rename(columns={c: f"{c}__new" for c in new.columns if c in base.columns})
    return pd.concat([base, new], axis=1)
