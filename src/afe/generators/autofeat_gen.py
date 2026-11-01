"""`autofeat` arm: AutoFeat feature engineering and L1 selection on numeric columns (plan sections 1.6, 6).

Wrapper decisions (all recorded in describe()):
- features are built from numeric columns only (`feateng_cols`); NaN filled with fit-row medians;
- categoricals with <= 50 levels are one-hot encoded here on fit-row levels (unseen or NaN -> all zeros)
  and passed as numeric inputs, because autofeat's own encoder refits on every `transform` batch;
  wider categoricals are left out; dummies are inputs to the selection, not new features;
- constant columns are dropped (autofeat's standardization turns them into NaN);
- `max_gb=4`; above it autofeat subsamples rows with the global numpy generator, so it is seeded;
- columns are renamed x0000, x0001, ... for sympy and the new feature names are mapped back.
"""

import re

import numpy as np
import pandas as pd
from autofeat import AutoFeatModel
from autofeat.feateng import n_cols_generated

from afe.generators.base import FeatureGenerator

MAX_LEVELS = 50
MAX_GB = 4
FEATENG_STEPS = 2
TOKEN = re.compile(r"x\d{4}")


def _numeric(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]


class AutoFeat(FeatureGenerator):
    name = "autofeat"
    env = "autofeat"

    def applicable(self, X) -> bool:
        return len(_numeric(X)) >= 1

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        nunique = X_fit.nunique()
        # constant inputs are dropped: autofeat standardizes every column and a zero std gives NaN
        self.constant = nunique.index[nunique <= 1].tolist()
        self.num = [c for c in _numeric(X_fit) if c not in self.constant]
        self.medians = X_fit[self.num].median()
        others = [c for c in X_fit.columns if c not in self.num and c not in self.constant and c not in _numeric(X_fit)]
        self.levels = {c: X_fit[c].dropna().unique().tolist() for c in others if nunique[c] <= MAX_LEVELS}
        self.left_out = [c for c in others if nunique[c] > MAX_LEVELS]
        inputs = self._inputs(X_fit)
        self.names = dict(zip([f"x{i:04}" for i in range(inputs.shape[1])], inputs.columns))
        feateng = [f"x{i:04}" for i in range(len(self.num))]
        n_cols = n_cols_generated(len(feateng), FEATENG_STEPS)
        self.rows_used = min(len(X_fit), int(MAX_GB * 250_000_000 / n_cols))  # autofeat's own subsample rule
        self.model = AutoFeatModel(problem_type="regression" if task == "regression" else "classification",
                                   feateng_cols=feateng, feateng_steps=FEATENG_STEPS, max_gb=MAX_GB, n_jobs=1)
        np.random.seed(seed)
        self.model.fit(inputs.set_axis(list(self.names), axis=1), y_fit.to_numpy())
        return self.transform(X_fit)

    def _inputs(self, X) -> pd.DataFrame:
        """Numeric columns (NaN -> fit median) first, then 0/1 dummies on fit-row levels."""
        parts = [X[self.num].astype(float).fillna(self.medians)]
        for c, levels in self.levels.items():
            parts.append(pd.DataFrame({f"{c}=={v}": (X[c] == v).astype(float) for v in levels}, index=X.index))
        return pd.concat(parts, axis=1)

    def _name(self, formula: str) -> str:
        return TOKEN.sub(lambda m: self.names[m.group()], formula)

    def transform(self, X) -> pd.DataFrame:
        out = self.model.transform(self._inputs(X).set_axis(list(self.names), axis=1))[self.model.new_feat_cols_]
        return out.set_axis([self._name(c) for c in out.columns], axis=1).set_axis(X.index).astype(float)

    def describe(self) -> dict:
        return {"numeric_inputs": self.num, "constant_dropped": self.constant,
                "one_hot": {c: len(v) for c, v in self.levels.items()},
                "left_out_categoricals": self.left_out, "max_gb": MAX_GB, "rows_used": self.rows_used,
                "new_features": [self._name(c) for c in self.model.new_feat_cols_]}
