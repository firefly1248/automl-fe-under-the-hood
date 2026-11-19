"""`skrub_tv` arm: skrub's `TableVectorizer` with all defaults (protocol change A4, 2026-11-17).

A third implementation of "type guessing + encoding" next to `ag_default` and `lama`, in the scikit-learn ecosystem.
Like `ag_default` its output replaces the input: the downstream model gets only the vectorized, all-numeric table, so no
categorical column reaches CatBoost (part of what the arm measures). It does not use `y`, so there is no out-of-fold
step. The only non-default thing is the global numpy seed set before the fit: the default `StringEncoder` has
`random_state=None` and its SVD would otherwise differ between runs.
"""

import numpy as np
import pandas as pd
from skrub import TableVectorizer

from afe.generators.base import FeatureGenerator


class SkrubTV(FeatureGenerator):
    name = "skrub_tv"
    env = "skrub"
    replaces_input = True
    unseen_default = "categories -> all-zero one-hot block (handle_unknown='ignore'); high-cardinality text -> its own embedding"

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        np.random.seed(seed)
        self.tv = TableVectorizer()
        return self.tv.fit_transform(X_fit)

    def transform(self, X) -> pd.DataFrame:
        return self.tv.transform(X)

    def describe(self) -> dict:
        return {"column_to_kind": self.tv.column_to_kind_, "n_outputs": len(self.tv.all_outputs_)}
