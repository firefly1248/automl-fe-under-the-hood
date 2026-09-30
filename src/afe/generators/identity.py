"""`base` arm: no new features; downstream sees base_view only."""

import pandas as pd

from afe.generators.base import FeatureGenerator


class Identity(FeatureGenerator):
    name = "base"
    env = "core"

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        return pd.DataFrame(index=X_fit.index)

    def transform(self, X) -> pd.DataFrame:
        return pd.DataFrame(index=X.index)
