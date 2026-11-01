"""`mljar_golden` arm: MLJAR's GoldenFeaturesTransformer as MLJAR applies it (plan sections 1.6, 6).

MLJAR's own caller passes numeric columns only, with NaN filled by medians; the wrapper does the same
with fit-row medians. `features_count` stays at the library default (10 on every dataset here). The
transformer saves golden_features.json to `results_path` and reloads it on the next fit, so each fit
gets a fresh directory, removed afterwards. Its split seed is hard-coded (`random_state=1`).
"""

import tempfile

import pandas as pd
from supervised.algorithms.registry import BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION, REGRESSION
from supervised.preprocessing.goldenfeatures_transformer import GoldenFeaturesTransformer

from afe.generators.base import FeatureGenerator

ML_TASK = {"binary": BINARY_CLASSIFICATION, "multiclass": MULTICLASS_CLASSIFICATION, "regression": REGRESSION}


def _numeric(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c])]


class MljarGolden(FeatureGenerator):
    name = "mljar_golden"
    env = "mljar"

    def applicable(self, X) -> bool:
        return len(_numeric(X)) >= 2

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        self.cols = _numeric(X_fit)
        self.medians = X_fit[self.cols].median()
        with tempfile.TemporaryDirectory(dir=".") as path:
            self.gf = GoldenFeaturesTransformer(results_path=path, ml_task=ML_TASK[task], n_jobs=1)
            self.gf.fit(self._input(X_fit), y_fit)
        return self.transform(X_fit)

    def _input(self, X) -> pd.DataFrame:
        return X[self.cols].fillna(self.medians)  # a new frame: `transform` writes into its input

    def transform(self, X) -> pd.DataFrame:
        return self.gf.transform(self._input(X))[self.gf._new_columns]

    def describe(self) -> dict:
        return {"numeric_inputs": self.cols, "new_features": self.gf._new_features}
