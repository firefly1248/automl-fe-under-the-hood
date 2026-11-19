"""`base_numcast` arm (exp09): the `base` view with the categorical columns whose levels all parse as numbers cast to float.

Diagnostic for the large CatBoost effects of `ag_default` and `skrub_tv` on fps_benchmark, auction_verification and
Brazilian_houses: AutoGluon keeps number-like categories as numbers and skrub turns them into float32 columns, while the
`base` view hands every non-numeric dtype to CatBoost as a categorical feature. Nothing else changes: other categoricals
stay pandas categories (levels from the fit rows, unseen -> NaN), numeric columns pass through. Its output replaces the input.
A categorical column with no level in the fit rows (all missing: fps_benchmark's GpuNumberOfExecutionUnits) becomes an all-NaN float
column, because an empty pandas category does not survive the parquet cache; it carries no information either way.
"""

import pandas as pd

from afe.generators.base import BaseView, FeatureGenerator


class BaseNumcast(FeatureGenerator):
    name = "base_numcast"
    env = "core"
    replaces_input = True

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        self.view = BaseView().fit(X_fit)
        self.empty = [c for c, lv in self.view.levels.items() if not len(lv)]
        self.numeric = [c for c, lv in self.view.levels.items()
                        if len(lv) and pd.to_numeric(pd.Series(lv.astype(str)), errors="coerce").notna().all()]
        return self.transform(X_fit)

    def transform(self, X) -> pd.DataFrame:
        out = self.view.transform(X)
        for c in self.empty:
            out[c] = float("nan")
        for c in self.numeric:
            out[c] = pd.to_numeric(out[c].astype(object), errors="coerce").astype(float)
        return out

    def describe(self) -> dict:
        return {"cast_to_numeric": self.numeric, "all_missing_to_nan": self.empty}
