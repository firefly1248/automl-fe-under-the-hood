"""`base_codes` arm (exp09): the `base` view with every categorical column replaced by its integer level codes (float, NaN for
missing and unseen levels).

Diagnostic for fps_benchmark, where `base_numcast` explains almost none of the CatBoost effect of `ag_default` and `skrub_tv`:
this arm gives both models every categorical column as a plain number, so the difference to `base` is the categorical handling
alone. Codes follow the levels of the fit rows sorted as text. Its output replaces the input.
"""

import pandas as pd

from afe.generators.base import BaseView, FeatureGenerator


class BaseCodes(FeatureGenerator):
    name = "base_codes"
    env = "core"
    replaces_input = True

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        self.view = BaseView().fit(X_fit)
        return self.transform(X_fit)

    def transform(self, X) -> pd.DataFrame:
        out = self.view.transform(X)
        for c in self.view.levels:
            out[c] = out[c].cat.codes.astype(float).where(out[c].notna())
        return out

    def describe(self) -> dict:
        return {"coded": list(self.view.levels)}
