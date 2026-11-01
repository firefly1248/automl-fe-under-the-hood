"""`ag_default` arm: AutoGluon's default preprocessing, `AutoMLPipelineFeatureGenerator` (plan section 1.1).

Its output replaces the input, as it does inside AutoGluon before every model.
"""

import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator

from afe.generators.base import FeatureGenerator


class AgDefault(FeatureGenerator):
    name = "ag_default"
    env = "autogluon"
    replaces_input = True
    unseen_default = "categories -> NaN; two-valued columns -> 0 (AutoGluon maps value == fit 'true' value to 1)"

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        self.gen = AutoMLPipelineFeatureGenerator(verbosity=0)
        return self.gen.fit_transform(X_fit, y_fit)

    def transform(self, X) -> pd.DataFrame:
        return self.gen.transform(X)

    def describe(self) -> dict:
        return {"type_map_raw_out": self.gen.feature_metadata.type_map_raw}
