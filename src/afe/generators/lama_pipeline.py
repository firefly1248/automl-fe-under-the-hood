"""`lama_adv` arm: LightAutoML 0.4.2 reader, selection mode 1 and `LGBAdvancedPipeline`, wired as
`TabularAutoML` wires them for its GBMs (`create_automl`, `get_selector`, `get_gbms`, `MLPipeline.fit_predict`).

Fit rows get the pipeline's `fit_transform` output: target encodings are out-of-fold over the reader's folds.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # before torch/LightGBM load: their two libomp copies segfault in threaded predict (macOS)

import numpy as np
import pandas as pd
from lightautoml.automl.presets.tabular_presets import TabularAutoML
from lightautoml.ml_algo.boost_lgbm import BoostLGBM
from lightautoml.pipelines.features.lgb_pipeline import LGBAdvancedPipeline, LGBSimpleFeatures
from lightautoml.pipelines.selection.importance_based import ImportanceCutoffSelector, ModelBasedImportanceEstimator
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task
from lightautoml.validation.utils import create_validation_iterator

from afe.generators.base import FeatureGenerator

LAMA_TASK = {"binary": "binary", "multiclass": "multiclass", "regression": "reg"}
TARGET = "__target__"


def role_str(role) -> str:
    extra = [f"{k}={getattr(role, k)}" for k in ("encoding_type", "ordinal") if getattr(role, k, None)]
    return role.name + (f"({', '.join(extra)})" if extra else "")


class LamaAdv(FeatureGenerator):
    name = "lama_adv"
    env = "lama"
    replaces_input = True
    target_encoding = True
    unseen_default = "LAMA encoders: frequency 1, label code 0, target encoding of the rare/unseen bucket; ordinal NaN"

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        lama_task = Task(LAMA_TASK[task])
        preset = TabularAutoML(task=lama_task, cpu_limit=1)  # default params from LAMA's tabular_config.yml
        self.settings = {
            "reader": {**preset.reader_params, "n_jobs": 1, "random_state": seed},
            "selection": preset.selection_params,
            "selector_lgb": {**preset.lgb_params["default_params"], "feature_fraction": 1, "num_threads": 1,
                             "random_state": seed},  # get_selector sets feature_fraction=1
            "gbm_pipeline": preset.gbm_pipeline_params,
        }
        self.reader = PandasToPandasReader(lama_task, **self.settings["reader"])
        data = self.reader.fit_read(X_fit.assign(**{TARGET: y_fit.to_numpy()}), roles={"target": TARGET})
        train_valid = create_validation_iterator(data, None, n_folds=None)  # FoldsIterator over the reader's cv folds
        sel = self.settings["selection"]
        lgb = BoostLGBM(default_params=self.settings["selector_lgb"], freeze_defaults=preset.lgb_params["freeze_defaults"])
        self.selector = ImportanceCutoffSelector(LGBSimpleFeatures(), lgb, ModelBasedImportanceEstimator(),
                                                 cutoff=sel["cutoff"], fit_on_holdout=sel["fit_on_holdout"])
        train_valid = train_valid.apply_selector(self.selector)
        self.pipe = LGBAdvancedPipeline(**self.settings["gbm_pipeline"], feats_imp=self.selector)
        train_valid = train_valid.apply_feature_pipeline(self.pipe)
        out = self._frame(train_valid.train, X_fit.index)
        self.columns = list(out.columns)
        return out

    def transform(self, X) -> pd.DataFrame:
        return self._frame(self.pipe.transform(self.selector.select(self.reader.read(X))), X.index)

    @staticmethod
    def _frame(dataset, index) -> pd.DataFrame:
        dataset = dataset.to_numpy()
        return pd.DataFrame(np.asarray(dataset.data), index=index, columns=dataset.features)

    def describe(self) -> dict:
        return {
            "roles": {c: role_str(r) for c, r in self.reader.roles.items()},
            "reader_dropped": [c for c in self.reader.dropped_features if c != TARGET],
            "selector_importance": {k: float(v) for k, v in self.selector.get_features_score().items()},
            "selected": list(self.selector.selected_features),
            "selector_dropped": list(self.selector.dropped_features),
            "output_features": self.columns,
            "target_encoded": [c for c in self.columns if c.startswith(("oof__", "multioof__"))],
            "n_folds": int(self.reader.cv),
            "settings": self.settings,
        }
