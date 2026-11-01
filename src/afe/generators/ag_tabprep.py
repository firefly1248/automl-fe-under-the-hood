"""AutoGluon TabPrep feature group, built by AutoGluon's own code path (plan section 1.2).

`ModelAgnosticPrepMixin.get_preprocessor_tabprep` is what `GBM_PREP` calls inside `_preprocess`.
The shim below gives it the three attributes it reads, so the group (BulkFeatureGenerator with
`remove_unused_features="false_recursive"`, `post_drop_duplicates=True`, `passthrough=True`) is
the library's, not a re-implementation.
"""

import pandas as pd
from autogluon.features.generators import AutoMLPipelineFeatureGenerator
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular.models.tabprep.prep_mixin import ModelAgnosticPrepMixin

from afe.generators.base import FeatureGenerator

PORTFOLIO = "zeroshot_2025_12_18_cpu"
PROBLEM_TYPE = {"binary": "binary", "multiclass": "multiclass", "regression": "regression"}
NEW_FROM = ("ArithmeticFeatureGenerator", "OOFTargetEncodingFeatureGenerator")  # generators whose outputs are new


class _Prep(ModelAgnosticPrepMixin):
    def __init__(self, ag_params: dict, problem_type: str, seed: int):
        self._ag_params, self.problem_type, self.random_seed = ag_params, problem_type, seed

    def _get_ag_params(self) -> dict:
        return self._ag_params


def prep_config(suffix: str = "_r13", portfolio: str = PORTFOLIO) -> dict:
    """The `GBM_PREP` hyperparameter dict with the given name suffix."""
    return next(c for c in get_hyperparameter_config(portfolio)["GBM_PREP"] if c["ag_args"]["name_suffix"] == suffix)


def build_group(problem_type: str, seed: int, suffix: str = "_r13", portfolio: str = PORTFOLIO):
    cfg = prep_config(suffix, portfolio)
    ag_params = {k.removeprefix("ag."): v for k, v in cfg.items() if k.startswith("ag.prep_params")}
    return _Prep(ag_params, problem_type, seed).get_preprocessor_tabprep()


def build_single(cls_name: str, problem_type: str, seed: int, **params):
    """One generator with the init params GBM_PREP would give it."""
    return _Prep({}, problem_type, seed)._init_preprocessor(cls_name, params)


def walk(gen):
    yield gen
    for group in getattr(gen, "generators", []):
        for g in group:
            yield from walk(g)


class AgTabPrep(FeatureGenerator):
    """`ag_tabprep` arm: the `_r13` group after `AutoMLPipelineFeatureGenerator`, output as AutoGluon uses it
    (categoricals replaced by OOF target encodings). `fit_transform` gives the out-of-fold encodings that
    `GBM_PREP` trains on; `transform` gives the full-fit encodings.

    New columns (for `ag_tabprep_keepcat`) are identified by provenance: outputs of Arithmetic and of
    OOF-TE (which encodes the originals and the CatInteraction outputs); pass-through columns are not new.
    """

    name = "ag_tabprep"
    env = "autogluon"
    replaces_input = True
    target_encoding = True
    unseen_default = "AutoGluon maps unseen categories to NaN, then OOF-TE gives them the prior encoding"

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        self.base = AutoMLPipelineFeatureGenerator(verbosity=0)
        X1 = self.base.fit_transform(X_fit, y_fit)
        self.group = build_group(PROBLEM_TYPE[task], seed)
        out = self.group.fit_transform(X1, y_fit, feature_metadata_in=self.base.feature_metadata)
        made = {c for g in walk(self.group) if type(g).__name__ in NEW_FROM for c in g.features_out}
        self.new_columns = [c for c in out.columns if c in made]
        return out

    def transform(self, X) -> pd.DataFrame:
        return self.group.transform(self.base.transform(X))

    def describe(self) -> dict:
        return {"new_columns": self.new_columns, "n_out": len(self.group.features_out),
                "by_generator": {type(g).__name__: len(g.features_out) for g in walk(self.group)
                                 if type(g).__name__ in NEW_FROM}}
