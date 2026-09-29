"""AutoGluon TabPrep feature group, built by AutoGluon's own code path (plan section 1.2).

`ModelAgnosticPrepMixin.get_preprocessor_tabprep` is what `GBM_PREP` calls inside `_preprocess`.
The shim below gives it the three attributes it reads, so the group (BulkFeatureGenerator with
`remove_unused_features="false_recursive"`, `post_drop_duplicates=True`, `passthrough=True`) is
the library's, not a re-implementation.
"""

from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
from autogluon.tabular.models.tabprep.prep_mixin import ModelAgnosticPrepMixin

PORTFOLIO = "zeroshot_2025_12_18_cpu"


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
