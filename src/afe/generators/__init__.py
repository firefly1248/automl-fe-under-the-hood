"""Arm registry: arm -> (module, class, env). Modules import lazily so each env loads only its tools."""

import importlib

ARMS = {
    "base": ("afe.generators.identity", "Identity", "core"),
    "manual_baseline": ("afe.generators.manual_baseline", "ManualBaseline", "core"),
    "ag_default": ("afe.generators.ag_default", "AgDefault", "autogluon"),
    "ag_tabprep": ("afe.generators.ag_tabprep", "AgTabPrep", "autogluon"),
    "lama_adv": ("afe.generators.lama_pipeline", "LamaAdv", "lama"),
    "openfe": ("afe.generators.openfe_gen", "OpenFEInductive", "openfe"),
    "openfe_transductive": ("afe.generators.openfe_transductive", "OpenFETransductive", "openfe"),  # exp05 only
    "autofeat": ("afe.generators.autofeat_gen", "AutoFeat", "autofeat"),
    "mljar_golden": ("afe.generators.mljar_golden", "MljarGolden", "mljar"),
    "skrub_tv": ("afe.generators.skrub_tv", "SkrubTV", "skrub"),
    "base_numcast": ("afe.generators.base_numcast", "BaseNumcast", "core"),  # exp09 only
    "base_codes": ("afe.generators.base_codes", "BaseCodes", "core"),  # exp09 only
}
# fit-only arms: arm -> source arm whose cached generator output they reuse with their own view
FIT_ONLY = {"ag_tabprep_keepcat": "ag_tabprep"}


def env_of(arm: str) -> str:
    return ARMS[arm][2]


def make(arm: str, **params):
    module, cls, _ = ARMS[arm]
    return getattr(importlib.import_module(module), cls)(**params)
