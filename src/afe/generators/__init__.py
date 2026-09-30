"""Arm registry: arm -> (module, class, env). Modules import lazily so each env loads only its tools."""

import importlib

ARMS = {
    "base": ("afe.generators.identity", "Identity", "core"),
}


def env_of(arm: str) -> str:
    return ARMS[arm][2]


def make(arm: str, **params):
    module, cls, _ = ARMS[arm]
    return getattr(importlib.import_module(module), cls)(**params)
