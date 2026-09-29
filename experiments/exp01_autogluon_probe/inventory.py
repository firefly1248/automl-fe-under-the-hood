"""exp01 step 3: generator defaults, preset -> portfolio resolution, every `ag.prep_params` chain.

Run in the autogluon env. If the output differs from plan section 1, the script wins.
"""

import inspect
import json

import autogluon.features.generators as G
import pandas as pd
from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config, get_hyperparameter_config_options
from autogluon.tabular.configs.presets_configs import tabular_presets_alias, tabular_presets_dict

from afe.data.loaders import ROOT

TIDY = ROOT / "results" / "tidy"


def generators() -> pd.DataFrame:
    rows = []
    for name, cls in inspect.getmembers(G, inspect.isclass):
        if not issubclass(cls, G.AbstractFeatureGenerator):
            continue
        for p in inspect.signature(cls.__init__).parameters.values():
            if p.name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            rows.append(
                {
                    "class": name,
                    "module": cls.__module__,
                    "param": p.name,
                    "default": "" if p.default is p.empty else repr(p.default),
                }
            )
    return pd.DataFrame(rows)


def configs(portfolio: dict):
    for model, cfgs in portfolio.items():
        for cfg in cfgs if isinstance(cfgs, list) else [cfgs]:
            yield model, cfg


def presets() -> pd.DataFrame:
    rows = []
    names = {**{p: p for p in tabular_presets_dict}, **tabular_presets_alias}
    for name, target in names.items():
        preset = tabular_presets_dict.get(target)
        row = {"preset": name, "resolves_to": target, "is_alias": name != target}
        if preset is None:
            rows.append({**row, "hyperparameters": "UNRESOLVED"})
            continue
        hp = preset.get("hyperparameters", "default")
        portfolio = get_hyperparameter_config(hp) if isinstance(hp, str) else hp
        models = sorted(portfolio) if isinstance(portfolio, dict) else []
        rows.append(
            {
                **row,
                "hyperparameters": hp if isinstance(hp, str) or hp is None else "custom dict",
                "has_GBM_PREP": "GBM_PREP" in models,
                "n_GBM_PREP_configs": sum(m == "GBM_PREP" for m, _ in configs(portfolio)) if models else 0,
                "model_keys": ",".join(models),
                "auto_stack": preset.get("auto_stack"),
                "num_stack_levels": preset.get("num_stack_levels"),
                "dynamic_stacking": preset.get("dynamic_stacking"),
                "callbacks": json.dumps(preset.get("callbacks")) if "callbacks" in preset else "",
                "time_limit": preset.get("time_limit"),
                "other_keys": ",".join(sorted(set(preset) - {"hyperparameters", "auto_stack", "num_stack_levels",
                                                             "dynamic_stacking", "callbacks", "time_limit"})),
            }
        )
    return pd.DataFrame(rows)


def short(p) -> str:
    if isinstance(p, (list, tuple)) and len(p) == 2 and isinstance(p[0], str):
        name = p[0].removesuffix("FeatureGenerator")
        return name + (f"({', '.join(f'{k}={v}' for k, v in p[1].items())})" if p[1] else "")
    return "[" + ", ".join(short(x) for x in p) + "]"


def prep_configs() -> pd.DataFrame:
    rows = []
    for option in get_hyperparameter_config_options():
        for model, cfg in configs(get_hyperparameter_config(option)):
            if "ag.prep_params" not in cfg:
                continue
            prep = cfg["ag.prep_params"]
            flat = json.dumps(prep)
            rows.append(
                {
                    "portfolio": option,
                    "model": model,
                    "suffix": cfg.get("ag_args", {}).get("name_suffix"),
                    "chain": short(prep),
                    "arithmetic": "Arithmetic" in flat,
                    "cat_interaction": "CategoricalInteraction" in flat,
                    "oof_te": "OOFTargetEncoding" in flat,
                    "groupby": "GroupBy" in flat,
                    "passthrough_types": json.dumps(cfg.get("ag.prep_params.passthrough_types")),
                    "prep_params": flat,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    TIDY.mkdir(parents=True, exist_ok=True)
    generators().to_csv(TIDY / "exp01_generators.csv", index=False)
    presets().to_csv(TIDY / "exp01_presets.csv", index=False)
    prep_configs().to_csv(TIDY / "exp01_prep_configs.csv", index=False)
