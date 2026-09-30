"""Unit definitions and hashes (plan section 5). stdlib only."""

import hashlib
import json
from pathlib import Path


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def gen_unit(exp, dataset, repeat, fold, arm, arm_params, protocol_hash, data_hash, splits_hash, env_lock_hash) -> dict:
    return {
        "exp": exp, "dataset": dataset, "repeat": repeat, "fold": fold, "arm": arm, "arm_params": arm_params,
        "protocol_hash": protocol_hash, "data_hash": data_hash, "splits_hash": splits_hash,
        "env_lock_hash": env_lock_hash,
    }


def fit_unit(gen_hash, arm, view, model, model_params, seed, core_lock_hash) -> dict:
    return {
        "gen_hash": gen_hash, "arm": arm, "view": view, "model": model, "model_params": model_params,
        "seed": seed, "core_lock_hash": core_lock_hash,
    }
