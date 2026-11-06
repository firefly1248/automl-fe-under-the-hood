"""Unit definitions and hashes (plan section 5). stdlib only."""

import hashlib
import json
from pathlib import Path


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:16]


def code_hash(*paths: Path) -> str:
    """Hash of every .py file under the given files or directories, by relative name and content."""
    files = sorted(f for p in map(Path, paths) for f in ([p] if p.is_file() else p.rglob("*.py")))
    # only the file name is hashed, so two files of one name could swap code without changing the hash
    assert len({f.name for f in files}) == len(files), f"hashed files share a name: {[f.name for f in files]}"
    h = hashlib.sha1()
    for f in files:
        h.update(f.name.encode() + f.read_bytes())
    return h.hexdigest()[:16]


def gen_unit(exp, dataset, repeat, fold, arm, arm_params, protocol_hash, data_hash, splits_hash, env_lock_hash,
             code_hash) -> dict:
    return {
        "exp": exp, "dataset": dataset, "repeat": repeat, "fold": fold, "arm": arm, "arm_params": arm_params,
        "protocol_hash": protocol_hash, "data_hash": data_hash, "splits_hash": splits_hash,
        "env_lock_hash": env_lock_hash, "code_hash": code_hash,
    }


def fit_unit(gen_hash, arm, view, model, model_params, seed, core_lock_hash, code_hash) -> dict:
    return {
        "gen_hash": gen_hash, "arm": arm, "view": view, "model": model, "model_params": model_params,
        "seed": seed, "core_lock_hash": core_lock_hash, "code_hash": code_hash,
    }
