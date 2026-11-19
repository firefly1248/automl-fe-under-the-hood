"""One-off (2026-11-19), core env: `python experiments/exp04_open_benchmark/rekey.py [rekey|prune]`.

The LightGBM categorical fix in `afe.models.downstream` changes the fit-code hash of every fit unit, though the CatBoost code
path is untouched. `rekey` copies each CatBoost record whose unit is otherwise unchanged (same generator output) to the unit of
the new hash and marks it `rekeyed_from`; a record of an `ag_default` unit has a new generator hash and is not found, so it is
rerun, as are all LightGBM units. `prune` deletes the records no planned unit refers to (the old hashes); run it after the
reruns, delete the `*.jsonl.gz` archives of the touched stages and `make pack`.
"""

import importlib.util
import sys

import yaml

from afe.paths import RESULTS, ROOT
from afe.run import cli, store
from afe.run.unit import digest

OLD_FIT, OLD_EXP07 = "26d6f4b41f560b89", "df5ec31fee261e7d"  # code hashes before the fix (models dir; exp07 wrapper + models)
EXP = ROOT / "experiments"
CONFIGS = {  # config -> stages
    EXP / "exp04_open_benchmark" / "config.yaml": ["full_r0", "full_r1", "timing"],
    EXP / "exp05_leakage_checks" / "permuted.yaml": ["run"],
    EXP / "exp05_leakage_checks" / "transductive.yaml": ["run"],
    EXP / "exp05_leakage_checks" / "planted.yaml": ["run"],
    EXP / "exp09_numcast" / "config.yaml": ["run"],
}


def units():
    """(exp, stage, unit) of every planned generator and fit unit, exp07 included."""
    for path, stages in CONFIGS.items():
        cfg = yaml.safe_load(path.read_text())
        for stage in stages:
            for g, fits in cli.plan(cfg, stage):
                yield cli.exp_key(cfg, stage), "gen", g
                for f in fits:
                    yield g["exp"], "fit", f
    spec = importlib.util.spec_from_file_location("exp07_run", EXP / "exp07_ctr_ablation" / "run.py")
    exp07 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp07)
    for _, fits in exp07.plan():
        for f in fits:
            yield f["exp"], "fit", f


def rekey() -> None:
    n = 0
    for exp, stage, u in units():
        if stage != "fit" or u["model"] != "catboost":
            continue
        old_unit = {**u, "code_hash": OLD_EXP07 if exp == "exp07_ctr_ablation" else OLD_FIT}
        old, new = store.record_path(exp, "fit", digest(old_unit)), store.record_path(exp, "fit", digest(u))
        if old.exists() and not new.exists():
            store.write_json(new, {**store.read(old), "hash": digest(u), "unit": u, "rekeyed_from": digest(old_unit)})
            n += 1
    print("rekeyed", n)


def prune() -> None:
    planned = {(exp, stage, digest(u)) for exp, stage, u in units()}
    n = 0
    for exp in sorted({e for e, _, _ in planned}):
        for stage in ("gen", "fit"):
            for f in (RESULTS / "raw" / exp / stage).glob("*.json"):
                if (exp, stage, f.stem) not in planned:
                    f.unlink()
                    n += 1
    print("pruned", n)


if __name__ == "__main__":
    {"rekey": rekey, "prune": prune}[sys.argv[1]]()
