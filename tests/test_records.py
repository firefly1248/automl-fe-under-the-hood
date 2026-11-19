"""Committed run records against the code in the tree: every record's code_hash is the one this code gives.

Records carry git_sha "...-dirty" (written while the tree had uncommitted edits); this shows that the
hashed code they ran is the committed one. Orchestration, limits and report code are not hashed (plan section 5).
"""

import gzip
import importlib.util
import json
from collections import Counter
from functools import cache

from afe.paths import RESULTS, ROOT
from afe.run import cli
from afe.run.unit import code_hash

arm_code = cache(cli.arm_code)


RUNNERS = {"exp02_autogluon_ablation", "exp06_automl_end_to_end", "exp07_ctr_ablation"}  # standalone runners with their own CODE


@cache
def runner_code(exp: str) -> str | dict:
    """The CODE of a standalone runner (exp02 and exp07: one hash; exp06: one per arm)."""
    spec = importlib.util.spec_from_file_location(exp, ROOT / "experiments" / exp / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CODE


fit_code = cache(lambda: code_hash(*cli.FIT_CODE))


def expected(exp: str, stage: str, arm: str) -> str:
    if exp in RUNNERS:
        code = runner_code(exp)
        return code[arm] if isinstance(code, dict) else code
    return arm_code(arm) if stage == "gen" else fit_code()


def mismatches() -> Counter:
    """(exp, stage, arm) -> records whose code_hash differs from the tree's."""
    out = Counter()
    for gz in sorted(RESULTS.glob("raw/*/*.jsonl.gz")):
        exp, stage = gz.parent.name, gz.name.split(".")[0]
        for line in gzip.open(gz, "rt"):
            u = json.loads(line)["unit"]
            if u["code_hash"] != expected(exp, stage, u["arm"]):
                out[exp, stage, u["arm"]] += 1
    return out


def test_records_match_tree_code():
    assert not mismatches()
