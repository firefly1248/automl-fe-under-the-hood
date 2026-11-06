"""Committed run records against the code in the tree: every record's code_hash is the one this code gives.

Records carry git_sha "...-dirty" (written while the tree had uncommitted edits); this shows that the
hashed code they ran is the committed one. Orchestration, limits and report code are not hashed (plan section 5).
"""

import gzip
import json
from collections import Counter
from functools import cache

from afe.paths import RESULTS
from afe.run import cli
from afe.run.unit import code_hash

arm_code = cache(cli.arm_code)


def mismatches() -> Counter:
    """(exp, stage, arm) -> records whose code_hash differs from the tree's."""
    fit_code, out = code_hash(*cli.FIT_CODE), Counter()
    for gz in sorted(RESULTS.glob("raw/*/*.jsonl.gz")):
        stage = gz.name.split(".")[0]
        for line in gzip.open(gz, "rt"):
            u = json.loads(line)["unit"]
            if u["code_hash"] != (arm_code(u["arm"]) if stage == "gen" else fit_code):
                out[gz.parent.name, stage, u["arm"]] += 1
    return out


def test_records_match_tree_code():
    assert not mismatches()
