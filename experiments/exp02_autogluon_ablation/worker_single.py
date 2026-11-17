"""exp02 single-model diagnostic arms (autogluon env): `python worker_single.py <task.json>`.

Same unit as `worker.py` (it supplies the data, scoring and record), with the arms below. They separate the
`_r13` feature group from the `_r13` model hyperparameters, which `A` and `B` carry together (`_r13` is a
portfolio member tuned for bagged ensembles):
- A_def:  `GBM` with AutoGluon's default hyperparameters, no bagging, no prep;
- B_def:  `GBM_PREP` with only the `_r13` `ag.prep_params`, default model hyperparameters (B_def - A_def = features);
- A_et0:  `A` with `extra_trees=False`;
- B_et0:  `B` with `extra_trees=False`.
"""

import json
import sys
from pathlib import Path

import worker
from afe.generators.ag_tabprep import prep_config

R13 = prep_config("_r13")
PREP = {k: v for k, v in R13.items() if k.startswith("ag.prep_params")}
HP = {k: v for k, v in R13.items() if k not in PREP}
ARMS = {
    "A_def": {"hyperparameters": {"GBM": [{}]}},
    "B_def": {"hyperparameters": {"GBM_PREP": [PREP]}},
    "A_et0": {"hyperparameters": {"GBM": [HP | {"extra_trees": False}]}},
    "B_et0": {"hyperparameters": {"GBM_PREP": [R13 | {"extra_trees": False}]}},
}
arm_kwargs = worker.arm_kwargs
worker.arm_kwargs = lambda arm: ARMS[arm] if arm in ARMS else arm_kwargs(arm)

if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(worker.main(task), default=str))
