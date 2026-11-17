"""exp02 hyperparameter diagnostic arms (autogluon env): `python worker_hp.py <task.json>`.

`A_et0` (the `_r13` hyperparameters with `extra_trees=False`, no prep) still trails `A_def` on elevators and
kr-vs-kp. Each arm here returns one group of `_r13` values to AutoGluon's default, keeping the rest of `A_et0`:
- A_et0_nosamp: no `feature_fraction`, `bagging_fraction`, `bagging_freq`;
- A_et0_nocap:  no `learning_rate`, `num_leaves`, `min_data_in_leaf`;
- A_et0_noreg:  no `lambda_l1`, `lambda_l2`;
- A_et0_nocat:  no `cat_l2`, `cat_smooth`, `max_cat_to_onehot`, `min_data_per_group`.
"""

import json
import sys
from pathlib import Path

import worker
import worker_single

GROUPS = {
    "A_et0_nosamp": ("feature_fraction", "bagging_fraction", "bagging_freq"),
    "A_et0_nocap": ("learning_rate", "num_leaves", "min_data_in_leaf"),
    "A_et0_noreg": ("lambda_l1", "lambda_l2"),
    "A_et0_nocat": ("cat_l2", "cat_smooth", "max_cat_to_onehot", "min_data_per_group"),
}
A_ET0 = worker_single.ARMS["A_et0"]["hyperparameters"]["GBM"][0]
ARMS = {arm: {"hyperparameters": {"GBM": [{k: v for k, v in A_ET0.items() if k not in drop}]}}
        for arm, drop in GROUPS.items()}
arm_kwargs = worker.arm_kwargs
worker.arm_kwargs = lambda arm: ARMS[arm] if arm in ARMS else arm_kwargs(arm)

if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(worker.main(task), default=str))
