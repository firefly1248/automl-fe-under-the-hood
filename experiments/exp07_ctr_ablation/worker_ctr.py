"""One exp07 fit unit (core env): `python worker_ctr.py <task.json>`.

The exp04 fit stage (`afe.run.worker.fit`: same base view, same cached generator output, same early stopping) with
CatBoost's `max_ctr_complexity` set from the unit. CatBoost is built inside the downstream function with a late import,
so wrapping its two estimator classes here changes that one parameter and nothing else.
"""

import functools
import json
import sys
from pathlib import Path

import catboost

from afe.run import worker


def set_ctr_complexity(value: int) -> None:
    for name in ("CatBoostClassifier", "CatBoostRegressor"):
        setattr(catboost, name, functools.partial(getattr(catboost, name), max_ctr_complexity=value))


def main(task: dict) -> dict:
    ctr = task["unit"]["model_params"]["max_ctr_complexity"]
    set_ctr_complexity(ctr)
    return {**worker.fit(task), "max_ctr_complexity": ctr}


if __name__ == "__main__":
    task = json.loads(Path(sys.argv[1]).read_text())
    Path(task["out"]).write_text(json.dumps(main(task), default=str))
