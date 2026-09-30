"""Kill-and-resume: kill the orchestrator mid-run, rerun, check no lost and no duplicate units."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

from afe.data import splits as S
from afe.data.loaders import encode_y
from afe.data.synthetic import hard_columns

N_FOLDS, MODELS = 5, 2


def _setup(tmp: Path) -> tuple[Path, dict]:
    cache, results = tmp / "cache", tmp / "results"
    (cache / "data").mkdir(parents=True)
    (results / "splits").mkdir(parents=True)
    X, y, meta = hard_columns(n=600, seed=0)
    X.drop(columns="mixed").assign(target=y).to_parquet(cache / "data" / "toy.parquet")  # mixed: not Arrow-typed
    (cache / "data" / "toy.json").write_text(json.dumps({"key": "toy", "target": "target", "task": "binary",
                                                         "frame_hash": "0"}))
    S.make(encode_y(y, "binary"), "binary").to_parquet(results / "splits" / "toy.parquet", index=False)
    cfg = {"name": "resume_test", "arms": ["base"],
           "protocol": {"n_folds": N_FOLDS, "repeats": 1,
                        "models": {"lgbm": {"n_estimators": 30, "learning_rate": 0.1, "early_stopping": 10},
                                   "catboost": {"iterations": 30, "learning_rate": 0.1, "early_stopping": 10}}},
           "limits": {"gen_timeout_s": 120, "fit_timeout_s": 120, "mem_gb": 4, "min_free_gb": 0.5},
           "stages": {"all": {"datasets": ["toy"]}}}
    (tmp / "config.yaml").write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "AFE_CACHE": str(cache), "AFE_RESULTS": str(results)}
    env.pop("VIRTUAL_ENV", None)
    return results, env


def _cli(tmp, env):
    return [sys.executable, "-m", "afe.run.cli", str(tmp / "config.yaml"), "all", "--jobs", "2"]


def _records(results):
    return {p: json.loads(p.read_text()) for p in (results / "raw" / "resume_test").glob("*/*.json")}


def test_kill_and_resume(tmp_path):
    results, env = _setup(tmp_path)
    p = subprocess.Popen(_cli(tmp_path, env), env=env, start_new_session=True,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while len(_records(results)) < 3 and time.time() - t0 < 300:
        time.sleep(0.2)
    os.killpg(p.pid, signal.SIGKILL)
    p.wait()
    first = _records(results)
    assert 3 <= len(first) < N_FOLDS * (1 + MODELS), "kill must land mid-run"

    subprocess.run(_cli(tmp_path, env), env=env, check=True, capture_output=True)
    final = _records(results)
    assert len(final) == N_FOLDS * (1 + MODELS)
    assert all(r["status"] == "ok" for r in final.values())
    assert len({r["hash"] for r in final.values()}) == len(final)
    for path, rec in first.items():  # finished units were skipped, not rerun
        assert final[path]["started"] == rec["started"]

    out = subprocess.run(_cli(tmp_path, env), env=env, check=True, capture_output=True, text=True).stdout
    assert "nothing to run" in out
