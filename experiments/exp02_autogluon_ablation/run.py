"""exp02 orchestrator (core env): `python experiments/exp02_autogluon_ablation/run.py [--datasets K ...]
[--arms A ...] [--retry-failed]`.

One unit at a time with the whole machine (plan section 8): `num_cpus = physical cores - 2` inside
AutoGluon, no per-unit thread caps, memory watchdog at `ram_gb - 4`. Records and resume as in exp04;
units are stage "fit" so `make pack` tracks them.
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import psutil
import yaml

from afe.data import splits as S
from afe.machine import machine_info
from afe.paths import CACHE, ROOT
from afe.run import limits, store
from afe.run.cli import PYTHONHASHSEED, git_sha
from afe.run.unit import code_hash, digest, file_hash

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
SRC = ROOT / "src" / "afe"
# code that affects results; orchestration and limits are not hashed (plan section 5)
CODE = code_hash(HERE / "worker.py", SRC / "generators" / "ag_tabprep.py", SRC / "data" / "loaders.py",
                 SRC / "data" / "splits.py")


def plan() -> list[dict]:
    proto = CFG["protocol"]
    protocol_hash = digest({**proto, "pythonhashseed": PYTHONHASHSEED})
    lock = file_hash(ROOT / "envs" / "autogluon" / "uv.lock")
    units = []
    for key in CFG["datasets"]:
        data_hash = json.loads((CACHE / "data" / f"{key}.json").read_text())["frame_hash"]
        splits_hash = S.splits_hash(S.load(key))
        for arm in CFG["arms"]:
            for fold in range(proto["n_folds"]):
                units.append({"exp": CFG["name"], "dataset": key, "repeat": proto["repeat"], "fold": fold,
                              "arm": arm, "protocol_hash": protocol_hash, "data_hash": data_hash,
                              "splits_hash": splits_hash, "env_lock_hash": lock, "code_hash": CODE})
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    retry = store.RETRY_FAILED if a.retry_failed else set()
    num_cpus = psutil.cpu_count(logical=False) - 2
    mem_gb = round(psutil.virtual_memory().total / 2**30) - 4
    lim = {**CFG["limits"], "mem_gb": mem_gb, "time_limit": CFG["protocol"]["time_limit"], "num_cpus": num_cpus}
    base = {"git_sha": git_sha(), "machine": machine_info(), "jobs": 1}
    units = [u for u in plan() if (not a.datasets or u["dataset"] in a.datasets)
             and (not a.arms or u["arm"] in a.arms)]
    print(f"{CFG['name']}: {len(units)} units, num_cpus={num_cpus}, mem watchdog {mem_gb} GB", flush=True)
    counts: dict[str, int] = {}
    for u in units:
        h = digest(u)
        if store.is_done(store.read(store.record_path(u["exp"], "fit", h)), retry):
            continue
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.json"
            task = {"unit": u, "out": str(out_path), "ag_path": str(Path(tmp) / "ag"),
                    "time_limit": CFG["protocol"]["time_limit"], "eval_metric": CFG["protocol"]["eval_metric"],
                    "num_cpus": num_cpus}
            (Path(tmp) / "task.json").write_text(json.dumps(task))
            cmd = ["uv", "run", "--frozen", "--project", str(ROOT / "envs" / "autogluon"), "python",
                   str(HERE / "worker.py"), str(Path(tmp) / "task.json")]
            env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"} | {
                "PYTHONHASHSEED": str(PYTHONHASHSEED),
                "RAY_ENABLE_UV_RUN_RUNTIME_ENV": "0",  # Ray's uv hook rejects a cwd outside the project
            }
            limits.wait_for_memory(lim["min_free_gb"])
            started = datetime.now(timezone.utc).isoformat()
            res = limits.run(cmd, lim["unit_timeout_s"], mem_gb, tmp, env, thread_env={})
            result = json.loads(out_path.read_text()) if res.status == "ok" and out_path.exists() else None
            if res.status == "ok" and result is None:
                res.status, res.error = "failed", "child exited 0 without writing out.json"
        rec = {"hash": h, "stage": "fit", "unit": u, "status": res.status, "error": res.error,
               "wall_s": res.wall_s, "peak_rss_mb": res.peak_rss_mb, "watchdog": res.kill, "limits": lim,
               "started": started, "finished": datetime.now(timezone.utc).isoformat(), "result": result, **base}
        store.write_json(store.record_path(u["exp"], "fit", h), rec)
        counts[res.status] = counts.get(res.status, 0) + 1
        print(f"fit {res.status:14} {u['dataset']:>24} {u['arm']:>3} {h}", flush=True)
    print("done:", counts or "nothing to run", flush=True)


if __name__ == "__main__":
    main()
