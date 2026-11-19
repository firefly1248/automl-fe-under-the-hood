"""exp06 orchestrator (core env): `python experiments/exp06_automl_end_to_end/run.py [--datasets K ...]
[--arms A ...] [--retry-failed]`.

As exp02: one unit at a time, the whole machine per unit, memory watchdog at ram_gb - 4. Each arm's
code hash covers its own worker, so a fix to one tool's wrapper reruns only that arm.
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
from afe.run import checks, limits, store
from afe.run.cli import PYTHONHASHSEED, git_sha
from afe.run.unit import code_hash, digest, file_hash

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
SRC = ROOT / "src" / "afe"
WORKER = {"ag_best": ("autogluon", "worker_ag.py"), "lama": ("lama", "worker_lama.py"),
          "h2o": ("h2o", "worker_h2o.py"), "h2o_te": ("h2o", "worker_h2o.py"),
          **{a: ("lama", "worker_lama_diag.py") for a in CFG["diagnostics"]["arms"]}}
SHARED = (SRC / "data" / "loaders.py", SRC / "data" / "splits.py")
CODE = {arm: code_hash(HERE / w, *SHARED) for arm, (_, w) in WORKER.items()}


def plan(noise: bool = False) -> list[dict]:
    """The 230 main units; with `noise`, instead the repeat runs of `CFG["noise"]` (the same units plus a `rep` key,
    so they get their own hash and records and never enter the main tables)."""
    proto = CFG["protocol"]
    protocol_hash = digest({**proto, "pythonhashseed": PYTHONHASHSEED})
    n = CFG["noise"]
    units = []
    for rep in n["reps"] if noise else [None]:
        for key in n["datasets"] if noise else CFG["datasets"]:
            data_hash = json.loads((CACHE / "data" / f"{key}.json").read_text())["frame_hash"]
            splits_hash = S.splits_hash(S.load(key))
            arms = n["arms"] if noise else CFG["arms"] + (CFG["diagnostics"]["arms"] if key in CFG["diagnostics"]["datasets"] else [])
            for arm in arms:
                lock = file_hash(ROOT / "envs" / WORKER[arm][0] / "uv.lock")
                for fold in range(proto["n_folds"]):
                    u = {"exp": CFG["name"], "dataset": key, "repeat": proto["repeat"], "fold": fold,
                         "arm": arm, "protocol_hash": protocol_hash, "data_hash": data_hash,
                         "splits_hash": splits_hash, "env_lock_hash": lock, "code_hash": CODE[arm]}
                    units.append(u | {"rep": rep} if noise else u)
    return units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+")
    ap.add_argument("--arms", nargs="+")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--noise", action="store_true", help="run the repeat runs of the run-to-run noise measurement")
    a = ap.parse_args()
    retry = store.RETRY_FAILED if a.retry_failed else set()
    num_cpus = psutil.cpu_count(logical=False) - 2
    mem_gb = round(psutil.virtual_memory().total / 2**30) - 4
    lim = {**CFG["limits"], "mem_gb": mem_gb, "time_limit": CFG["protocol"]["time_limit"], "num_cpus": num_cpus}
    base = {"git_sha": git_sha(), "machine": machine_info(), "jobs": 1}
    units = [u for u in plan(a.noise) if (not a.datasets or u["dataset"] in a.datasets)
             and (not a.arms or u["arm"] in a.arms)]
    print(f"{CFG['name']}: {len(units)} units, num_cpus={num_cpus}, mem watchdog {mem_gb} GB", flush=True)
    counts: dict[str, int] = {}
    for u in units:
        h = digest(u)
        if store.is_done(store.read(store.record_path(u["exp"], "fit", h)), retry):
            continue
        env_name, worker = WORKER[u["arm"]]
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "out.json"
            task = {"unit": u, "out": str(out_path), "ag_path": str(Path(tmp) / "ag"),
                    "time_limit": CFG["protocol"]["time_limit"], "eval_metric": CFG["protocol"]["eval_metric"],
                    "num_cpus": num_cpus, "mem_gb": mem_gb}
            (Path(tmp) / "task.json").write_text(json.dumps(task))
            cmd = ["uv", "run", "--frozen", "--project", str(ROOT / "envs" / env_name), "python",
                   str(HERE / worker), str(Path(tmp) / "task.json")]
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
            if res.status == "ok" and (err := checks.verdict(result)):
                res.status, res.error = "failed", err
        rec = {"hash": h, "stage": "fit", "unit": u, "status": res.status, "error": res.error,
               "wall_s": res.wall_s, "peak_rss_mb": res.peak_rss_mb, "watchdog": res.kill, "limits": lim,
               "started": started, "finished": datetime.now(timezone.utc).isoformat(), "result": result, **base}
        store.write_json(store.record_path(u["exp"], "fit", h), rec)
        counts[res.status] = counts.get(res.status, 0) + 1
        print(f"fit {res.status:14} {u['dataset']:>24} {u['arm']:>6} {h}", flush=True)
    print("done:", counts or "nothing to run", flush=True)


if __name__ == "__main__":
    main()
