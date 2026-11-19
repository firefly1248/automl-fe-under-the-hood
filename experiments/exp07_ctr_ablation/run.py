"""exp07 orchestrator (core env): `python experiments/exp07_ctr_ablation/run.py [--jobs N] [--retry-failed]`.

Plans the CatBoost fits with `max_ctr_complexity=1` for the arms and datasets of `config.yaml` from the exp04 plan: the
generator units, their records and caches are exp04's (nothing is generated here); a chain whose generator record is not ok
(not applicable, failed) is skipped. Each fit is a child process under the exp04 limits; records go to exp07 as stage "fit".
"""

import argparse
import copy
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from afe.machine import machine_info
from afe.paths import RESULTS, ROOT
from afe.run import cli, limits, store
from afe.run.unit import code_hash, digest

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text())
CFG4 = yaml.safe_load((ROOT / CFG["exp04_config"]).read_text())
# exp04's fit code plus the wrapper: a change to either marks the records stale (tests/test_records.py)
CODE = code_hash(HERE / "worker_ctr.py", *cli.FIT_CODE)


def datasets() -> list[str]:
    d = pd.read_csv(RESULTS / "datasets.csv")
    return d[d.primary & d.hicard].key.tolist()  # primary datasets with a categorical column of 10 or more levels


def plan() -> list[tuple[dict, list[dict]]]:
    """[(exp04 generator unit, [exp07 fit units])]; a fit unit is the exp04 CatBoost fit with the parameter added."""
    cfg = copy.deepcopy(CFG4)
    arms = CFG["arms"] + ["ag_tabprep"]  # the keepcat view reads the ag_tabprep generator output
    cfg["stages"]["ablation"] = {"datasets": datasets(), "repeats": CFG["repeats"], "arms": arms}
    out = []
    for g, fits in cli.plan(cfg, "ablation"):
        units = [{"exp": CFG["name"], "dataset": g["dataset"], "repeat": g["repeat"], "fold": g["fold"],
                  "arm": f["arm"], "view": f["view"], "model": f["model"], "gen_hash": f["gen_hash"],
                  "model_params": {**f["model_params"], **CFG["catboost"]}, "seed": f["seed"],
                  "core_lock_hash": f["core_lock_hash"], "code_hash": CODE}
                 for f in fits if f["model"] == "catboost" and f["arm"] in CFG["arms"]]
        if units:
            out.append((g, units))
    return out


class Runner:
    def __init__(self, retry: set[str], jobs: int):
        self.retry, self.lim = retry, CFG["limits"]
        self.base = {"git_sha": cli.git_sha(), "machine": machine_info(), "jobs": jobs}
        self.counts: dict[str, int] = {}

    def chain(self, g: dict, units: list[dict]) -> None:
        gh = digest(g)
        gen = store.read(store.record_path(CFG4["name"], "gen", gh))
        if gen is None or gen["status"] != "ok" or not store.cache_complete(CFG4["name"], gh):
            self.counts["no_generator"] = self.counts.get("no_generator", 0) + len(units)
            return
        for u in units:
            h = digest(u)
            if store.is_done(store.read(store.record_path(CFG["name"], "fit", h)), self.retry):
                continue
            with tempfile.TemporaryDirectory() as tmp:
                out_path = Path(tmp) / "out.json"
                task = {"stage": "fit", "unit": u, "gen_unit": g, "gen": gen["result"], "permute_y": False,
                        "cache": str(store.cache_dir(CFG4["name"], gh)), "out": str(out_path)}
                (Path(tmp) / "task.json").write_text(json.dumps(task))
                cmd = ["uv", "run", "--frozen", "--project", str(ROOT / "envs" / "core"), "python",
                       str(HERE / "worker_ctr.py"), str(Path(tmp) / "task.json")]
                env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"} | {"PYTHONHASHSEED": str(cli.PYTHONHASHSEED)}
                limits.wait_for_memory(self.lim["min_free_gb"])
                started = datetime.now(timezone.utc).isoformat()
                res = limits.run(cmd, self.lim["fit_timeout_s"], self.lim["mem_gb"], tmp, env)
                result = json.loads(out_path.read_text()) if res.status == "ok" and out_path.exists() else None
                if res.status == "ok" and result is None:
                    res.status, res.error = "failed", "child exited 0 without writing out.json"
            rec = {"hash": h, "stage": "fit", "unit": u, "status": res.status, "error": res.error, "wall_s": res.wall_s,
                   "peak_rss_mb": res.peak_rss_mb, "watchdog": res.kill, "limits": self.lim, "started": started,
                   "finished": datetime.now(timezone.utc).isoformat(), "result": result, **self.base}
            store.write_json(store.record_path(CFG["name"], "fit", h), rec)
            self.counts[res.status] = self.counts.get(res.status, 0) + 1
            print(f"fit {res.status:14} {u['dataset']:>24} {u['arm']:>20} {h}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, help="default: RAM-budgeted as in exp04")
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args()
    jobs = cli.default_jobs(CFG["limits"]["mem_gb"])
    jobs = min(a.jobs, jobs) if a.jobs else jobs
    chains = plan()
    runner = Runner(store.RETRY_FAILED if a.retry_failed else set(), jobs)
    print(f"{CFG['name']}: {sum(len(u) for _, u in chains)} fit units in {len(chains)} chains, jobs={jobs}", flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda c: runner.chain(*c), chains))
    print("done:", runner.counts or "nothing to run", flush=True)


if __name__ == "__main__":
    main()
