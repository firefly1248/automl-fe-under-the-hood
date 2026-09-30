"""Orchestrator (core env): `python -m afe.run.cli <exp config.yaml> <stage> [--jobs N] [--retry-failed]`.

Plans generator and fit units from the experiment config, skips finished units, runs each unit as a
child process in its env under limits, and writes one record per unit. Safe to kill and rerun.
"""

import argparse
import json
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import psutil
import yaml

from afe.data import splits as S
from afe.generators import env_of
from afe.machine import machine_info
from afe.paths import CACHE, RESULTS, ROOT
from afe.run import limits, store
from afe.run.unit import digest, file_hash, fit_unit, gen_unit

def default_jobs() -> int:
    return min(psutil.cpu_count(logical=False) - 2, 10)


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def plan(cfg: dict, stage: str) -> list[tuple[dict, list[dict]]]:
    """[(gen unit, [fit units])] for one stage."""
    proto, st = cfg["protocol"], cfg["stages"][stage]
    protocol_hash = digest(proto)
    core_lock = file_hash(ROOT / "envs" / "core" / "uv.lock")
    datasets = st["datasets"]
    if datasets == "primary":
        d = pd.read_csv(RESULTS / "datasets.csv")
        datasets = d[d.primary].key.tolist()
    chains = []
    for key in datasets:
        data_hash = json.loads((CACHE / "data" / f"{key}.json").read_text())["frame_hash"]
        splits_hash = S.splits_hash(S.load(key))
        for arm in st.get("arms", cfg["arms"]):
            lock = file_hash(ROOT / "envs" / env_of(arm) / "uv.lock")
            for r in range(proto["repeats"]):
                for k in st.get("folds", range(proto["n_folds"])):
                    g = gen_unit(cfg["name"], key, r, k, arm, cfg.get("arm_params", {}).get(arm, {}),
                                 protocol_hash, data_hash, splits_hash, lock)
                    fits = [fit_unit(digest(g), arm, "default", m, p, S.tool_seed(r, k), core_lock)
                            for m, p in proto["models"].items()]
                    chains.append((g, fits))
    return chains


class Runner:
    def __init__(self, cfg: dict, retry_failed: bool):
        self.cfg, self.retry = cfg, retry_failed
        self.lim = cfg["limits"]
        self.base = {"git_sha": git_sha(), "machine": machine_info()}
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def _launch(self, env: str, task: dict, timeout_s: float) -> tuple[limits.Result, dict | None]:
        with tempfile.TemporaryDirectory() as tmp:  # per-unit temp dir, also the child's cwd
            task["out"] = str(Path(tmp) / "out.json")
            (Path(tmp) / "task.json").write_text(json.dumps(task))
            cmd = ["uv", "run", "--frozen", "--project", str(ROOT / "envs" / env), "python", "-m", "afe.run.worker",
                   str(Path(tmp) / "task.json")]
            env_vars = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
            limits.wait_for_memory(self.lim["min_free_gb"])
            started = datetime.now(timezone.utc).isoformat()
            res = limits.run(cmd, timeout_s, self.lim["mem_gb"], tmp, env_vars)
            out = json.loads(Path(task["out"]).read_text()) if res.status == "ok" else None
        return res, {"started": started, "finished": datetime.now(timezone.utc).isoformat(), "result": out}

    def _record(self, stage: str, h: str, unit: dict, res: limits.Result, extra: dict) -> dict:
        rec = {"hash": h, "stage": stage, "unit": unit, "status": res.status, "error": res.error,
               "wall_s": res.wall_s, "peak_rss_mb": res.peak_rss_mb, "limits": self.lim, **extra, **self.base}
        store.write_json(store.record_path(self.cfg["name"], stage, h), rec)
        with self.lock:
            self.counts[res.status] = self.counts.get(res.status, 0) + 1
            print(f"{stage} {res.status:7} {unit.get('dataset', unit.get('model')):>24} {unit['arm']:>12} {h}", flush=True)
        return rec

    def chain(self, g: dict, fits: list[dict]) -> None:
        exp, gh = self.cfg["name"], digest(g)
        rec = store.read(store.record_path(exp, "gen", gh))
        if not store.is_done(rec, self.retry) or (rec["status"] == "ok" and not store.cache_complete(exp, gh)):
            task = {"stage": "gen", "unit": g, "cache": str(store.cache_dir(exp, gh))}
            res, extra = self._launch(env_of(g["arm"]), task, self.lim["gen_timeout_s"])
            rec = self._record("gen", gh, g, res, extra)
        if rec["status"] != "ok":
            return
        for f in fits:
            fh = digest(f)
            if store.is_done(store.read(store.record_path(exp, "fit", fh)), self.retry):
                continue
            task = {"stage": "fit", "unit": f, "gen_unit": g, "gen": rec["result"], "cache": str(store.cache_dir(exp, gh))}
            res, extra = self._launch("core", task, self.lim["fit_timeout_s"])
            self._record("fit", fh, f, res, extra)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("stage")
    ap.add_argument("--jobs", type=int, default=default_jobs())
    ap.add_argument("--retry-failed", action="store_true")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    chains = plan(cfg, a.stage)
    runner = Runner(cfg, a.retry_failed)
    print(f"{cfg['name']} {a.stage}: {len(chains)} generator units, jobs={a.jobs}", flush=True)
    with ThreadPoolExecutor(a.jobs) as pool:
        list(pool.map(lambda c: runner.chain(*c), chains))
    print("done:", runner.counts or "nothing to run", flush=True)


if __name__ == "__main__":
    main()
