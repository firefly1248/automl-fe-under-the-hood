"""Orchestrator (core env): `python -m afe.run.cli <exp config.yaml> <stage> [--jobs N] [--retry-failed]
[--retry-status S ...]`.

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
from afe.generators import ARMS, FIT_ONLY, env_of
from afe.machine import machine_info
from afe.paths import CACHE, RESULTS, ROOT
from afe.run import limits, store
from afe.run.unit import code_hash, digest, file_hash, fit_unit, gen_unit

SRC = ROOT / "src" / "afe"
# shared code that affects results (plan section 5); orchestration, limits and logging are not hashed
SHARED_CODE = (SRC / "generators" / "base.py", SRC / "run" / "worker.py", SRC / "data" / "loaders.py",
               SRC / "data" / "splits.py")
FIT_CODE = (*SHARED_CODE, SRC / "models")
RESERVED_GB = 4  # left for the system and other apps (plan section 5)
SLOW_FIRST = ("autofeat", "openfe", "openfe_transductive")  # launched first so no slot idles at the end of a stage
# str/set hash order is per process and autofeat's output depends on it (A3 review: two PYTHONHASHSEED values,
# every arm and model); fixed for every child and part of the protocol hash
PYTHONHASHSEED = 0


def default_jobs(mem_gb: float) -> int:
    """min(physical cores - 2, units whose memory caps fit in RAM minus the reserve)."""
    ram_gb = psutil.virtual_memory().total / 2**30
    return max(1, min(psutil.cpu_count(logical=False) - 2, int((ram_gb - RESERVED_GB) // mem_gb)))


def git_sha() -> str:
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return sha + ("-dirty" if dirty else "")


def arm_code(arm: str) -> str:
    """The arm's own module and registry entry plus the shared code: a fix to one wrapper reruns only that arm."""
    module = ROOT / "src" / Path(*ARMS[arm][0].split(".")).with_suffix(".py")
    return digest([code_hash(module, *SHARED_CODE), ARMS[arm]])


def exp_key(cfg: dict, stage: str) -> str:
    """A stage may run under its own experiment key (e.g. the timing pass) so it never reuses main-run records."""
    return cfg["stages"][stage].get("exp", cfg["name"])


def excluded(cfg: dict) -> set[tuple[str, str]]:
    """(dataset, model) pairs not run."""
    return {(e["dataset"], e["model"]) for e in cfg.get("exclude", [])}


def records(cfg: dict, stage: str):
    """(gen unit, gen record, [(fit unit, fit record)]) for every planned chain; records may be None."""
    read = lambda s, u: store.read(store.record_path(exp_key(cfg, stage), s, digest(u)))  # noqa: E731
    for g, fits in plan(cfg, stage):
        yield g, read("gen", g), [(f, read("fit", f)) for f in fits]


def plan(cfg: dict, stage: str) -> list[tuple[dict, list[dict]]]:
    """[(gen unit, [fit units])] for one stage. A fit-only arm (ag_tabprep_keepcat) adds fit units to the
    chain of its source arm."""
    proto, st = cfg["protocol"], cfg["stages"][stage]
    protocol_hash = digest({**proto, "pythonhashseed": PYTHONHASHSEED})
    core_lock = file_hash(ROOT / "envs" / "core" / "uv.lock")
    fit_code = code_hash(*FIT_CODE)
    datasets = st["datasets"]
    if datasets == "primary":
        d = pd.read_csv(RESULTS / "datasets.csv")
        datasets = d[d.primary].key.tolist()
    arms = st.get("arms", cfg["arms"])
    skip = excluded(cfg)
    chains = []
    for key in datasets:
        data_hash = json.loads((CACHE / "data" / f"{key}.json").read_text())["frame_hash"]
        splits_hash = S.splits_hash(S.load(key))
        for arm in (a for a in arms if a not in FIT_ONLY):
            lock, gen_code = file_hash(ROOT / "envs" / env_of(arm) / "uv.lock"), arm_code(arm)
            views = [(arm, "default")] + [(a, "keepcat") for a in arms if FIT_ONLY.get(a) == arm]
            for r in st.get("repeats", range(proto["repeats"])):
                for k in st.get("folds", range(proto["n_folds"])):
                    g = gen_unit(exp_key(cfg, stage), key, r, k, arm, cfg.get("arm_params", {}).get(arm, {}),
                                 protocol_hash, data_hash, splits_hash, lock, gen_code)
                    fits = [fit_unit(digest(g), a, v, m, p, S.tool_seed(r, k), core_lock, fit_code)
                            for a, v in views for m, p in proto["models"].items() if (key, m) not in skip]
                    chains.append((g, fits))
    return chains


class Runner:
    def __init__(self, cfg: dict, exp: str, retry: set[str], jobs: int):
        self.cfg, self.exp, self.retry = cfg, exp, retry
        self.lim = cfg["limits"]
        self.permute_y = cfg["protocol"].get("permute_y", False)  # in protocol_hash, so in every unit hash
        self.base = {"git_sha": git_sha(), "machine": machine_info(), "jobs": jobs}
        self.lock = threading.Lock()
        self.counts: dict[str, int] = {}

    def _launch(self, env: str, task: dict, timeout_s: float) -> tuple[limits.Result, dict]:
        with tempfile.TemporaryDirectory() as tmp:  # per-unit temp dir, also the child's cwd
            out_path = Path(tmp) / "out.json"
            task["out"], task["permute_y"] = str(out_path), self.permute_y
            (Path(tmp) / "task.json").write_text(json.dumps(task))
            cmd = ["uv", "run", "--frozen", "--project", str(ROOT / "envs" / env), "python", "-m", "afe.run.worker",
                   str(Path(tmp) / "task.json")]
            env_vars = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"} | {"PYTHONHASHSEED": str(PYTHONHASHSEED)}
            limits.wait_for_memory(self.lim["min_free_gb"])
            started = datetime.now(timezone.utc).isoformat()
            res = limits.run(cmd, timeout_s, self.lim["mem_gb"], tmp, env_vars)
            out = None
            if res.status == "ok":
                if out_path.exists():
                    out = json.loads(out_path.read_text())
                    if out.get("not_applicable"):
                        res.status = "not_applicable"
                else:  # e.g. a library called exit() and the process ended with code 0
                    res.status, res.error = "failed", "child exited 0 without writing out.json"
        return res, {"started": started, "finished": datetime.now(timezone.utc).isoformat(), "result": out}

    def _record(self, stage: str, h: str, unit: dict, res: limits.Result, extra: dict) -> dict:
        rec = {"hash": h, "stage": stage, "unit": unit, "status": res.status, "error": res.error,
               "wall_s": res.wall_s, "peak_rss_mb": res.peak_rss_mb, "watchdog": res.kill, "limits": self.lim,
               **extra, **self.base}
        store.write_json(store.record_path(self.exp, stage, h), rec)
        with self.lock:
            self.counts[res.status] = self.counts.get(res.status, 0) + 1
            what = unit.get("dataset", unit.get("model"))
            print(f"{stage} {res.status:14} {what:>24} {unit['arm']:>20} {h}", flush=True)
        return rec

    def chain(self, g: dict, fits: list[dict]) -> None:
        gh = digest(g)
        rec = store.read(store.record_path(self.exp, "gen", gh))
        if not store.is_done(rec, self.retry) or (rec["status"] == "ok" and not store.cache_complete(self.exp, gh)):
            task = {"stage": "gen", "unit": g, "cache": str(store.cache_dir(self.exp, gh))}
            res, extra = self._launch(env_of(g["arm"]), task, self.lim["gen_timeout_s"])
            rec = self._record("gen", gh, g, res, extra)
        if rec["status"] != "ok":
            return
        for f in fits:
            fh = digest(f)
            if store.is_done(store.read(store.record_path(self.exp, "fit", fh)), self.retry):
                continue
            task = {"stage": "fit", "unit": f, "gen_unit": g, "gen": rec["result"],
                    "cache": str(store.cache_dir(self.exp, gh))}
            res, extra = self._launch("core", task, self.lim["fit_timeout_s"])
            self._record("fit", fh, f, res, extra)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("stage")
    ap.add_argument("--jobs", type=int, help="default: RAM-budgeted (plan section 5); may only lower it")
    ap.add_argument("--retry-failed", action="store_true", help="rerun failed, timeout and oom units")
    ap.add_argument("--retry-status", nargs="+", default=[], choices=sorted(store.RETRY_FAILED),
                    help="rerun only units with these statuses (e.g. timeout after a cap is raised)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(a.config).read_text())
    jobs = default_jobs(cfg["limits"]["mem_gb"])
    jobs = min(a.jobs, jobs) if a.jobs else jobs
    retry = (store.RETRY_FAILED if a.retry_failed else set()) | set(a.retry_status)
    chains = sorted(plan(cfg, a.stage), key=lambda c: c[0]["arm"] not in SLOW_FIRST)  # stable: plan order otherwise
    runner = Runner(cfg, exp_key(cfg, a.stage), retry, jobs)
    print(f"{runner.exp} {a.stage}: {len(chains)} generator units, jobs={jobs}", flush=True)
    with ThreadPoolExecutor(jobs) as pool:
        list(pool.map(lambda c: runner.chain(*c), chains))
    print("done:", runner.counts or "nothing to run", flush=True)


if __name__ == "__main__":
    main()
