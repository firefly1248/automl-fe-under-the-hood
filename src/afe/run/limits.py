"""Wall-clock and memory limits for one child process tree (plan section 5).

Memory is enforced by polling the RSS of the whole tree with psutil and killing it above the cap
(RLIMIT_AS is not used: it kills torch/numba/LightGBM early).
"""

import subprocess
import tempfile
import time
from dataclasses import dataclass

import psutil

from afe.machine import strip_home

THREAD_ENV = {v: "1" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
POLL_S = 0.2
ERROR_BYTES = 5000


@dataclass
class Result:
    status: str  # ok | failed | timeout | oom
    error: str
    wall_s: float
    peak_rss_mb: float


def _tree(proc: psutil.Process) -> list[psutil.Process]:
    try:
        return [proc, *proc.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []


def _rss(procs) -> int:
    total = 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def kill_tree(procs) -> None:
    for p in procs:
        try:
            p.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(procs, timeout=10)


def run(cmd: list[str], timeout_s: float, mem_gb: float, cwd, env: dict) -> Result:
    with tempfile.TemporaryFile() as err:
        t0 = time.monotonic()
        p = subprocess.Popen(cmd, cwd=cwd, env={**env, **THREAD_ENV}, stdout=subprocess.DEVNULL, stderr=err)
        proc, peak, status = psutil.Process(p.pid), 0, None
        while p.poll() is None:
            procs = _tree(proc)
            peak = max(peak, _rss(procs))
            if peak > mem_gb * 2**30:
                status = "oom"
            elif time.monotonic() - t0 > timeout_s:
                status = "timeout"
            if status:
                kill_tree(procs)
                p.wait()
                break
            time.sleep(POLL_S)
        wall = time.monotonic() - t0
        status = status or ("ok" if p.returncode == 0 else "failed")
        err.seek(0)
        text = err.read()[-ERROR_BYTES:].decode(errors="replace") if status != "ok" else ""
    if status == "timeout":
        text = f"wall clock limit {timeout_s} s exceeded\n" + text
    elif status == "oom":
        text = f"memory limit {mem_gb} GB exceeded (tree RSS)\n" + text
    return Result(status, strip_home(text)[-ERROR_BYTES:], round(wall, 2), round(peak / 2**20, 1))


def wait_for_memory(min_free_gb: float) -> None:
    """Launch gate: start a unit only when enough RAM is free."""
    while psutil.virtual_memory().available < min_free_gb * 2**30:
        time.sleep(5)
