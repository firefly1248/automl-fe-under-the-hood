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
    status: str  # ok | failed | timeout | oom (the parent may turn ok into failed or not_applicable)
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
        # stdout goes to the same log: OpenFE prints its tracebacks there before calling exit()
        p = subprocess.Popen(cmd, cwd=cwd, env={**env, **THREAD_ENV}, stdout=err, stderr=err)
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
        # SIGKILL not sent by the watchdog: the OS killed the tree for memory (macOS compressed memory
        # is invisible to the RSS watchdog). 137 = 128 + 9, as `uv run` reports a child killed by SIGKILL.
        status = status or {0: "ok", -9: "oom", 137: "oom"}.get(p.returncode, "failed")
        err.seek(0)
        text = error_text(err.read().decode(errors="replace")) if status != "ok" else ""
    if status == "timeout":
        text = f"wall clock limit {timeout_s} s exceeded\n" + text
    elif status == "oom":
        cause = "tree RSS" if peak > mem_gb * 2**30 else "killed by SIGKILL outside the watchdog"
        text = f"memory limit {mem_gb} GB exceeded ({cause})\n" + text
    return Result(status, strip_home(text), round(wall, 2), round(peak / 2**20, 1))


def error_text(log: str) -> str:
    """The log tail; if the first traceback starts before the tail, its head goes first (a library may
    print the root cause and fail differently later, e.g. OpenFE before exit())."""
    first = log.find("Traceback (most recent call last)")
    if first == -1 or first >= len(log) - ERROR_BYTES:
        return log[-ERROR_BYTES:]
    half = ERROR_BYTES // 2
    return log[first:first + half] + "\n[...]\n" + log[-half:]


def wait_for_memory(min_free_gb: float) -> None:
    """Launch gate: start a unit only when enough RAM is free."""
    while psutil.virtual_memory().available < min_free_gb * 2**30:
        time.sleep(5)
