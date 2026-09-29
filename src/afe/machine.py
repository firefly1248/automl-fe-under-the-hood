"""Machine info and privacy helpers (plan rule 7). Imports only stdlib and psutil."""

import os
import platform
import subprocess
from importlib import metadata

import psutil

HOME = os.path.expanduser("~")


def strip_home(text: str) -> str:
    return text.replace(HOME, "~")


def cpu_model() -> str:
    if platform.system() == "Darwin":
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def machine_info() -> dict:
    """CPU model, core counts, RAM and OS only. No hostname, no user name."""
    return {
        "cpu": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
    }


def versions(dists: list[str]) -> dict:
    out = {"python": platform.python_version()}
    for d in dists:
        try:
            out[d] = metadata.version(d)
        except metadata.PackageNotFoundError:
            out[d] = None
    return out
