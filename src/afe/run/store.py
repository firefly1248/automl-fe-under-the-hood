"""Run store: atomic records and generator-output cache (plan section 5)."""

import json
import os
from pathlib import Path

import pandas as pd

from afe.paths import RESULTS

DONE = {"ok", "failed", "timeout", "oom", "not_applicable"}
RETRY_FAILED = {"failed", "timeout", "oom"}
SPLITS = ("train", "inner", "test")  # train = fit rows


def record_path(exp: str, stage: str, h: str) -> Path:
    return RESULTS / "raw" / exp / stage / f"{h}.json"


def cache_dir(exp: str, gen_hash: str) -> Path:
    return RESULTS / "cache" / exp / gen_hash


def _atomic(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json(path: Path, obj: dict) -> None:
    _atomic(path, lambda f: f.write(json.dumps(obj, indent=1, default=str).encode()))


def write_parquet(path: Path, df: pd.DataFrame) -> None:
    _atomic(path, lambda f: df.to_parquet(f))


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def is_done(rec: dict | None, retry: set[str]) -> bool:
    """Finished, unless its status is one the caller asked to rerun."""
    return rec is not None and rec["status"] in DONE and rec["status"] not in retry


def cache_complete(exp: str, gen_hash: str) -> bool:
    return all((cache_dir(exp, gen_hash) / f"{s}.parquet").exists() for s in SPLITS)
