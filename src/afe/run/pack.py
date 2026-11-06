"""Pack run records into one jsonl.gz per experiment and stage for git, and unpack them back.

Per-unit JSON (results/raw/<exp>/<stage>/<hash>.json) is the working store and is not tracked;
git tracks results/raw/<exp>/<stage>.jsonl.gz, one record per line, sorted by hash. `pack` merges the
records on disk into the existing archive (a record on disk replaces the archived one of its hash), so a
clone that was never unpacked loses nothing; to drop stale records, delete the archive and pack again.
Unfinished `*.json.tmp` writes are neither packed nor touched.
"""

import gzip
import json
import sys
from pathlib import Path

from afe.paths import RESULTS
from afe.run import store

STAGES = ("gen", "fit")


def _archived(gz: Path) -> list[dict]:
    return [json.loads(line) for line in gzip.open(gz, "rt")] if gz.exists() else []


def pack(raw: Path) -> None:
    for d in sorted(p for p in raw.glob("*/*") if p.is_dir() and p.name in STAGES):
        out = d.with_suffix(".jsonl.gz")
        recs = {r["hash"]: r for r in _archived(out)} | {r["hash"]: r for r in map(store.read, d.glob("*.json"))}
        with open(out, "wb") as raw_f, gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0) as f:
            f.write("".join(json.dumps(recs[h]) + "\n" for h in sorted(recs)).encode())
        print(out.relative_to(raw), len(recs))


def unpack(raw: Path) -> None:
    for gz in sorted(raw.glob("*/*.jsonl.gz")):
        d = gz.with_name(gz.name.removesuffix(".jsonl.gz"))
        new = 0
        for rec in _archived(gz):
            path = d / f"{rec['hash']}.json"
            if not path.exists():
                store.write_json(path, rec)
                new += 1
        print(gz.relative_to(raw), new)


if __name__ == "__main__":
    cmd, raw = sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else RESULTS / "raw"
    {"pack": pack, "unpack": unpack}[cmd](raw)
