"""Dataset registry (plan section 4.2). Runs in the core env: `make data`.

Selects datasets from OpenML-CC18 (suite 99) and OpenML-CTR23 (suite 353) with fixed filters and a
deterministic fill, adds the typed extra set, downloads everything to ~/.cache/afe/data and writes
results/datasets.csv and results/datasets.md.
"""

import hashlib
import json
import re
from pathlib import Path

import openml
import pandas as pd

from afe.data.loaders import CACHE, ROOT

SUITES = {99: "CC18", 353: "CTR23"}
MIN_ROWS, MAX_ROWS = 1_000, 100_000
MIN_COLS, MAX_COLS = 4, 100
MAX_CLASSES = 20
QUOTAS = {"binary": 12, "multiclass": 6, "regression": 12}
MIN_WITH_CATEGORICALS = 8
IMAGE_NAMES = {"mnist_784", "Fashion-MNIST", "CIFAR_10", "Devnagari-Script"}
PIXEL = re.compile(r"pixel|^p\d+$")
# name -> (expected default target, task kind)
TYPED = {
    "adult": ("class", "classification"),
    "bank-marketing": ("Class", "classification"),
    "credit-g": ("class", "classification"),
    "Amazon_employee_access": ("target", "classification"),
    "diamonds": ("price", "regression"),
    "Bike_Sharing_Demand": ("count", "regression"),
}
DATE_LIKE = re.compile(r"date|time|year|month|day|hour|season|weekday|holiday", re.I)

log: list[str] = []


def _features(did: int) -> tuple[openml.OpenMLDataset, list]:
    ds = openml.datasets.get_dataset(did, download_data=False, download_qualities=False, download_features_meta_data=True)
    return ds, list(ds.features.values())


def _describe(did: int, target: str, task: str, task_id: int | None, source: str, quals: pd.Series) -> dict | None:
    """Candidate record, or None with a logged reason if a filter rejects it."""
    ds, feats = _features(did)
    skip = {target, ds.row_id_attribute, *(ds.ignore_attribute or [])}
    cols = [f for f in feats if f.name not in skip]
    rows = int(quals["NumberOfInstances"])
    n_classes = int(quals["NumberOfClasses"]) if task != "regression" else 0
    reason = None
    if not MIN_ROWS <= rows <= MAX_ROWS:
        reason = f"rows={rows}"
    elif not MIN_COLS <= len(cols) <= MAX_COLS:
        reason = f"cols={len(cols)}"
    elif task != "regression" and n_classes > MAX_CLASSES:
        reason = f"classes={n_classes}"
    elif ds.name in IMAGE_NAMES or sum(bool(PIXEL.search(f.name)) for f in cols) > 0.9 * len(cols):
        reason = "image"
    if reason:
        log.append(f"reject {did} {ds.name} ({source}): {reason}")
        return None
    if task != "regression":
        task = "binary" if n_classes == 2 else "multiclass"
    return {
        "key": ds.name,
        "did": did,
        "version": int(ds.version),
        "name": ds.name,
        "task": task,
        "task_id": task_id,
        "target": target,
        "source": source,
        "rows": rows,
        "cols": len(cols),
        "n_categorical_meta": sum(f.data_type in ("nominal", "string") for f in cols),
        "date_typed_cols": ",".join(f.name for f in cols if f.data_type == "date"),
        "date_like_names": ",".join(f.name for f in cols if DATE_LIKE.search(f.name)),
    }


def _dedupe_key(r: dict) -> tuple:
    return (re.sub(r"[^a-z0-9]", "", r["name"].lower()), r["rows"], r["cols"], r["target"])


def pool() -> list[dict]:
    tasks = []
    for sid, sname in SUITES.items():
        for tid in openml.study.get_suite(sid).tasks:
            t = openml.tasks.get_task(tid, download_splits=False, download_data=False, download_qualities=False)
            kind = "regression" if t.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION else "classification"
            tasks.append((t.dataset_id, t.target_name, kind, tid, sname))
    quals = openml.datasets.list_datasets(data_id=[t[0] for t in tasks], output_format="dataframe").set_index("did")
    out, seen = [], set()
    for did, target, kind, tid, sname in sorted(tasks):
        r = _describe(did, target, kind, tid, sname, quals.loc[did])
        if r is None:
            continue
        if _dedupe_key(r) in seen:
            log.append(f"duplicate {did} {r['name']}")
            continue
        seen.add(_dedupe_key(r))
        out.append(r)
    return out


def select(cands: list[dict]) -> list[dict]:
    chosen = []
    for task, quota in QUOTAS.items():
        chosen += [c for c in cands if c["task"] == task][:quota]
    while sum(c["n_categorical_meta"] > 0 for c in chosen) < MIN_WITH_CATEGORICALS:
        swapped = False
        for out in sorted((c for c in chosen if c["n_categorical_meta"] == 0), key=lambda c: -c["did"]):
            repl = next(
                (c for c in cands if c not in chosen and c["task"] == out["task"] and c["n_categorical_meta"] > 0), None
            )
            if repl:
                chosen[chosen.index(out)] = repl
                log.append(f"swap {out['did']} {out['name']} -> {repl['did']} {repl['name']} ({out['task']})")
                swapped = True
                break
        if not swapped:
            log.append("could not reach the categorical minimum")
            break
    return sorted(chosen, key=lambda c: c["did"])


def typed(chosen: list[dict]) -> list[dict]:
    have = {_dedupe_key(c) for c in chosen}
    out = []
    for name, (target, kind) in TYPED.items():
        ds = openml.datasets.get_dataset(name, download_data=False, download_qualities=False)
        if ds.default_target_attribute != target:
            log.append(f"typed {name}: default target {ds.default_target_attribute!r} != {target!r}, skipped")
            continue
        quals = openml.datasets.list_datasets(data_id=[ds.dataset_id], output_format="dataframe").set_index("did")
        r = _describe(ds.dataset_id, target, kind, None, "typed", quals.loc[ds.dataset_id])
        if r is None:
            continue
        if _dedupe_key(r) in have:
            match = next(c for c in chosen if _dedupe_key(c) == _dedupe_key(r))
            match["source"] += "+typed"
            log.append(f"typed {name}: already selected as {match['did']}")
            continue
        out.append(r)
    return out


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def download(r: dict):
    ds = openml.datasets.get_dataset(r["did"], download_data=True, download_qualities=False)
    df, *_ = ds.get_data()
    file = Path(ds.parquet_file or ds.data_file)
    r["md5_file"], r["file_format"] = _md5(file), file.suffix.lstrip(".")
    frames = {r["key"]: df}
    if r["name"] == "Amazon_employee_access":
        nominal = [c for c in df.columns if c != r["target"] and df[c].dtype == "category"]
        frames[r["key"] + "_int"] = df.astype({c: str for c in nominal}).astype({c: "int64" for c in nominal})
    for key, frame in frames.items():
        rec = {**r, "key": key}
        rec["n_categorical"] = int(
            sum(not pd.api.types.is_numeric_dtype(frame[c]) for c in frame.columns if c != r["target"])
        )
        rec["frame_hash"] = str(int(pd.util.hash_pandas_object(frame, index=True).sum()))
        frame.to_parquet(CACHE / "data" / f"{key}.parquet")
        (CACHE / "data" / f"{key}.json").write_text(json.dumps(rec, indent=1))
        yield rec


def main() -> None:
    (CACHE / "data").mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(CACHE / "openml"))
    chosen = select(pool())
    records = []
    for r in chosen + typed(chosen):
        records += list(download(r))
    df = pd.DataFrame(records)
    df.to_csv(ROOT / "results" / "datasets.csv", index=False)
    cols = ["key", "did", "version", "task", "rows", "cols", "n_categorical", "source", "md5_file", "frame_hash", "date_typed_cols", "date_like_names"]
    counts = df.groupby("task").size().to_dict()
    md = [
        "# Datasets",
        "",
        "Written by `make data` (`src/afe/data/registry.py`). Source: OpenML-CC18 (suite 99), "
        "OpenML-CTR23 (suite 353) and the typed extra set. `cols` excludes the target, row id and ignored columns. "
        "`md5_file` is the md5 of the file as downloaded from OpenML; `frame_hash` is "
        "`pd.util.hash_pandas_object(df, index=True).sum()` of the loaded frame (target included). "
        "`date_typed_cols`: OpenML type `date`; `date_like_names`: a name regex, informational only.",
        "",
        f"Counts by task: {counts}. With categoricals: {int((df.n_categorical > 0).sum())}.",
        "",
        df[cols].to_markdown(index=False),
        "",
        "## Selection log",
        "",
        *[f"- {line}" for line in log],
        "",
    ]
    (ROOT / "results" / "datasets.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
