"""Dataset registry (plan section 4.2). Runs in the core env: `make data`.

Candidates come from six OpenML suites. Metadata filters, then data filters (duplicate rows), then one
dataset per family, then a deterministic fill of the task quotas with datasets that have a categorical
column with >= 10 levels first. Adds the typed extra set, saves frames to ~/.cache/afe/data and writes
results/datasets.csv and results/datasets.md.
"""

import hashlib
import json
import re
from pathlib import Path

import openml
import pandas as pd

from afe.data.loaders import CACHE, ROOT

SUITES = {99: "CC18", 353: "CTR23", 271: "AMLB-clf", 269: "AMLB-reg", 334: "TB-cat-clf", 335: "TB-cat-reg"}
MIN_ROWS, MAX_ROWS = 1_000, 100_000
MIN_COLS, MAX_COLS = 4, 100
MAX_CLASSES = 20
MAX_DUP_FRAC = 0.05
HICARD_LEVELS = 10
QUOTAS = {"binary": 20, "multiclass": 10, "regression": 20}
IMAGE_NAMES = {"mnist_784", "Fashion-MNIST", "CIFAR_10", "Devnagari-Script", "optdigits"}
PIXEL = re.compile(r"pixel|^p\d+$")
EXCLUDE = {"electricity": "temporal order; random CV over-rewards date features"}
# datasets that are views or versions of the same data; everything else: family = normalized name
FAMILIES = [(re.compile(r"^mfeat"), "mfeat"), (re.compile(r"^(pc|kc|jm|mc|mw)\d"), "nasa_mdp"),
            (re.compile(r"^amazon"), "amazon"), (re.compile(r"^kings_county$"), "housesales"),
            (re.compile(r"^satellite$"), "satimage")]  # King County house sales; Landsat (ODDS version)
# name -> (expected default target, task kind, pinned did or None = resolve by name). Decided at A1:
# adult pinned to CC18 did 1590 (name lookup gives did 179, v1, with binned numeric columns); house_sales replaces
# Bike_Sharing_Demand (no date-typed column, only integer year/month/hour fields).
TYPED = {
    "adult": ("class", "classification", 1590),
    "bank-marketing": ("Class", "classification", None),
    "credit-g": ("class", "classification", None),
    "Amazon_employee_access": ("target", "classification", None),
    "diamonds": ("price", "regression", None),
    "house_sales": ("price", "regression", 42092),
}
DATE_LIKE = re.compile(r"date|time|year|month|day|hour|season|weekday|holiday", re.I)

log: list[str] = []


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def family(name: str) -> str:
    return next((fam for pat, fam in FAMILIES if pat.search(name.lower())), _norm(name))


def _describe(did: int, target: str, task: str, task_id: int | None, source: str, quals: pd.Series) -> dict | None:
    """Candidate record from metadata, or None with a logged reason if a filter rejects it."""
    ds = openml.datasets.get_dataset(did, download_data=False, download_qualities=False, download_features_meta_data=True)
    skip = {target, ds.row_id_attribute, *(ds.ignore_attribute or [])}
    cols = [f for f in ds.features.values() if f.name not in skip]
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
    elif ds.name in EXCLUDE:
        reason = EXCLUDE[ds.name]
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
        "family": family(ds.name),
        "task": task,
        "task_id": task_id,
        "target": target,
        "source": source,
        "rows": rows,
        "cols": len(cols),
        "date_typed_cols": ",".join(f.name for f in cols if f.data_type == "date"),
        "date_like_names": ",".join(f.name for f in cols if DATE_LIKE.search(f.name)),
    }


def _dedupe_key(r: dict) -> tuple:
    return (_norm(r["name"]), r["rows"], r["cols"], r["target"])


def _frame(r: dict) -> pd.DataFrame:
    """Download the frame and add the data-level stats to the record."""
    ds = openml.datasets.get_dataset(r["did"], download_data=True, download_qualities=False)
    df, *_ = ds.get_data()
    file = Path(ds.parquet_file or ds.data_file)
    X = df.drop(columns=r["target"])
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    r["md5_file"], r["file_format"] = hashlib.md5(file.read_bytes()).hexdigest(), file.suffix.lstrip(".")
    r["dup_frac"] = round(float(df.duplicated().mean()), 4)
    r["n_categorical"] = len(cat)
    r["max_cat_levels"] = int(max((X[c].nunique() for c in cat), default=0))
    r["hicard"] = r["max_cat_levels"] >= HICARD_LEVELS
    return df


def pool() -> list[dict]:
    tasks = {}
    for sid, sname in SUITES.items():
        for tid in openml.study.get_suite(sid).tasks:
            t = openml.tasks.get_task(tid, download_splits=False, download_data=False, download_qualities=False)
            kind = "regression" if t.task_type_id == openml.tasks.TaskType.SUPERVISED_REGRESSION else "classification"
            key = (t.dataset_id, t.target_name)
            if key in tasks:
                tasks[key][4] += f"+{sname}"
            else:
                tasks[key] = [t.dataset_id, t.target_name, kind, tid, sname]
    quals = openml.datasets.list_datasets(data_id=[t[0] for t in tasks.values()], output_format="dataframe")
    quals = quals.set_index("did")
    out, seen = [], set()
    for did, target, kind, tid, sname in sorted(tasks.values()):
        r = _describe(did, target, kind, tid, sname, quals.loc[did])
        if r is None:
            continue
        if _dedupe_key(r) in seen:
            log.append(f"duplicate {did} {r['name']}")
            continue
        seen.add(_dedupe_key(r))
        _frame(r)
        if r["dup_frac"] > MAX_DUP_FRAC:
            log.append(f"reject {did} {r['name']} ({sname}): duplicate rows {r['dup_frac']:.1%}")
            continue
        out.append(r)
    return out


def one_per_family(cands: list[dict]) -> list[dict]:
    """Keep one dataset per family: one with a >= 10-level categorical first, then the lowest id."""
    out = {}
    for c in sorted(cands, key=lambda c: (not c["hicard"], c["did"])):
        if c["family"] in out:
            log.append(f"family {c['family']}: drop {c['did']} {c['name']} (kept {out[c['family']]['did']})")
        else:
            out[c["family"]] = c
    return sorted(out.values(), key=lambda c: c["did"])


def select(cands: list[dict]) -> list[dict]:
    chosen = []
    for task, quota in QUOTAS.items():
        of_task = sorted((c for c in cands if c["task"] == task), key=lambda c: (not c["hicard"], c["did"]))
        chosen += of_task[:quota]
        if len(of_task) < quota:
            log.append(f"quota {task}: only {len(of_task)} of {quota} available")
    left = sum(c["hicard"] for c in cands if c not in chosen)
    log.append(f"with a >= {HICARD_LEVELS}-level categorical: {sum(c['hicard'] for c in chosen)} selected, {left} left over")
    return sorted(chosen, key=lambda c: c["did"])


def typed(chosen: list[dict]) -> list[dict]:
    have = {_dedupe_key(c): c for c in chosen}
    families = {c["family"] for c in chosen}
    out = []
    for name, (target, kind, did) in TYPED.items():
        ds = openml.datasets.get_dataset(did or name, download_data=False, download_qualities=False)
        if ds.default_target_attribute != target:
            log.append(f"typed {name}: default target {ds.default_target_attribute!r} != {target!r}, skipped")
            continue
        quals = openml.datasets.list_datasets(data_id=[ds.dataset_id], output_format="dataframe").set_index("did")
        r = _describe(ds.dataset_id, target, kind, None, "typed", quals.loc[ds.dataset_id])
        if r is None:
            continue
        if _dedupe_key(r) in have:
            have[_dedupe_key(r)]["source"] += "+typed"
            log.append(f"typed {name}: already selected as {have[_dedupe_key(r)]['did']}")
            continue
        _frame(r)
        r["primary"] = r["family"] not in families
        families.add(r["family"])
        out.append(r)
    return out


def save(r: dict):
    df = _frame(r)
    frames = {r["key"]: df}
    if r["name"] == "Amazon_employee_access":
        nominal = [c for c in df.columns if c != r["target"] and df[c].dtype == "category"]
        frames[r["key"] + "_int"] = df.astype({c: str for c in nominal}).astype({c: "int64" for c in nominal})
    for key, frame in frames.items():
        # the _int variant is a case study of the same data, never in the primary aggregate
        rec = {**r, "key": key, "primary": r["primary"] and key == r["key"]}
        X = frame.drop(columns=r["target"])
        rec["n_categorical"] = int(sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns))
        rec["frame_hash"] = str(int(pd.util.hash_pandas_object(frame, index=True).sum()))
        frame.to_parquet(CACHE / "data" / f"{key}.parquet")
        (CACHE / "data" / f"{key}.json").write_text(json.dumps(rec, indent=1))
        yield rec


def main() -> None:
    (CACHE / "data").mkdir(parents=True, exist_ok=True)
    openml.config.set_root_cache_directory(str(CACHE / "openml"))
    chosen = select(one_per_family(pool()))
    for c in chosen:
        c["primary"] = True
    records, keys = [], set()
    for r in chosen + typed(chosen):
        if r["key"] in keys:  # same name, different dataset (e.g. two house_sales versions)
            r["key"] = f"{r['name']}_{r['did']}"
        keys.add(r["key"])
        records += list(save(r))
    df = pd.DataFrame(records)
    df.to_csv(ROOT / "results" / "datasets.csv", index=False)
    prim = df[df.primary]
    cols = ["key", "did", "version", "family", "task", "rows", "cols", "n_categorical", "max_cat_levels", "dup_frac",
            "primary", "source", "md5_file", "frame_hash", "date_typed_cols", "date_like_names"]
    md = [
        "# Datasets",
        "",
        "Written by `make data` (`src/afe/data/registry.py`). Pool: OpenML suites "
        + ", ".join(f"{k} ({v})" for k, v in SUITES.items())
        + ", plus the typed extra set. `cols` excludes the target, row id and ignored columns. `primary`: in the "
        "primary aggregate (one dataset per family). `max_cat_levels`: most levels of any categorical column. "
        "`dup_frac`: share of exact duplicate rows (target included). `md5_file`: md5 of the file as downloaded "
        "from OpenML; `frame_hash`: `pd.util.hash_pandas_object(df, index=True).sum()` of the loaded frame. "
        "`date_typed_cols`: OpenML type `date`; `date_like_names`: a name regex, informational only.",
        "",
        f"Primary: {len(prim)} datasets, by task {prim.groupby('task').size().to_dict()}; with a categorical column "
        f"of >= {HICARD_LEVELS} levels: {int(prim.hicard.sum())} ({prim.hicard.mean():.0%}). "
        f"Case studies (not primary): {int((~df.primary).sum())}.",
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
