"""`make claims` (plan section 9): recompute every value in results/claims.yaml from the tidy tables.

Number entries name a source file (csv, parquet or a yaml flattened to one row), a pandas query (optional for
a yaml) and a stat (a column of the single matching row, or `count`). Figure entries name the figure and its rendered data table; the claim value is the table's
sha1. The script writes the recomputed `value`/`ci`/`coverage` (or `data_hash`) back and exits 1 if any
recomputed value differs from its `approved_value` (`approved_data_hash`) by more than `tolerance`
(exactly, for hashes). `approved_*` fields are set only by Ilia and are never written here.
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd
import yaml

from afe.paths import RESULTS


def load_source(path: str) -> pd.DataFrame:
    p = RESULTS.parent / path
    if p.suffix == ".yaml":  # one row, a column per leaf: `tasks.binary.selection_params.mode`
        return pd.json_normalize(yaml.safe_load(p.read_text()))
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def number(c: dict) -> dict:
    df = load_source(c["source"])
    df = df.query(c["query"], engine="python") if c.get("query") else df  # python: allows .str methods
    if c["stat"] == "count":
        return {"value": int(len(df))}
    if len(df) != 1:
        raise ValueError(f"{c['id']}: query matched {len(df)} rows, need exactly 1")
    row = df.iloc[0]
    out = {"value": round(float(row[c["stat"]]), 6)}
    if {"ci_lo", "ci_hi"} <= set(df.columns) and c["stat"] in ("effect", "median_effect"):  # a CI belongs to an effect, not to a p-value
        out["ci"] = [round(float(row.ci_lo), 6), round(float(row.ci_hi), 6)]
    if "coverage" in df.columns and c["stat"] != "coverage":
        out["coverage"] = round(float(row.coverage), 4)
    return out


def figure(c: dict) -> dict:
    return {"data_hash": hashlib.sha1((RESULTS.parent / c["data"]).read_bytes()).hexdigest()[:16]}


def main() -> None:
    path = RESULTS / "claims.yaml"
    claims = yaml.safe_load(path.read_text()) or []
    failed = []
    for c in claims:
        c.update(figure(c) if c.get("kind") == "figure" else number(c))
        if c.get("kind") == "figure":
            ok = c.get("approved_data_hash") in (None, c["data_hash"])
        else:
            ok = c.get("approved_value") is None or abs(c["value"] - c["approved_value"]) <= c.get("tolerance", 0.001)
        if not ok:
            failed.append(c["id"])
    path.write_text(yaml.safe_dump(claims, sort_keys=False, allow_unicode=True, width=110))
    print(f"{len(claims)} claims recomputed, {len(failed)} differ from approved: {failed or ''}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
