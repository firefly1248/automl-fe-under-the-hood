"""Tidy table for exp04 (plan section 9): one row per planned fit unit, built from the run records.

`python -m afe.analysis.aggregate` writes results/tidy/exp04.parquet. A row's `status` is the first
failure along its chain (`failing_stage`: gen or fit); `n_new_features` is the downstream input width
minus the `base` arm's on the same dataset, repeat, fold and model.
"""

from pathlib import Path

import pandas as pd
import yaml

from afe.paths import RESULTS, ROOT
from afe.run import cli, store

METRICS = ("auc", "logloss", "accuracy", "rmse", "r2")
STAGES = ("full_r0", "full_r1")


def rows(cfg: dict):
    ds = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
    for stage in STAGES:
        for g, grec, fits in cli.records(cfg, stage):
            d = ds.loc[g["dataset"]]
            gres = (grec or {}).get("result") or {}
            for f, frec in fits:
                row = {
                    "dataset": g["dataset"], "family": d.family, "task": d.task, "primary": bool(d.primary),
                    "hicard": bool(d.hicard), "repeat": g["repeat"], "fold": g["fold"], "arm": f["arm"],
                    "view": f["view"], "model": f["model"], "gen_status": (grec or {}).get("status", "missing"),
                    "gen_wall_s": (grec or {}).get("wall_s"), "gen_fit_s": gres.get("fit_s"),
                    "n_features_out": gres.get("n_features_out"),
                }
                if row["gen_status"] != "ok":
                    row |= {"status": row["gen_status"], "failing_stage": "gen"}
                elif frec is None:
                    row |= {"status": "missing", "failing_stage": "fit"}
                elif frec["status"] != "ok":
                    row |= {"status": frec["status"], "failing_stage": "fit", "fit_wall_s": frec["wall_s"]}
                else:
                    r = frec["result"]
                    row |= {"status": "ok", "failing_stage": "", "fit_wall_s": frec["wall_s"],
                            "fit_s": r.get("fit_s"), "best_iteration": r.get("best_iteration"),
                            "inner_metric": r.get("inner"), "n_features": r.get("n_features"),
                            "n_train": r["n_rows"]["train"], "n_test": r["n_rows"]["test"],
                            **{m: r["test"].get(m) for m in METRICS}}
                yield row


def exp04(cfg_path: Path = ROOT / "experiments" / "exp04_open_benchmark" / "config.yaml") -> pd.DataFrame:
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    df = pd.DataFrame(rows(cfg))
    base = df[df.arm == "base"][["dataset", "repeat", "fold", "model", "status", "n_features",
                                 "n_train", "n_test", *METRICS]]
    base.columns = ["dataset", "repeat", "fold", "model", "base_status", "base_n_features",
                    "base_n_train", "base_n_test", *(f"base_{m}" for m in METRICS)]
    df = df.merge(base, on=["dataset", "repeat", "fold", "model"], how="left")
    df["n_new_features"] = df.n_features - df.base_n_features
    store.write_parquet(RESULTS / "tidy" / "exp04.parquet", df)
    return df


def timing(cfg_path: Path = ROOT / "experiments" / "exp04_open_benchmark" / "config.yaml") -> pd.DataFrame:
    """Per-unit times of the jobs=1 timing pass (plan 7.7) -> exp04_timing.csv, plus a per-arm summary
    (median and range of generator time; median of (generator + downstream) / base downstream)."""
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    rows = []
    for g, grec, fits in cli.records(cfg, "timing"):
        if grec is None:
            continue
        gres = grec.get("result") or {}
        gen_s = sum(v for k, v in gres.items() if isinstance(v, (int, float)) and k.endswith("_s"))
        for f, frec in fits:
            fres = (frec or {}).get("result") or {}
            rows.append({"dataset": g["dataset"], "arm": f["arm"], "model": f["model"],
                         "gen_status": grec["status"], "fit_status": (frec or {}).get("status", "missing"),
                         "gen_s": gen_s if grec["status"] == "ok" else None, "fit_s": fres.get("fit_s")})
    df = pd.DataFrame(rows)
    base = df[df.arm == "base"].set_index(["dataset", "model"]).fit_s.rename("base_fit_s")
    df = df.join(base, on=["dataset", "model"])
    df["ratio"] = (df.gen_s + df.fit_s) / df.base_fit_s
    df.to_csv(RESULTS / "tidy" / "exp04_timing.csv", index=False)
    g = df[df.gen_s.notna()].groupby("arm")
    summ = pd.DataFrame({"n": g.size() // 2, "gen_s_median": g.gen_s.median(), "gen_s_min": g.gen_s.min(),
                         "gen_s_max": g.gen_s.max(), "ratio_median": g.ratio.median(),
                         "ratio_max": g.ratio.max()}).round(3)
    summ.to_csv(RESULTS / "tidy" / "exp04_timing_summary.csv")
    return summ


if __name__ == "__main__":
    df = exp04()
    print(f"exp04.parquet: {len(df)} rows")
    print(df.status.value_counts().to_string())
    print(timing().to_string())
