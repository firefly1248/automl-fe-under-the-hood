"""Full-run time estimate from the pilot records (plan section 7.8). Core env:

    uv run --project envs/core python experiments/exp04_open_benchmark/estimate.py

Per arm and stage (generator, or downstream model), unit wall time of a primary dataset is
interpolated in log rows x columns between the median times of the pilot datasets (timeouts count at
the cap), summed over folds. Wall clock = unit-hours / jobs. Pilot times come from the parallel run, so
they are for budgeting only (section 7.7). Writes results/tidy/exp04_estimate.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from afe.paths import RESULTS
from afe.run import cli, store
from afe.run.unit import digest

CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
LIM = CFG["limits"]
D = pd.read_csv(RESULTS / "datasets.csv").set_index("key")
SIZE = D.rows * D.cols

rows = []
for g, fits in cli.plan(CFG, "pilot"):
    gr = store.read(store.record_path(CFG["name"], "gen", digest(g)))
    if gr is None:
        continue
    rows.append((g["dataset"], g["arm"], "gen", gr["wall_s"]))
    for f in fits:
        fr = store.read(store.record_path(CFG["name"], "fit", digest(f)))
        if fr:
            rows.append((g["dataset"], f["arm"], f["model"], fr["wall_s"]))
t = pd.DataFrame(rows, columns=["dataset", "arm", "stage", "wall_s"])
t["size"] = t.dataset.map(SIZE)

primary = D[D.primary].index
excluded = {(e["dataset"], e["model"]) for e in CFG.get("exclude", [])}
out = []
for (arm, stage), s in t.groupby(["arm", "stage"]):
    cap = LIM["gen_timeout_s"] if stage == "gen" else LIM["fit_timeout_s"]
    m = s.groupby("dataset").agg(wall=("wall_s", "median"), size=("size", "first")).sort_values("size")
    keys = [k for k in primary if (k, stage) not in excluded]
    # a power law misses step-like arms (autofeat: seconds on diamonds, the cap on the smaller mfeat-fourier)
    pred = np.exp(np.interp(np.log(SIZE[keys]), np.log(m["size"]), np.log(m.wall.clip(lower=0.1))))
    new = ~pd.Index(keys).isin(t.dataset.unique())  # the pilot datasets are already done for 5 folds
    out.append({"arm": arm, "stage": stage, "pilot_h": s.wall_s.sum() / 3600, "unit_h_per_fold": pred.sum() / 3600,
                "unit_h_per_fold_not_in_pilot": pred[new].sum() / 3600,
                "datasets_near_cap": int((pred >= cap / 2).sum())})
e = pd.DataFrame(out)
jobs = cli.default_jobs(LIM["mem_gb"])
total, rest = e.unit_h_per_fold.sum(), e.unit_h_per_fold_not_in_pilot.sum()
md = [
    "# exp04 full-run estimate", "",
    f"From {t.dataset.nunique()} pilot datasets; {len(primary)} primary datasets; jobs = {jobs}. "
    f"Unit-hours per fold over all arms and stages: {total:.0f}; the pilot took {e.pilot_h.sum():.0f} over "
    f"{CFG['protocol']['n_folds']} folds. Ten pilot points per arm and stage, so a rough budget: autofeat "
    "time depends on more than rows x columns, and every primary dataset it reaches near the cap adds up to "
    "an hour per fold.", "",
    "| splits | unit-hours, all | wall clock, all | unit-hours left after the pilot | wall clock left |",
    "|---|---|---|---|---|",
    # with 10 splits the pilot datasets also need 5 more folds
    *[f"| {k} | {total * k:.0f} | {total * k / jobs:.0f} h | {rest * 5 + total * (k - 5):.0f} | "
      f"{(rest * 5 + total * (k - 5)) / jobs:.0f} h |" for k in (5, 10)], "",
    "Per arm and stage (unit-hours per fold, summed over primary datasets):", "",
    e.sort_values("unit_h_per_fold", ascending=False).round(2).to_markdown(index=False), "",
]
(RESULTS / "tidy" / "exp04_estimate.md").write_text("\n".join(md))
print("\n".join(md))
