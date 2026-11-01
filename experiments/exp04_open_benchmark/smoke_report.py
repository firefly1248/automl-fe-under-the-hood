"""Smoke report for A2 (plan section 13): per arm and dataset, generator status, features, time, memory,
and the test metric of each downstream model next to `base`. Writes results/tidy/exp04_smoke.md.

    uv run --project envs/core python experiments/exp04_open_benchmark/smoke_report.py
"""

from pathlib import Path

import pandas as pd
import yaml

from afe.paths import RESULTS
from afe.run import store
from afe.run.cli import exp_key, plan
from afe.run.unit import digest

CFG = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
METRIC = {"binary": "auc", "multiclass": "logloss", "regression": "rmse"}


def main() -> None:
    """Only the units the current code plans for the smoke stage (older records have other hashes)."""
    exp, chains = exp_key(CFG, "smoke"), plan(CFG, "smoke")
    read = lambda stage, u: store.read(store.record_path(exp, stage, digest(u)))  # noqa: E731
    tasks = pd.read_csv(RESULTS / "datasets.csv").set_index("key").task
    gens = {digest(g): r for g, _ in chains if (r := read("gen", g))}
    fit_records = [r for _, fits in chains for f in fits if (r := read("fit", f))]
    rows = []
    for g in gens.values():
        u, res = g["unit"], g["result"] or {}
        rows.append({"dataset": u["dataset"], "arm": u["arm"], "stage": "gen", "status": g["status"],
                     "n_out": res.get("n_features_out"), "wall_s": g["wall_s"], "peak_mb": g["peak_rss_mb"],
                     "error": ((g["error"] or "").strip().splitlines() or [""])[-1][:120]})
    fits = []
    for f in fit_records:
        g = gens[f["unit"]["gen_hash"]]
        ds = g["unit"]["dataset"]
        m = METRIC[tasks[ds]]
        fits.append({"dataset": ds, "arm": f["unit"]["arm"], "model": f["unit"]["model"], "status": f["status"],
                     "metric": m, "value": (f["result"] or {}).get("test", {}).get(m),
                     "n_features": (f["result"] or {}).get("n_features"), "wall_s": f["wall_s"],
                     "peak_mb": f["peak_rss_mb"]})
    gen, fit = pd.DataFrame(rows), pd.DataFrame(fits)
    base = fit[fit.arm == "base"].set_index(["dataset", "model"]).value
    fit["base"] = [base.get((d, m)) for d, m in zip(fit.dataset, fit.model)]
    wide = fit.pivot_table(index=["dataset", "arm"], columns="model", values="value", aggfunc="first")
    md = ["# exp04 smoke report", "", "## Generator units", "",
          gen.sort_values(["arm", "dataset"]).to_markdown(index=False), "",
          "## Fit units (test metric: AUC binary, log loss multiclass, RMSE regression)", "",
          fit.sort_values(["dataset", "arm", "model"]).to_markdown(index=False), "",
          "## Test metric by dataset and arm", "", wide.to_markdown(), ""]
    (RESULTS / "tidy" / "exp04_smoke.md").write_text("\n".join(md))
    print(gen.groupby(["arm", "status"]).size().to_string(), "\n", fit.groupby(["arm", "status"]).size().to_string())


if __name__ == "__main__":
    main()
