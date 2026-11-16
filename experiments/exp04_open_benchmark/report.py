"""Render results/tidy/exp04_report.md from the tidy tables (no number typed by hand).

Run in the core env: `uv run --frozen --project envs/core python experiments/exp04_open_benchmark/report.py`.
"""

import pandas as pd

from afe.paths import RESULTS

PRIMARY_EFFECT = {"binary": "auc_diff", "multiclass": "logloss_rel", "regression": "rmse_rel"}


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    summ = pd.read_csv(RESULTS / "tidy" / "exp04_summary.csv")
    tidy = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    lines = ["# exp04 report", "", "Written by `experiments/exp04_open_benchmark/report.py` from",
             "`results/tidy/exp04.parquet`, `exp04_effects.csv` and `exp04_summary.csv`.", ""]

    lines += ["## Coverage (ok / planned fit units, both repeats)", ""]
    cov = (tidy.assign(ok=tidy.status == "ok").groupby(["arm", "model"], observed=True)
           .agg(planned=("status", "size"), ok=("ok", "sum")))
    cov["coverage"] = (cov.ok / cov.planned).round(3)
    lines += [fmt(cov.reset_index()), ""]

    for rule in ("base_fallback", "complete_cases"):
        lines += [f"## Median effect vs `base` over primary datasets ({rule})", "",
                  "Positive favours the arm. 95% bootstrap CI over datasets; Wilcoxon signed-rank with",
                  "Holm within (task, model, effect, rule).", ""]
        s = summ[(summ.rule == rule) & summ.in_family]
        s = s[[r.effect == PRIMARY_EFFECT[r.task] for r in s.itertuples()]]
        cols = ["task", "effect", "model", "arm", "n_datasets", "median_effect", "ci_lo", "ci_hi",
                "wilcoxon_p_holm", "share_win", "share_loss", "coverage"]
        lines += [fmt(s[cols].sort_values(["task", "model", "arm"])), ""]

    lines += ["## Binary secondary effect: relative log loss improvement (base_fallback)", ""]
    s = summ[(summ.rule == "base_fallback") & (summ.task == "binary") & (summ.effect == "logloss_rel")]
    lines += [fmt(s[["model", "arm", "n_datasets", "median_effect", "ci_lo", "ci_hi", "share_win",
                     "share_loss"]].sort_values(["model", "arm"])), ""]

    lines += ["## Breakdown: datasets with a categorical of >= 10 levels (base_fallback, descriptive)", ""]
    eff = pd.read_csv(RESULTS / "tidy" / "exp04_effects.csv")
    hic = tidy[["dataset", "hicard"]].drop_duplicates().set_index("dataset").hicard
    e = eff[(eff.rule == "base_fallback")].join(hic, on="dataset")
    e = e[[r.effect == PRIMARY_EFFECT[r.task] for r in e.itertuples()]]
    g = (e.groupby(["hicard", "model", "arm"], observed=True).value.median().rename("median_effect")
         .reset_index())
    lines += [fmt(g.pivot(index=["model", "arm"], columns="hicard", values="median_effect")
                  .rename(columns={False: "no_hicard", True: "hicard"}).reset_index()), ""]

    (RESULTS / "tidy" / "exp04_report.md").write_text("\n".join(lines))
    print("results/tidy/exp04_report.md")


if __name__ == "__main__":
    main()
