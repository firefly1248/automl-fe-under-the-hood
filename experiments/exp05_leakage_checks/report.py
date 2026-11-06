"""exp05 checks 1-4 (plan section 8) from the run records. Core env:

    uv run --project envs/core python experiments/exp05_leakage_checks/report.py

Writes results/tidy/exp05_permuted.csv, exp05_te_delivery.csv, exp05_transductive.csv, exp05_planted.csv and
exp05_report.md. Check 5 is `openfe_ops.py` (openfe env); the planted oracle is `planted_oracle.py`.
"""

from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from afe.data import splits as S
from afe.data.loaders import load
from afe.paths import CACHE, RESULTS, ROOT
from afe.run import cli, store
from afe.run.worker import _data

HERE = Path(__file__).parent
EXP04 = ROOT / "experiments" / "exp04_open_benchmark" / "config.yaml"
TIDY = RESULTS / "tidy"
PRIMARY = {"binary": "auc", "multiclass": "logloss", "regression": "r2"}
TASK = pd.read_csv(RESULTS / "datasets.csv").set_index("key").task.to_dict()
BOOT = 2000


def chains(config: Path, stage: str):
    return cli.records(yaml.safe_load(config.read_text()), stage)


def fit_table(config: Path, stage: str) -> pd.DataFrame:
    rows = []
    for g, gr, fits in chains(config, stage):
        for f, fr in fits:
            status = fr["status"] if fr else (gr["status"] if gr and gr["status"] != "ok" else "missing")
            rows.append({"dataset": g["dataset"], "task": TASK[g["dataset"]], "fold": g["fold"], "arm": f["arm"],
                         "model": f["model"], "status": status,
                         **(fr["result"]["test"] if fr and fr["status"] == "ok" else {})})
    return pd.DataFrame(rows)


def boot_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float]:
    means = np.random.default_rng(seed).choice(x, (BOOT, len(x))).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@cache
def prior_entropy(dataset: str, fold: int) -> float:
    """Entropy of the class shares of the (permuted) fit rows."""
    _, y, _, idx = _data({"dataset": dataset, "repeat": 0, "fold": fold}, permute_y=True)
    p = y.iloc[idx["train"]].value_counts(normalize=True).to_numpy()
    return float(-(p * np.log(p)).sum())


def check1() -> tuple[pd.DataFrame, dict]:
    """Permuted labels: chance-level test metric per arm, model and task type."""
    d = fit_table(HERE / "permuted.yaml", "run")
    ok = d[d.status == "ok"].copy()
    ok["stat"] = [r.auc - 0.5 if r.task == "binary" else r.r2 if r.task == "regression"
                  else prior_entropy(r.dataset, r.fold) - r.logloss for r in ok.itertuples()]
    rows = []
    for (arm, model, task), s in ok.groupby(["arm", "model", "task"]):
        lo, hi = boot_ci(s.stat.to_numpy())
        rows.append({"arm": arm, "model": model, "task": task, "n_units": len(s), "mean": s.stat.mean(),
                     "ci_lo": lo, "ci_hi": hi, "pass": lo <= 0})
    out = pd.DataFrame(rows)
    out.to_csv(TIDY / "exp05_permuted.csv", index=False)
    return out, d.status.value_counts().to_dict()


def _abs_corr(x: pd.Series, y) -> float:
    return abs(float(np.corrcoef(x, y)[0, 1])) if x.std() > 0 else 0.0


def _score(col: pd.Series, y: pd.Series, task: str) -> float:
    """Direction-free association of one column with y: AUC (max over classes, one vs rest) or |corr|."""
    keep = col.notna()
    x, y = col[keep].astype(float), y[keep]
    if task == "regression":
        return _abs_corr(x, y)
    aucs = [roc_auc_score(y == c, x) for c in (sorted(y.unique())[1:] if task == "binary" else sorted(y.unique()))
            if 0 < (y == c).mean() < 1]
    return max(max(a, 1 - a) for a in aucs) if aucs else 0.5


def check2() -> pd.DataFrame:
    """TE delivery: association of each TE column with y on fit rows (as delivered) vs test rows. Raw numeric
    columns of the same outputs give the gap from sampling noise alone."""
    rows = []
    for g, gr, _ in chains(EXP04, "pilot"):
        if not gr or gr["status"] != "ok" or "target_encoded" not in gr["result"]["describe"]:
            continue
        X, y, meta, idx = _data(g)
        task = meta["task"]
        limit = 0.1 if task == "regression" else 0.05
        out = {s: pd.read_parquet(store.cache_dir(g["exp"], gr["hash"]) / f"{s}.parquet") for s in ("train", "test")}
        te = [c for c in gr["result"]["describe"]["target_encoded"] if c in out["train"]]
        raw = [c for c in X.columns if c in out["train"] and pd.api.types.is_numeric_dtype(out["train"][c])]
        for kind, c in [("te", c) for c in te] + [("raw", c) for c in raw]:
            fit_s = _score(out["train"][c].set_axis(y.index[idx["train"]]), y.iloc[idx["train"]], task)
            test_s = _score(out["test"][c].set_axis(y.index[idx["test"]]), y.iloc[idx["test"]], task)
            gap = fit_s - test_s
            rows.append({"arm": g["arm"], "dataset": g["dataset"], "task": task, "fold": g["fold"], "kind": kind, "column": c,
                         "fit": fit_s, "test": test_s, "gap": gap, "flag": gap > limit})
    out = pd.DataFrame(rows)
    out.to_csv(TIDY / "exp05_te_delivery.csv", index=False)
    return out


def check3() -> pd.DataFrame:
    """openfe_transductive vs openfe (exp04 pilot), same dataset, fold and model."""
    t = fit_table(HERE / "transductive.yaml", "run")
    i = fit_table(EXP04, "pilot")
    i = i[i.arm == "openfe"]
    rows = []
    for r in t.itertuples():
        m = PRIMARY[r.task]
        ind = i[(i.dataset == r.dataset) & (i.fold == r.fold) & (i.model == r.model)].iloc[0]
        rows.append({"dataset": r.dataset, "task": r.task, "fold": r.fold, "model": r.model, "metric": m,
                     "status_transductive": r.status, "status_inductive": ind.status,
                     "transductive": getattr(r, m, np.nan), "inductive": ind.get(m, np.nan)})
    out = pd.DataFrame(rows)
    out["diff"] = out.transductive - out.inductive
    out.to_csv(TIDY / "exp05_transductive.csv", index=False)
    return out


def _relation(col: pd.Series, oracle: np.ndarray) -> float:
    """|corr| for numeric columns; correlation ratio (eta) of the oracle on a categorical column."""
    o = pd.Series(oracle, index=col.index)
    if isinstance(col.dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(col):
        means = o.groupby(col.astype(object), dropna=True).transform("mean")
        keep = means.notna()
        return float(np.sqrt(((means[keep] - o[keep].mean()) ** 2).sum() / ((o[keep] - o[keep].mean()) ** 2).sum()))
    keep = col.notna() & np.isfinite(col.astype(float))
    return _abs_corr(col[keep].astype(float), o[keep])


def check4() -> pd.DataFrame:
    """Planted signals: metric vs the oracle (within 1% relative) and the true feature among the created ones."""
    oracle = pd.read_csv(TIDY / "exp05_oracle.csv").set_index(["dataset", "fold", "model"]).r2
    rows = []
    for g, gr, fits in chains(HERE / "planted.yaml", "run"):
        found = {}
        if gr and gr["status"] == "ok":
            X, _, _ = load(g["dataset"])
            test_rows = S.rows(S.load(g["dataset"]), g["repeat"], g["fold"])[2]
            o = pd.read_parquet(CACHE / "data" / f"{g['dataset']}.oracle.parquet").oracle.to_numpy()[test_rows]
            out = pd.read_parquet(store.cache_dir(g["exp"], gr["hash"]) / "test.parquet")
            new = gr["result"]["describe"].get("new_columns") or [c for c in out.columns if c not in X.columns]
            best = max(((_relation(out[c], o), c) for c in new), default=(0.0, None))
            found = {"found": best[0] >= 0.95, "best_relation": best[0], "best_column": best[1], "n_created": len(new)}
        for f, fr in fits:  # keepcat shares the ag_tabprep generator; its created columns are the same list
            r2 = fr["result"]["test"]["r2"] if fr and fr["status"] == "ok" else np.nan
            orc = oracle[g["dataset"], g["fold"], f["model"]]
            rows.append({"dataset": g["dataset"], "kind": g["dataset"].split("_s")[0].removeprefix("planted_"),
                         "fold": g["fold"], "arm": f["arm"], "model": f["model"], "r2": r2, "oracle_r2": orc,
                         "reach": r2 >= 0.99 * orc, **found})
    out = pd.DataFrame(rows)
    out.to_csv(TIDY / "exp05_planted.csv", index=False)
    return out


def main() -> None:
    c1, c1_status = check1()
    c2, c3, c4 = check2(), check3(), check4()
    ops = pd.read_csv(TIDY / "exp05_openfe_ops.csv")
    c2s = c2.groupby(["arm", "task", "kind"]).agg(columns=("flag", "size"), flagged_share=("flag", "mean"),
                                                  max_gap=("gap", "max"), median_gap=("gap", "median")).reset_index()
    c4s = (c4.groupby(["kind", "arm", "model"]).agg(units=("reach", "size"), reach=("reach", "mean"),
                                                   found=("found", "mean")).reset_index())
    md = [
        "# exp05 leakage checks", "",
        "Written by `experiments/exp05_leakage_checks/report.py`.", "",
        "## 1. Permuted labels", "",
        f"Unit statuses: {c1_status}. Statistic per unit: binary AUC - 0.5; multiclass prior entropy - log loss; "
        f"regression R2. Pass: the 95% bootstrap CI over units contains 0 or lies below it.", "",
        c1.round(4).to_markdown(index=False), "",
        "## 2. Target-encoding delivery (exp04 pilot)", "",
        "Gap = association with y on fit rows (as delivered) - on test rows; flag above 0.05 (AUC) or 0.1 (|corr|). "
        "kind raw: the arm's untouched numeric input columns, the gap from sampling noise alone.", "",
        c2s.round(4).to_markdown(index=False), "",
        "## 3. openfe_transductive - openfe (exp04 pilot, repeat 0)", "",
        c3.round(4).to_markdown(index=False), "",
        "## 4. Planted signals", "",
        "Share of units that reach the oracle R2 within 1% relative, and share of generator units where a created "
        "column has |corr| (numeric) or correlation ratio (categorical) >= 0.95 with the true feature on test rows.", "",
        c4s.round(3).to_markdown(index=False), "",
        "## 5. OpenFE operator equivalence (fit rows)", "",
        ops.to_markdown(index=False), "",
    ]
    (TIDY / "exp05_report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
