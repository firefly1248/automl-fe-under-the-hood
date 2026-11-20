"""Figures for the article (plan section 9): matplotlib, SVG + PNG, one fixed colour per arm.

`python -m afe.analysis.figures` writes results/figures/exp04/*.{svg,png} and exp05/*. Each figure used
in the article gets a claims entry with the hash of its rendered data table (analysis/claims.py).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afe.paths import RESULTS

ARMS = ["base", "ag_default", "ag_tabprep", "ag_tabprep_keepcat", "lama_adv", "openfe", "autofeat",
        "mljar_golden", "manual_baseline", "skrub_tv"]
COLOR = dict(zip(ARMS, plt.get_cmap("tab10").colors))
# What the article images call an arm: the only mapping, used by every picture in results/figures/article (CSVs, claims and
# tidy tables keep the arm names).
ARM_LABEL = {"ag_default": "AutoGluon default", "ag_tabprep": "AutoGluon TabPrep",
             "ag_tabprep_keepcat": "AutoGluon TabPrep (raw categoricals kept)", "lama_adv": "LightAutoML pipeline",
             "openfe": "OpenFE", "autofeat": "autofeat", "mljar_golden": "MLJAR golden features",
             "skrub_tv": "skrub TableVectorizer", "manual_baseline": "manual baseline", "base_numcast": "numbers where possible",
             "base_codes": "integer codes"}


def label(arm: str) -> str:
    return ARM_LABEL.get(arm, arm)
EFFECT_LABEL = {"auc_diff": "AUC difference vs base", "logloss_rel": "relative log loss improvement vs base",
                "rmse_rel": "relative RMSE improvement vs base"}
PRIMARY_EFFECT = {"binary": "auc_diff", "multiclass": "logloss_rel", "regression": "rmse_rel"}
LOW_COVERAGE = 0.8  # arms with a smaller share of ok units get a hollow marker and the coverage written out
plt.rcParams.update({"svg.hashsalt": "afe", "figure.dpi": 100, "savefig.bbox": "tight", "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def save(fig, sub: str, name: str, data: pd.DataFrame = None) -> None:
    """Write the figure and, for claims, the exact data table it rendered (plan section 9)."""
    out = RESULTS / "figures" / sub
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.svg", metadata={"Date": None})  # with svg.hashsalt: same data, same bytes
    fig.savefig(out / f"{name}.png", dpi=150)
    plt.close(fig)
    if data is not None:
        data.to_csv(out / f"{name}.csv", index=False)
    print(f"{sub}/{name}")


def median_effect(summ: pd.DataFrame, rule: str = "base_fallback"):
    """Median effect with bootstrap CI, per task type and model, arms on the y axis."""
    s = summ[(summ.rule == rule) & summ.in_family & (summ.effect != "logloss_diff")]
    s = s[[e == PRIMARY_EFFECT[t] for e, t in zip(s.effect, s.task)]]
    tasks, models = ["binary", "multiclass", "regression"], ["lgbm", "catboost"]
    fig, axes = plt.subplots(len(models), len(tasks), figsize=(12, 7), sharey=True)
    arms = [a for a in ARMS if a != "base"]
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            g = s[(s.task == task) & (s.model == model)].set_index("arm").reindex(arms)
            y = np.arange(len(arms))
            for yi, arm in zip(y, arms):
                r = g.loc[arm]
                ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=COLOR[arm], lw=2)
            colors, low = np.array([COLOR[a] for a in arms]), (g.coverage < LOW_COVERAGE).to_numpy()
            ax.scatter(g.median_effect[~low], y[~low], c=colors[~low], zorder=3)
            ax.scatter(g.median_effect[low], y[low], facecolors="none", edgecolors=colors[low], linewidths=1.8,
                       zorder=3)
            for yi, r in zip(y[low], g[low].itertuples()):
                ax.annotate(f"{r.coverage:.0%} ok", (r.ci_hi, yi), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8)
            ax.axvline(0, color="grey", lw=1)
            ax.set_yticks(y, arms)
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(f"{task} ({int(g.n_datasets.max())} datasets)")
            if i == len(models) - 1:
                ax.set_xlabel(EFFECT_LABEL[PRIMARY_EFFECT[task]].replace(" vs base", ""))
            if j == 0:
                ax.set_ylabel(model)
    fig.suptitle(f"Median effect vs base over primary datasets, 95% bootstrap CI ({rule}); "
                 f"hollow marker: under {LOW_COVERAGE:.0%} of units ok")
    cols = ["task", "model", "arm", "n_datasets", "median_effect", "ci_lo", "ci_hi", "coverage"]
    save(fig, "exp04", f"median_effect_{rule}", s[cols])


def win_tie_loss(eff: pd.DataFrame, rule: str = "base_fallback"):
    """Stacked descriptive win/tie/loss counts per arm, primary effect of each task type."""
    e = eff[(eff.rule == rule)]
    e = e[[r.effect == PRIMARY_EFFECT[r.task] for r in e.itertuples()]]
    tasks, models = ["binary", "multiclass", "regression"], ["lgbm", "catboost"]
    arms = [a for a in ARMS if a != "base"]
    fig, axes = plt.subplots(len(models), len(tasks), figsize=(12, 7), sharey=True)
    for i, model in enumerate(models):
        for j, task in enumerate(tasks):
            ax = axes[i, j]
            g = e[(e.task == task) & (e.model == model)]
            counts = g.groupby("arm").verdict.value_counts().unstack(fill_value=0).reindex(arms).fillna(0)
            cov = g.groupby("arm").coverage.mean().reindex(arms)
            for v in ("win", "tie", "loss"):
                if v not in counts:
                    counts[v] = 0
            y = np.arange(len(arms))
            left = np.zeros(len(arms))
            for v, c in (("win", "#2a9d3f"), ("tie", "#d0d0d0"), ("loss", "#d62728")):
                ax.barh(y, counts[v], left=left, color=c, label=v)
                left += counts[v].to_numpy()
            for yi, c, x in zip(y, cov, left):
                if c < LOW_COVERAGE:
                    ax.annotate(f"{c:.0%} ok", (x, yi), xytext=(4, 0), textcoords="offset points", va="center",
                                fontsize=8)
            ax.set_yticks(y, arms)
            ax.invert_yaxis()
            if i == 0:
                ax.set_title(task)
            if j == 0:
                ax.set_ylabel(model)
            ax.set_xlabel("datasets")
    axes[0, 0].legend(loc="lower right", fontsize=9)
    fig.suptitle(f"Descriptive win / tie / loss vs base, Nadeau-Bengio t-test at 0.05 ({rule}); "
                 f"label: arms with under {LOW_COVERAGE:.0%} of units ok")
    wtl = e.groupby(["task","model","arm"], observed=True).verdict.value_counts().unstack(fill_value=0).reset_index()
    wtl = wtl.merge(e.groupby(["task", "model", "arm"], observed=True).coverage.mean().reset_index())
    save(fig, "exp04", f"win_tie_loss_{rule}", wtl)


def _scatter(df: pd.DataFrame, eff: pd.DataFrame, xcol: str, xlabel: str, name: str):
    """Per-dataset effect (primary, complete cases) vs a per-dataset median of `xcol`, log x."""
    e = eff[eff.rule == "complete_cases"]
    e = e[[r.effect == PRIMARY_EFFECT[r.task] for r in e.itertuples()]]
    x = df[df.status == "ok"].groupby(["dataset", "arm", "model"], observed=True)[xcol].median()
    e = e.join(x.rename("x"), on=["dataset", "arm", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, ["lgbm", "catboost"]):
        g = e[e.model == model]
        for arm in [a for a in ARMS if a != "base"]:
            ga = g[g.arm == arm]
            ax.scatter(ga.x.clip(lower=0.5 if xcol == "n_new_features" else None), ga.value, s=18,
                       color=COLOR[arm], label=arm, alpha=0.75)
        ax.set_xscale("log")
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(model)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("effect vs base (primary metric)")
    axes[1].legend(fontsize=8, markerscale=1.2)
    fig.suptitle(f"Per-dataset effect vs {xlabel} (complete cases)")
    save(fig, "exp04", name, e[["dataset","task","arm","model","effect","value","x"]])


def exp05_permuted():
    p = pd.read_csv(RESULTS / "tidy" / "exp05_permuted.csv")
    tasks = ["binary", "multiclass", "regression"]
    stat = {"binary": "AUC - 0.5", "multiclass": "prior entropy - log loss", "regression": "R2"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
    arms = [a for a in ARMS if a in set(p.arm)]
    for ax, task in zip(axes, tasks):
        g = p[p.task == task]
        for k, model in enumerate(["lgbm", "catboost"]):
            gm = g[g.model == model].set_index("arm").reindex(arms)
            y = np.arange(len(arms)) + (k - 0.5) * 0.3
            ax.errorbar(gm["mean"], y, xerr=[gm["mean"] - gm.ci_lo, gm.ci_hi - gm["mean"]], fmt="o",
                        ms=4, color="#1f77b4" if model == "lgbm" else "#ff7f0e", label=model,
                        elinewidth=1.5, capsize=2)
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks(np.arange(len(arms)), arms)
        ax.invert_yaxis()
        ax.set_title(task)
        ax.set_xlabel(stat[task])
    axes[0].legend(fontsize=9)
    fig.suptitle("Permuted labels: test metric above chance would signal leakage (95% CI over units)")
    save(fig, "exp05", "permuted", p)


def exp05_planted():
    p = pd.read_csv(RESULTS / "tidy" / "exp05_planted.csv")
    arms = [a for a in ARMS if a in set(p.arm)]
    kinds = sorted(p.kind.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (col, title) in zip(axes, [("reach", "reaches oracle R2 (within 1% relative)"),
                                       ("found", "true feature among the created ones (|corr| >= 0.95)")]):
        share = p.groupby(["arm", "kind"], observed=True)[col].mean().unstack().reindex(arms)[kinds]
        im = ax.imshow(share.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(kinds)), kinds, rotation=20, ha="right")
        ax.set_yticks(range(len(arms)), arms)
        ax.set_title(title, fontsize=10)
        for i in range(len(arms)):
            for j in range(len(kinds)):
                v = share.to_numpy()[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of units")
    fig.suptitle("Planted signals (4 kinds x 3 seeds x 5 folds, both models)")
    tab = p.groupby(["arm","kind"], observed=True)[["reach","found"]].mean().reset_index()
    save(fig, "exp05", "planted", tab)


def main() -> None:
    df = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    eff = pd.read_csv(RESULTS / "tidy" / "exp04_effects.csv")
    summ = pd.read_csv(RESULTS / "tidy" / "exp04_summary.csv")
    for rule in ("base_fallback", "complete_cases"):
        median_effect(summ, rule)
        win_tie_loss(eff, rule)
    _scatter(df, eff, "gen_fit_s", "generator fit time, s", "effect_vs_gen_time")
    _scatter(df, eff, "n_new_features", "n new features", "effect_vs_n_features")
    exp05_permuted()
    exp05_planted()


if __name__ == "__main__":
    main()
