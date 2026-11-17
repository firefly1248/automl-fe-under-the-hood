"""Figures and tables for the Medium article (plan section 9): PNG only, at least 1400 px wide, text sized to stay
readable when Medium shows the image in a 700 px column (11 in x 140 dpi = 1540 px, fonts 13 to 15 pt).

`python -m afe.analysis.article` writes results/figures/article/<name>.png and <name>.csv (the exact rendered
data, hashed by the figure claims). The alt text of every image is a hand-written `<name>.alt.txt` next to it;
the run ends with an error listing any that is missing. `preview.png` is the 1500 x 680 px image with white margins that stands in the
text of section 4.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from afe.analysis import exp02, exp06
from afe.analysis.figures import ARMS, COLOR, EFFECT_LABEL, LOW_COVERAGE, PRIMARY_EFFECT
from afe.paths import RESULTS

OUT = RESULTS / "figures" / "article"
TASKS = ["binary", "multiclass", "regression"]
DPI = 140
HP_DATASETS = ["elevators", "kr-vs-kp", "socmob"]
HP_ARMS = ["A", "A_et0", "A_et0_nosamp", "A_et0_nocap", "A_et0_noreg", "A_et0_nocat", "A_def"]
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "savefig.bbox": "tight", "figure.facecolor": "white"})


def save(fig, name: str, data: pd.DataFrame, **kw) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=kw.pop("dpi", DPI), **kw)
    plt.close(fig)
    data.to_csv(OUT / f"{name}.csv", index=False)
    width = plt.imread(OUT / f"{name}.png").shape[1]
    assert width >= 1400, f"{name}: {width} px wide"
    print(f"article/{name} ({width} px)")


def _median_panels(summ: pd.DataFrame, model: str, rule: str, fig, axes) -> pd.DataFrame:
    s = summ[(summ.rule == rule) & summ.in_family & (summ.model == model)]
    s = s[[e == PRIMARY_EFFECT[t] for e, t in zip(s.effect, s.task)]]
    arms = [a for a in ARMS if a != "base"]
    y = np.arange(len(arms))
    for ax, task in zip(axes, TASKS):
        g = s[s.task == task].set_index("arm").reindex(arms)
        colors, low = np.array([COLOR[a] for a in arms]), (g.coverage < LOW_COVERAGE).to_numpy()
        ax.hlines(y, g.ci_lo, g.ci_hi, colors=colors, lw=2.5)
        ax.scatter(g.median_effect[~low], y[~low], c=colors[~low], s=50, zorder=3)
        ax.scatter(g.median_effect[low], y[low], facecolors="white", edgecolors=colors[low], linewidths=2, s=50,
                   zorder=3)
        for yi, r in zip(y[low], g[low].itertuples()):
            ax.annotate(f"{r.coverage:.0%} ok", (r.ci_hi, yi), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=10)
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks(y, arms)
        ax.invert_yaxis()
        ax.set_title(f"{task}, {int(g.n_datasets.max())} datasets")
        ax.set_xlabel(EFFECT_LABEL[PRIMARY_EFFECT[task]].replace(" vs base", "").replace("relative ", "rel. "))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    return s[["task", "model", "arm", "n_datasets", "median_effect", "ci_lo", "ci_hi", "coverage"]]


def median_effect(summ: pd.DataFrame, model: str, rule: str = "base_fallback") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.4), sharey=True)
    data = _median_panels(summ, model, rule, fig, axes)
    fig.tight_layout()
    save(fig, f"median_effect_{model}", data)


def preview(summ: pd.DataFrame) -> None:
    """1500 x 680 px, white margins, no tight cropping: the figure that stands in the text of section 4."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8), sharey=True)
    data = _median_panels(summ, "lgbm", "base_fallback", fig, axes)
    fig.subplots_adjust(left=0.15, right=0.97, top=0.9, bottom=0.17, wspace=0.08)
    with plt.rc_context({"savefig.bbox": "standard"}):  # no cropping: the size is exactly 1500 x 680
        save(fig, "preview", data, dpi=100)
    assert plt.imread(OUT / "preview.png").shape[:2] == (680, 1500)


def win_tie_loss(eff: pd.DataFrame, model: str, rule: str = "base_fallback") -> None:
    e = eff[(eff.rule == rule) & (eff.model == model)]
    e = e[[r.effect == PRIMARY_EFFECT[r.task] for r in e.itertuples()]]
    arms = [a for a in ARMS if a != "base"]
    y = np.arange(len(arms))
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.4), sharey=True)
    rows = []
    for ax, task in zip(axes, TASKS):
        g = e[e.task == task]
        counts = g.groupby("arm").verdict.value_counts().unstack(fill_value=0).reindex(arms).fillna(0)
        cov = g.groupby("arm").coverage.mean().reindex(arms)
        left = np.zeros(len(arms))
        for v, c in (("win", "#2a9d3f"), ("tie", "#d0d0d0"), ("loss", "#d62728")):
            n = counts[v] if v in counts else pd.Series(0, index=arms)
            ax.barh(y, n, left=left, color=c, label=v)
            left += n.to_numpy()
        for yi, c, x in zip(y, cov, left):
            if c < LOW_COVERAGE:
                ax.annotate(f"{c:.0%} ok", (x, yi), xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=10)
        ax.set_yticks(y, arms)
        ax.invert_yaxis()
        ax.set_title(task)
        ax.set_xlabel("datasets")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        rows.append(counts.assign(task=task, coverage=cov).reset_index())
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=11)
    fig.tight_layout()
    save(fig, f"win_tie_loss_{model}", pd.concat(rows))


def effect_scatter(df: pd.DataFrame, eff: pd.DataFrame, xcol: str, xlabel: str, name: str) -> None:
    e = eff[eff.rule == "complete_cases"]
    e = e[[r.effect == PRIMARY_EFFECT[r.task] for r in e.itertuples()]]
    x = df[df.status == "ok"].groupby(["dataset", "arm", "model"], observed=True)[xcol].median()
    e = e.join(x.rename("x"), on=["dataset", "arm", "model"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True)
    for ax, model in zip(axes, ["lgbm", "catboost"]):
        g = e[e.model == model]
        for arm in [a for a in ARMS if a != "base"]:
            ga = g[g.arm == arm]
            ax.scatter(ga.x.clip(lower=0.5 if xcol == "n_new_features" else None), ga.value, s=22, color=COLOR[arm],
                       label=arm, alpha=0.75)
        ax.set_xscale("log")
        ax.axhline(0, color="grey", lw=1)
        ax.set_title(model)
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel("effect vs base")
    axes[1].legend(fontsize=9, markerscale=1.2, loc="lower left")
    fig.tight_layout()
    save(fig, name, e[["dataset", "task", "arm", "model", "effect", "value", "x"]])


def exp05_permuted() -> None:
    p = pd.read_csv(RESULTS / "tidy" / "exp05_permuted.csv")
    stat = {"binary": "AUC - 0.5", "multiclass": "prior entropy - log loss", "regression": "R2"}
    arms = [a for a in ARMS if a in set(p.arm)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 5.4), sharey=True)
    for ax, task in zip(axes, TASKS):
        g = p[p.task == task]
        for k, model in enumerate(["lgbm", "catboost"]):
            gm = g[g.model == model].set_index("arm").reindex(arms)
            y = np.arange(len(arms)) + (k - 0.5) * 0.3
            ax.errorbar(gm["mean"], y, xerr=[gm["mean"] - gm.ci_lo, gm.ci_hi - gm["mean"]], fmt="o", ms=5,
                        color="#1f77b4" if model == "lgbm" else "#ff7f0e", label=model, elinewidth=2, capsize=2)
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks(np.arange(len(arms)), arms)
        ax.invert_yaxis()
        ax.set_title(task)
        ax.set_xlabel(stat[task])
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=11)
    fig.tight_layout()
    save(fig, "permuted_labels", p)


def exp05_planted() -> None:
    p = pd.read_csv(RESULTS / "tidy" / "exp05_planted.csv")
    arms, kinds = [a for a in ARMS if a in set(p.arm)], sorted(p.kind.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (col, title) in zip(axes, [("reach", "reaches the oracle R2"), ("found", "true feature created")]):
        share = p.groupby(["arm", "kind"], observed=True)[col].mean().unstack().reindex(arms)[kinds]
        im = ax.imshow(share.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(kinds)), kinds, rotation=25, ha="right")
        ax.set_yticks(range(len(arms)), arms)
        ax.set_title(title)
        ax.grid(False)
        for i in range(len(arms)):
            for j in range(len(kinds)):
                ax.text(j, i, f"{share.to_numpy()[i, j]:.1f}", ha="center", va="center", fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="share of units")
    save(fig, "planted_signals", p.groupby(["arm", "kind"], observed=True)[["reach", "found"]].mean().reset_index())


def contrasts(per: pd.DataFrame, pairs, name: str, ncols: int) -> None:
    names = [c for c in pairs if c in set(per.contrast)]
    nrows = -(-len(names) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4.6 * nrows), sharey=True, squeeze=False)
    order = per.drop_duplicates("dataset").sort_values(["task", "dataset"])
    y = np.arange(len(order))
    for ax, c in zip(axes.flat, names):
        g = per[per.contrast == c].set_index("dataset").reindex(order.dataset)
        ax.hlines(y, g.ci_lo, g.ci_hi, color="0.6", lw=2)
        ax.scatter(g.effect, y, s=36, zorder=3, c=np.select([g.ci_lo > 0, g.ci_hi < 0], ["#2a9d3f", "#d62728"], "0.5"))  # as the tables: fold CI clear of zero
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(c)
        ax.set_xlabel("relative improvement")
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_yticks(y, [f"{d} ({t[:3]})" for d, t in zip(order.dataset, order.task)])
    axes[0, 0].invert_yaxis()  # the y axis is shared: invert once
    fig.tight_layout()
    save(fig, name, per[per.contrast.isin(names)])


def table(df: pd.DataFrame, name: str, cell, shade, col_labels=None, col_width=1.3, first_width=2.6,
          group_gap: str = None) -> None:
    """Rendered table: `cell(value)` gives the text, `shade(value)` a colour or None; first column is the row label."""
    labels = list(col_labels or df.columns[1:])
    n, m = len(df), len(labels)
    fig, ax = plt.subplots(figsize=(max(11, first_width + col_width * m), 0.27 * (n + 1) + 0.5))
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99)
    ax.axis("off")
    widths = [first_width / (first_width + col_width * m)] + [col_width / (first_width + col_width * m)] * m
    t = ax.table(cellText=[[df.iloc[i, 0]] + [cell(v) for v in df.iloc[i, 1:]] for i in range(n)],
                 colLabels=[df.columns[0]] + labels, colWidths=widths, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(13)
    t.scale(1, 1.6)
    for (r, c), cl in t.get_celld().items():
        cl.set_edgecolor("0.8")
        if r == 0:
            cl.set_facecolor("#e8e8e8")
            cl.set_text_props(weight="bold")
        elif c > 0 and (col := shade(df.iloc[r - 1, c])):
            cl.set_facecolor(col)
        if c == 0:
            cl._loc = "left"
            cl.set_text_props(ha="left")
    shown = df.astype(object)  # the CSV is the text of the cells, what the figure claim hashes
    shown.iloc[:, 1:] = [[cell(v) for v in row] for row in df.iloc[:, 1:].to_numpy()]
    with plt.rc_context({"savefig.bbox": "standard"}):  # keep the full 11 in width, no cropping
        save(fig, name, shown)


def contrast_table(per: pd.DataFrame, summ: pd.DataFrame, pairs, name: str) -> None:
    """Per-dataset relative improvement in % (green or red when the fold CI excludes zero), then per task type the
    datasets better by the sign of the mean, clearly better and clearly worse (CI excludes zero), of M datasets.
    The median is not in the table."""
    names = [c for c in pairs if c in set(per.contrast)]
    rows = []
    for (task, dataset), g in per.groupby(["task", "dataset"]):
        g = g.set_index("contrast")
        rows.append([f"{dataset} ({task[:3]})"] + [(g.effect.get(c, np.nan), g.ci_lo.get(c, np.nan),
                                                    g.ci_hi.get(c, np.nan)) for c in names])
    for task in TASKS:
        s = summ[summ.task == task].set_index("contrast")
        if len(s):
            for label, col in (("better (sign)", "n_better"), ("clearly better", "n_clear_better"),
                               ("clearly worse", "n_clear_worse")):
                rows.append([f"{task}: {label}"] + [(None, int(s[col].get(c, 0)), int(s.n_datasets.get(c, 0)))
                                                  for c in names])
    df = pd.DataFrame(rows, columns=["dataset"] + names)

    def cell(v):
        if v[0] is None:
            return f"{v[1]} of {v[2]}"
        if np.isnan(v[0]):
            return ""
        r = round(v[0] * 100, 1) + 0.0  # + 0.0: no "-0.0"
        return "0.0%" if r == 0 else f"{r:+.1f}%"

    def shade(v):
        if v[0] is None:
            return "#f3f3f3"
        return "#cfe9d3" if v[1] > 0 else "#f3cccc" if v[2] < 0 else None

    table(df, name, cell, shade, col_width=min(1.5, (11 - 3.2) / len(names)), first_width=3.2)


def coverage_table(summ: pd.DataFrame) -> None:
    s = summ[(summ.rule == "base_fallback") & summ.in_family]
    s = s[[e == PRIMARY_EFFECT[t] for e, t in zip(s.effect, s.task)]]
    s = s.assign(col=s.task.str[:3] + " " + s.model).pivot_table(index="arm", columns="col", values="coverage")
    s = s.reindex([a for a in ARMS if a != "base"])[[f"{t[:3]} {m}" for t in TASKS for m in ("lgbm", "catboost")]]
    df = s.reset_index().rename(columns={"arm": "arm"})
    table(df, "coverage_table", lambda v: f"{v:.0%}", lambda v: "#f3cccc" if v < LOW_COVERAGE else None,
          col_width=1.3, first_width=2.8)


def hp_groups_table() -> None:
    """exp02 worker_hp.py: mean test loss on elevators, kr-vs-kp and socmob, three of the datasets where `A_et0` still
    trailed `A_def` after extra_trees=False (pol and okcupid-stem also trail and were not run)."""
    df = pd.read_parquet(RESULTS / "tidy" / "exp02.parquet")
    df = df[df.dataset.isin(HP_DATASETS) & (df.status == "ok")]
    df = df.assign(loss=[r.logloss if r.task != "regression" else r.rmse for r in df.itertuples()])
    m = df.pivot_table(index="arm", columns="dataset", values="loss")[HP_DATASETS].reindex(HP_ARMS)
    t = m.reset_index().rename(columns={"arm": "arm (mean of 5 folds, lower is better)", "elevators": "elevators (RMSE)", "socmob": "socmob (RMSE)", "kr-vs-kp": "kr-vs-kp (log loss)"})
    table(t, "exp02_hp_groups_table", lambda v: f"{v:.2f}" if v >= 1 else f"{v:.4f}", lambda v: None, col_width=1.7, first_width=3.4)


def lama_table() -> None:
    """exp06 LAMA diagnostics: mean RMSE per variant on pol and socmob, base GBMs of exp04 for reference."""
    d = pd.read_csv(RESULTS / "tidy" / "exp06_lama_diag.csv")
    t = d.pivot_table(index="arm", columns="dataset", values="rmse").sort_values("pol").reset_index()
    t = t.rename(columns={"arm": "arm (RMSE, mean of 5 folds)"})
    table(t, "exp06_lama_diag_table", lambda v: f"{v:.2f}", lambda v: None, col_width=1.7, first_width=3.6)


def main() -> None:
    summ = pd.read_csv(RESULTS / "tidy" / "exp04_summary.csv")
    eff = pd.read_csv(RESULTS / "tidy" / "exp04_effects.csv")
    df = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    for model in ("lgbm", "catboost"):
        median_effect(summ, model)
        win_tie_loss(eff, model)
    preview(summ)
    effect_scatter(df, eff, "gen_fit_s", "generator fit time, s", "effect_vs_gen_time")
    effect_scatter(df, eff, "n_new_features", "n new features", "effect_vs_n_features")
    coverage_table(summ)
    exp05_permuted()
    exp05_planted()
    for tag, pairs, ncols, name in (("exp02", exp02.MAIN, 3, "exp02_contrasts"), ("exp02", exp02.SINGLE, 4, "exp02_single_model"),
                                    ("exp06", None, 3, "exp06_contrasts")):
        per = pd.read_csv(RESULTS / "tidy" / f"{tag}_contrasts.csv")
        summary = pd.read_csv(RESULTS / "tidy" / f"{tag}_contrasts_summary.csv")
        pairs = pairs or list(exp06.CONTRASTS)
        contrasts(per, pairs, name, ncols)
        contrast_table(per, summary, pairs, f"{name}_table")
    hp_groups_table()
    lama_table()
    missing = [p.name for p in sorted(OUT.glob("*.png")) if not p.with_suffix(".alt.txt").exists()]
    assert not missing, f"no alt text (1 to 2 sentences on what is shown) for: {missing}"


if __name__ == "__main__":
    main()
