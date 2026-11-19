"""Regression tests for the analysis helpers (core env)."""

import numpy as np
import pandas as pd

from afe.analysis import exp02, stats


def test_contrast_counts_and_figure_name(monkeypatch, tmp_path):
    rows = [  # one dataset clearly better, one tied at zero, one clearly worse (fold CI)
        {"contrast": "X-Y", "dataset": "a", "task": "binary", "n_folds": 5, "effect": 0.1, "ci_lo": 0.05, "ci_hi": 0.15},
        {"contrast": "X-Y", "dataset": "b", "task": "binary", "n_folds": 5, "effect": 0.0, "ci_lo": 0.0, "ci_hi": 0.0},
        {"contrast": "X-Y", "dataset": "c", "task": "binary", "n_folds": 5, "effect": -0.1, "ci_lo": -0.2, "ci_hi": -0.05},
    ]
    per = pd.DataFrame(rows)
    ok = pd.DataFrame({"dataset": list("aaabbbccc"), "task": "binary", "fold": [0, 1, 2] * 3, "arm": "X",
                       "status": "ok", "logloss": [0.8, 0.9, 1.0, 1.0, 1.0, 1.0, 1.2, 1.1, 1.3]})
    ref = ok.assign(arm="Y", logloss=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    _, summ = exp02.contrasts(pd.concat([ok, ref]), {"X-Y": ("X", "Y")})
    row = summ.iloc[0]
    assert (row.n_datasets, row.n_better, row.n_clear_worse) == (3, 1, 1)
    saved = []
    monkeypatch.setattr(exp02, "save", lambda fig, sub, name, data=None: saved.append((sub, name)))
    exp02.figure(per, "expT", pairs=["X-Y"], name="contrasts")
    assert saved == [("expT", "contrasts")]  # the loop variable must not rename the file


def test_share_win_uses_datasets_with_data():
    eff = pd.DataFrame({
        "dataset": ["a", "b", "c", "d"], "task": "regression", "arm": "arm", "model": "lgbm", "effect": "rmse_rel",
        "rule": "complete_cases", "value": [0.1, 0.2, np.nan, 0.0], "verdict": ["win", "win", "no_data", "tie"],
        "coverage": [1.0, 1.0, 0.0, 1.0]})
    row = stats.summary_table(eff).iloc[0]
    assert row.n_datasets == 3 and abs(row.share_win - 2 / 3) < 1e-9 and row.share_loss == 0


def test_noise_spread_and_ci():
    """Three runs of D on pol against the repeated-less reference A_def: spread of the per-run effects in points."""
    from afe.analysis import noise

    rows = []
    for run, shift in ((0, 0.0), (1, 0.01), (2, -0.02)):
        for fold in range(5):
            rows += [{"exp": "exp02", "dataset": "pol", "task": "regression", "arm": "D", "fold": fold, "run": run,
                      "loss": 1.0 + shift, "wall_s": 600.0},
                     {"exp": "exp02", "dataset": "pol", "task": "regression", "arm": "A_def", "fold": fold, "run": 0,
                      "loss": 2.0, "wall_s": 10.0}]
    df = pd.DataFrame(rows)
    arms = noise.arms(df)
    assert len(arms) == 1 and arms.n_runs.iloc[0] == 3
    assert np.isclose(arms.spread_rel.iloc[0], 0.03 / (2.99 / 3))  # (1.01 - 0.98) / mean of the run means
    c = noise.contrasts(df)
    d = c[c.contrast == "D-A_def"].iloc[0]
    assert np.isclose(d.effect_run0, 0.5) and np.isclose(d.effect_run1, 0.495) and np.isclose(d.effect_run2, 0.51)
    assert np.isclose(d.spread_pp, 1.5)


def test_ctr_ablation_effects():
    """Arm over base with default CatBoost and with ctr=1, and their paired difference, on a toy loss table."""
    from afe.analysis import exp07

    folds = [(r, f) for r in (0, 1) for f in range(5)]
    base = {"default": 1.0, "ctr1": 1.1}
    arm = {"default": 0.9, "ctr1": 0.88}  # openfe: effect 10% by default, 20% with ctr=1
    drows, crows = [], []
    for r, f in folds:
        for a, vals in (("base", base), ("openfe", arm)):
            drows.append({"dataset": "d", "task": "binary", "repeat": r, "fold": f, "arm": a, "loss": vals["default"]})
            crows.append({"dataset": "d", "task": "binary", "repeat": r, "fold": f, "arm": a, "status": "ok",
                          "wall_s": 1.0, "loss": vals["ctr1"]})
    eff = exp07.effects(pd.DataFrame(crows), pd.DataFrame(drows))
    o = eff[eff.arm == "openfe"].iloc[0]
    assert np.isclose(o.effect_default, 0.10) and np.isclose(o.effect_ctr1, 0.20) and np.isclose(o.effect_diff, 0.10)
    b = eff[eff.arm == "base"].iloc[0]
    assert np.isclose(b.effect_diff, -0.10) and o.n_folds == 10  # CatBoost without combinations loses 10% on base
    s = exp07.summary(eff).set_index("arm")
    assert s.loc["openfe", "n_better"] == 1 and s.loc["base", "n_better"] == 0
