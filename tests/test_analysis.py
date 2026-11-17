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
