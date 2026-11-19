"""fps_benchmark probe (2026-11-19), core env, run by hand: `python experiments/exp09_numcast/probe_onehot.py`.

Why does `base_codes` (every categorical as an integer code) give CatBoost +60% RMSE over `base` on fps_benchmark while
`base_numcast` (only the number-like categoricals as numbers) gives +2.3%? Same 10 splits, same input as `base` (the base view),
the exp04 CatBoost fit (`afe.models.downstream`) with one change: `one_hot_max_size=255`, so every categorical column (at most 27
levels here) is one-hot encoded and CatBoost builds no target statistics (CTRs) on them; its default is 2, so only the two-level
columns are one-hot and the others get CTRs. Control: the unchanged fit of split (0, 0) must equal the exp04 `base` record.
Writes results/tidy/exp09_onehot_probe.csv: per split the test RMSE of `base` (exp04), `onehot255` (this probe) and `base_codes`
(exp09), then the mean effect over `base` of the two arms with the paired bootstrap CI of `afe.analysis.cat_subgroup`.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import yaml

from afe.analysis import exp09
from afe.analysis.cat_subgroup import effect_ci
from afe.data import splits as S
from afe.data.loaders import encode_y, load
from afe.generators.base import BaseView
from afe.models.downstream import fit_score
from afe.paths import RESULTS, ROOT

DATASET = "fps_benchmark"
PARAMS = yaml.safe_load((ROOT / "experiments" / "exp09_numcast" / "config.yaml").read_text())["protocol"]["models"]["catboost"]


def fit(r: int, k: int, one_hot: bool) -> float:
    import catboost

    if one_hot:  # downstream imports the estimator at call time, so the patch reaches it
        catboost.CatBoostRegressor = partial(catboost.CatBoostRegressor, one_hot_max_size=255)
    X, y, meta = load(DATASET)
    y = encode_y(y, meta["task"])
    idx = dict(zip(("train", "inner", "test"), S.rows(S.load(DATASET), r, k)))
    view = BaseView().fit(X.iloc[idx["train"]])
    parts = [(view.transform(X.iloc[idx[s]]), y.iloc[idx[s]]) for s in ("train", "inner", "test")]
    return fit_score("catboost", PARAMS, S.tool_seed(r, k), meta["task"], 0, *parts)["test"]["rmse"]


def main() -> None:
    splits = [(r, k) for r in range(2) for k in range(5)]
    with ProcessPoolExecutor(3) as pool:
        onehot = list(pool.map(fit, *zip(*splits), [True] * 10))
        control = fit(0, 0, False)
    e04 = pd.read_parquet(RESULTS / "tidy" / "exp04.parquet")
    base = e04[(e04.dataset == DATASET) & (e04.arm == "base") & (e04.model == "catboost")].set_index(["repeat", "fold"]).rmse
    assert control == base[0, 0], (control, base[0, 0])
    codes = exp09.tidy().query("dataset == @DATASET and arm == 'base_codes' and model == 'catboost'").set_index(["repeat", "fold"]).rmse
    df = pd.DataFrame({"repeat": [r for r, _ in splits], "fold": [k for _, k in splits], "base": base.loc[splits].to_numpy(),
                       "onehot255": onehot, "base_codes": codes.loc[splits].to_numpy()})
    df.round(6).to_csv(RESULTS / "tidy" / "exp09_onehot_probe.csv", index=False)
    rng = np.random.default_rng(0)
    print(df.round(4).to_string(index=False), "\ncontrol (0, 0) equals the exp04 record:", control)
    for arm in ("onehot255", "base_codes"):
        e, lo, hi = effect_ci(df[arm].to_numpy(), df["base"].to_numpy(), "regression", rng)
        print(f"{arm}: effect over base {e * 100:+.1f}% RMSE, CI [{lo * 100:.1f}, {hi * 100:.1f}]")


if __name__ == "__main__":
    main()
