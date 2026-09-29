"""Synthetic frames with a known generating process (plan section 4.1)."""

import numpy as np
import pandas as pd

TEXT_SHORT = ["fast delivery", "late delivery", "item damaged", "great service"]
TARGET_WORD = "refund"


def _vocabulary(rng: np.random.Generator, size: int = 500) -> np.ndarray:
    letters = np.array(list("abcdefghijklmnopqrstuvwxyz"))
    words = {"".join(rng.choice(letters, rng.integers(3, 9))) for _ in range(size * 2)}
    words.discard(TARGET_WORD)
    return np.array(sorted(words)[: size - 1] + [TARGET_WORD])


def hard_columns(n: int = 5000, seed: int = 0, cat_rare_mode: str = "mixed") -> tuple[pd.DataFrame, pd.Series, dict]:
    """Columns that stress type inference, plus a binary target with a known formula.

    cat_rare_mode: "mixed" (3 frequent levels + 300 levels seen once or twice) or
    "singletons" (every value seen exactly once).
    """
    rng = np.random.default_rng(seed)
    num = rng.normal(size=n)

    city_code = rng.integers(0, 40, n)
    city_effect = rng.normal(0, 1, 40)  # non-monotone in the code

    dates = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 5 * 365, n), unit="D")
    month = dates.month.to_numpy()
    month_effect = np.where(np.isin(month, [11, 12]), 1.0, 0.0)
    dates_str = pd.Timestamp("2019-01-01") + pd.to_timedelta(rng.integers(0, 5 * 365, n), unit="D")

    a = rng.lognormal(0, 0.5, n)
    b = rng.lognormal(0, 0.5, n)
    log_ratio = np.log(a / b)

    if cat_rare_mode == "mixed":
        rare = np.concatenate([np.repeat(np.arange(150), 2), np.arange(150, 300)])  # 150 twice, 150 once
        n_rare_rows = len(rare)
        rare = np.array([f"r{i}" for i in rare])
        freq = rng.choice(["f0", "f1", "f2"], n - n_rare_rows)
        cat_rare = rng.permutation(np.concatenate([freq, rare]))
    else:
        cat_rare = np.array([f"s{i}" for i in rng.permutation(n)])

    vocab = _vocabulary(rng)
    has_word = rng.random(n) < 0.2
    words = rng.choice(vocab[:-1], size=(n, 20))
    text_long = [" ".join(w) + (f" {TARGET_WORD}" if h else "") for w, h in zip(words, has_word)]

    mixed = rng.normal(10, 3, n).round(2).astype(object)
    mixed[rng.random(n) < 0.1] = "n/a"

    X = pd.DataFrame(
        {
            "num": num,
            "dup_num": num.copy(),
            "const": np.ones(n),
            "near_const": np.where(rng.random(n) < 0.001, 1.0, 0.0),
            "city_code": city_code,
            "date_int": (dates.year * 10000 + dates.month * 100 + dates.day).to_numpy().astype(np.int64),
            "date_str": dates_str.strftime("%Y-%m-%d"),
            "ts_unix": (dates.to_numpy().astype("datetime64[s]").astype(np.int64) + rng.integers(0, 86400, n)),
            "cat_low": rng.choice(["low", "mid", "high"], n),
            "cat_rare": cat_rare,
            "cat_binary": rng.choice(["yes", "no"], n),
            "uid": [f"id_{i:07d}" for i in rng.permutation(n)],
            "text_short": rng.choice(TEXT_SHORT, n),
            "text_long": text_long,
            "num_as_str": rng.normal(100, 20, n).round(1).astype(str),
            "mixed": mixed,
            "a": a,
            "b": b,
        }
    )
    logit = city_effect[city_code] + month_effect + 1.5 * log_ratio / log_ratio.std() + 1.0 * has_word - 0.8
    logit += rng.normal(0, 0.5, n)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="target")
    meta = {
        "task": "binary",
        "formula": (
            "logit = city_effect[city_code] + 1.0*[month(date_int) in {11,12}] + 1.5*log(a/b)/std(log(a/b))"
            f" + 1.0*['{TARGET_WORD}' in text_long] - 0.8 + N(0, 0.5); y ~ Bernoulli(sigmoid(logit))"
        ),
        "city_effect": city_effect.round(4).tolist(),
        "signal_columns": ["city_code", "date_int", "a", "b", "text_long"],
        "n": n,
        "seed": seed,
        "cat_rare_mode": cat_rare_mode,
    }
    return X, y, meta
