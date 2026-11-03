"""Fold assignments (plan sections 4.2, 7.1, 7.2). numpy/pandas only.

One row per (row, repeat): `outer_fold` is the fold in which the row is a test row; bit k of
`inner_mask` is set if the row is inner validation in fold k. Classification folds are stratified
(per-class shuffle, round-robin over folds); the inner split takes 15% of each class of the outer
train. Written once by `make splits` to results/splits/<key>.parquet and committed.
"""

import hashlib

import numpy as np
import pandas as pd

from afe.paths import RESULTS

N_FOLDS = 5
REPEATS = 2
INNER_SHARE = 0.15


def repeat_seed(r: int) -> int:
    return 1000 + r


def tool_seed(r: int, k: int) -> int:
    return 1000 + 10 * r + k


def _groups(y: np.ndarray, task: str) -> list[np.ndarray]:
    if task == "regression":
        return [np.arange(len(y))]
    return [np.flatnonzero(y == c) for c in np.unique(y)]


def make(y: pd.Series, task: str, n_folds: int = N_FOLDS, repeats: int = REPEATS) -> pd.DataFrame:
    y = y.to_numpy()
    frames = []
    for r in range(repeats):
        rng = np.random.default_rng(repeat_seed(r))
        outer = np.empty(len(y), dtype=np.int8)
        offset = 0
        for idx in _groups(y, task):
            perm = rng.permutation(idx)
            outer[perm] = (np.arange(len(perm)) + offset) % n_folds
            offset += len(perm)  # continue the round-robin so small classes do not all start at fold 0
        mask = np.zeros(len(y), dtype=np.int8)
        for k in range(n_folds):
            rng_k = np.random.default_rng(tool_seed(r, k))
            train = outer != k
            for idx in _groups(y, task):
                idx = idx[train[idx]]
                inner = rng_k.choice(idx, size=int(round(INNER_SHARE * len(idx))), replace=False)
                mask[inner] |= np.int8(1 << k)
        frames.append(pd.DataFrame({"row": np.arange(len(y)), "repeat": np.int8(r), "outer_fold": outer, "inner_mask": mask}))
    return pd.concat(frames, ignore_index=True)


def rows(splits: pd.DataFrame, repeat: int, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row positions (fit, inner, test) for one outer fold."""
    s = splits[splits.repeat == repeat]
    test = s.outer_fold.to_numpy() == fold
    inner = ~test & ((s.inner_mask.to_numpy() >> fold) & 1).astype(bool)
    fit = ~test & ~inner
    r = s.row.to_numpy()
    return r[fit], r[inner], r[test]


def path(key: str):
    return RESULTS / "splits" / f"{key}.parquet"


def load(key: str) -> pd.DataFrame:
    return pd.read_parquet(path(key))


def splits_hash(splits: pd.DataFrame) -> str:
    return hashlib.sha1(pd.util.hash_pandas_object(splits, index=False).to_numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    """Write splits for every dataset in results/datasets.csv that has none yet or fewer repeats. Each repeat
    has its own seeds, so adding a repeat leaves the earlier ones unchanged."""
    from afe.data.loaders import encode_y, load as load_data

    (RESULTS / "splits").mkdir(parents=True, exist_ok=True)
    for key in pd.read_csv(RESULTS / "datasets.csv").key:
        if path(key).exists() and load(key).repeat.nunique() == REPEATS:
            continue
        _, y, meta = load_data(key)
        make(encode_y(y, meta["task"]), meta["task"]).to_parquet(path(key), index=False)


if __name__ == "__main__":
    main()
