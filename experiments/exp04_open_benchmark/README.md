# exp04 open benchmark

## `openfe` failures (tool, not wrapper)

`openfe` fails on every fold of three datasets. Each failure reproduces on repeat 0 fold 0 with a direct
`OpenFE().fit` on the raw fit rows (original columns and dtypes, `task`, `categorical_features`, `n_jobs=1`,
the unit seed), without the wrapper:

| dataset | error | cause |
|---|---|---|
| sick | `KeyError` in `GroupByThenNUnique` (`FeatureGenerator.py:114`) | the operator looks up each row's group in a per-group count that has no entry for it: a missing value in the group column through the wrapper (`sex` has 150, `TBG` is all missing), a category value in the direct call |
| fps_benchmark | the same `KeyError` | the same; `GpuNumberOfExecutionUnits` is all missing |
| yeast | `ValueError: y contains previously unseen labels` (`openfe.py`, `_evaluate`) | 10 classes, down to 3 fit rows each: OpenFE's stage-1 scoring fits LightGBM on a row subsample that lacks a class present in its validation rows |

These units stay `failed`; OpenFE calls exit() in its worker pool, which the wrapper turns into an exception.
