# exp09: why `skrub_tv` (and, first, `ag_default`) helped CatBoost by 20 to 70% on some datasets

Question (author, 2026-11-19): the first exp04 runs showed +40 to +60% RMSE for `ag_default` and `skrub_tv` on CatBoost on
fps_benchmark, auction_verification and Brazilian_houses. Where does it come from? Diagnostic on those three datasets and, as a
contrast, socmob and Amazon_employee_access.

Parts: `diagnose.py` (type table, `make aggregate`), arm `base_numcast` (`src/afe/generators/base_numcast.py`: `base` with the
categorical columns whose levels all parse as numbers cast to float, all-missing columns as NaN floats), arm `base_codes`
(`base_codes.py`: `base` with every categorical column as integer codes, NaN for missing and unseen levels); both replace the input.
`make exp09` (exp04 protocol and limits, both repeats, both models, the five datasets: 100 generator and 200 fit units, all ok),
`afe.analysis.exp09` (effects over `base`, same definition as `exp04_effects.csv`: AUC difference or relative reduction of the
mean loss, paired bootstrap 95% CI over the 10 splits). Outputs: `results/tidy/exp09_columns.csv`, `exp09_dtypes.csv`,
`exp09_ordered.csv`, `exp09_effects.csv`; article tables `exp09_types_table`, `numcast_table`; claims `s2_types_*`, `s4_numcast_*`.

## Two protocol artifacts found, fixed the same day

The first exp09 run (only `base_numcast`) showed that the model inputs were not what the protocol described:

- LightGBM's `categorical_feature="auto"` takes unordered pandas categoricals only (`lightgbm/basic.py`, `cat_cols_not_ordered`).
  Every categorical column of the 28 primary datasets with categoricals (301 of 301 columns) is an ordered category in the cached
  frames (`s2_types_ordered_datasets`), so LightGBM had read them as integer codes (checked on socmob: identical test score and best
  iteration with the categorical and with its codes). Fix: `afe.models.downstream` makes ordered categoricals unordered before the
  LightGBM fit.
- AutoGluon returns a category with integer categories; the parquet cache of the run store turns such a column into `int64`
  (pandas 3.0.6, pyarrow 25, core env), so neither model saw a categorical column under `ag_default`. Fix: the arm renames the
  levels to strings (a contract test now rejects numeric levels in any arm output).

Both changed hashed code, so every LightGBM fit and the `ag_default` units of exp04, exp05 and exp09 were rerun; CatBoost records
whose code path did not change were moved to the new hash and 36 of them, recomputed, matched exactly (see the exp04 README).
Numbers below are after both fixes; the first ones are in git history (737a375 and the working tree of the same day).

## What reaches the models (after the fixes)

CatBoost gets a column as a categorical feature (`cat_features`) and LightGBM as a native categorical when its dtype is a pandas
category (`exp09_types_table`, `exp09_dtypes.csv`). `base` and `ag_default` keep the categorical columns (`ag_default` drops or
retypes some: fps_benchmark 14 to 6, Brazilian_houses 4 to 2, socmob 4 to 2); `skrub_tv` outputs only float32 columns (one-hot,
numeric, text encodings); `base_numcast` and `base_codes` give numbers for the columns they cast.

## Result (`numcast_table`, effect over `base`, % RMSE or AUC points; * = CI excludes zero)

CatBoost:

| dataset | `ag_default` | `skrub_tv` | `base_numcast` | `base_codes` | number-like categorical columns |
|---|---|---|---|---|---|
| auction_verification | 0.0 | +51.2* | +49.4* | +49.4* | 2 of 2 |
| fps_benchmark | +1.0 | +68.8* | +2.3 | +60.2* | 4 of 14 |
| Brazilian_houses | +3.4 | +20.7 | +15.4* | +35.1* | 1 of 4 |
| socmob | +2.0* | +9.4* | 0.0 | +1.6 | 0 of 4 |
| Amazon_employee_access | -0.5* | -3.0* | -5.3* | -5.3* | 9 of 9 |

LightGBM:

| dataset | `ag_default` | `skrub_tv` | `base_numcast` | `base_codes` |
|---|---|---|---|---|
| auction_verification | 0.0 | +9.1 | +19.9* | +19.9* |
| fps_benchmark | 0.0 | -20.6* | 0.0 | -59.7* |
| Brazilian_houses | 0.0 | +0.2 | +6.5 | +5.6 |
| socmob | 0.0 | -18.8* | 0.0 | -14.0* |
| Amazon_employee_access | +0.1* | +1.6* | -0.1 | -0.2 |

- The large `ag_default` effect of the first runs was the lost categorical type: with categorical columns kept, `ag_default` is
  within a few percent of `base` on all five datasets for both models (CatBoost +1.0, 0.0, +3.4 against +59.7, +49.4, +39.6 before).
- What is left is the effect of plain numbers against CatBoost's categorical handling. `skrub_tv` (all numbers) gains 51, 69 and 21%
  on CatBoost, and `base_codes` (only the types change) gains 49, 60 and 35%, so the gain is not made by generated features.
  On auction_verification it is the cast of the two number-like columns alone (`base_numcast` equals `base_codes`); on
  fps_benchmark only the coding of the nine string categoricals reproduces it (`base_numcast` +2.3, `base_codes` +60.2); on
  Brazilian_houses the numeric-like column `floor` gives +15.4 and the other three columns the rest, with wide CIs
  (RMSE varies from about 1,800 to 22,700 between folds).
- LightGBM shows the opposite on fps_benchmark and socmob: with native categorical handling `base` beats integer codes by 60 and
  14%, so the same type decision that helps CatBoost hurts LightGBM. On auction_verification numbers help both models.
- Amazon_employee_access is the contrast: all nine columns are identifiers with number-like levels (up to 6,136); numbers cost
  CatBoost 5.3 AUC points and LightGBM nothing.
- Next to exp07: with `max_ctr_complexity=1` the base CatBoost on fps_benchmark is 537% worse in RMSE, so CatBoost's own
  combinations matter a great deal there, and plain numbers still beat both.

Limits: five datasets, effects with wide CIs on Brazilian_houses and socmob; one cast rule for `base_numcast` (every level parses
as a number, levels from the fit rows); the all-missing column of fps_benchmark is an all-NaN float in `base_numcast` and
`base_codes` (an empty category does not survive the parquet cache), no information either way; `base_codes` orders levels as
text, which for LightGBM is an arbitrary ordering of the categories.
