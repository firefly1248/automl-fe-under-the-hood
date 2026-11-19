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

## `not_applicable` units (by design)

`autofeat` builds features from numeric columns and needs at least one; `mljar_golden` builds them from
pairs of numeric columns and needs at least two. Amazon_employee_access, connect-4 and kr-vs-kp have no
numeric column (both arms skip them), socmob has one (`mljar_golden` skips it). Every fold of both repeats.

## Findings (both repeats; numbers live in `results/tidy/exp04_report.md` and `results/claims.yaml`)

- With untuned default LightGBM and CatBoost downstream (a design choice, plan 7.3), no arm moves the
  median primary metric across primary datasets by more than noise: the 95% bootstrap CIs of the median straddle zero
  in all but 9 of the 72 (task, model, arm) cells under the base-fallback rule, all small (a few tenths of an AUC point, 1 to 2%
  in loss); `lama_adv` is worse in three of them and `manual_baseline` in one
  (`results/figures/exp04/median_effect_*.svg`).
- The one Holm-significant across-dataset effect (Holm over the 9 arms of each task and model, not over all 72 tests) is negative: `lama_adv` on regression makes RMSE worse
  (claims `s4_lama_reg_median_lgbm`, `s4_lama_reg_wilcoxon_lgbm`).
- `openfe` has the widest per-dataset spread: the largest single-dataset wins (multiclass log loss) next
  to the highest share of descriptive losses on binary (`exp04_summary.csv`).
- `skrub_tv` (skrub 0.11.0 `TableVectorizer()` with all defaults, own env `envs/skrub`, fit on fit rows only, replaces the
  input; probe `experiments/exp08_skrub_probe`, defaults in `results/tidy/exp08_skrub_defaults.yaml`) ran on all 50 datasets,
  both repeats, both models: 990 of 990 fit units ok. Its median effect is 0 to within 1e-10 on every (task, model) cell but one
  (regression, LightGBM, -0.026%; `s4_skrub_tv_*`): the 40-level threshold sends most categoricals to one-hot and the
  numeric columns pass through unchanged, so on the many numeric-only datasets only the float32 cast changes anything.
- On the 28 primary datasets that have a categorical column (`n_categorical > 0`), the headline for `skrub_tv` and `ag_default`
  is a count, not the median over all 50 (which is mostly 0; the medians stay as secondary claims).
  Better by sign / clearly better / clearly worse (bootstrap CI over the splits excludes zero), of 28 datasets (27 for CatBoost,
  which skips connect-4): `ag_default` on LightGBM 1 / 1 / 1 (22 of the 28 identical to `base` within 1e-9), on CatBoost 15 / 3 / 2 (4 identical); `skrub_tv` on LightGBM 12 / 6 / 10, on
  CatBoost 11 / 5 / 7 (`afe.analysis.cat_subgroup`, `s4_cat_*`, `cat_subgroup_table`). `ag_default` is close to `base` once it
  keeps categorical columns as categories (next bullet; it drops or retypes some); `skrub_tv` turns them into numbers.
- Two protocol corrections of 2026-11-19, found by the exp09 diagnostic (`experiments/exp09_numcast/README.md`), applied to every
  record of this experiment: (1) every categorical column of these datasets is an ordered pandas category, which LightGBM's
  `categorical_feature="auto"` skips, so LightGBM had read all of them as integer codes in the first runs; `afe.models.downstream`
  now makes them unordered before the fit, so LightGBM uses its native categorical handling for `base` and every arm that keeps the
  columns; (2) the `ag_default` output (a category with integer categories) came back from the parquet cache as `int64`, so neither
  model had seen a categorical column under that arm; the generator now renames the levels to strings. All LightGBM fits and the
  `ag_default` generator and CatBoost fits were rerun (full_r0, full_r1, timing, exp05, exp09); the other CatBoost records, whose
  code path did not change, were moved to the new code hash (`rekey.py`, `rekeyed_from` in the record), and 36 of them (nine arms)
  recomputed afterwards matched to the last digit. Effects of the arms over `base` on LightGBM and all `ag_default` effects on
  CatBoost changed; every other CatBoost effect is identical to the first run.
- What that did to the large CatBoost effects on fps_benchmark, auction_verification and Brazilian_houses: `ag_default` was +59.7,
  +49.4 and +39.6% RMSE with its categoricals lost, and is +1.0, 0.0 and +3.4 now; `skrub_tv` (all categoricals become numbers)
  keeps +68.8, +51.2 and +20.7. So the effect was CatBoost's categorical handling against plain numbers on these datasets, not
  generated features: `base_codes` (`base` with every categorical as integer codes) gives +60.2, +49.4 and +35.1 (`numcast_table`).
- Coverage separates the tools more than accuracy does: `autofeat` finishes well under the 3,600 s cap on
  62% of its planned units (65.5% of the 940 applicable ones: the 50 not-applicable units count as unfinished in the coverage, as
  70 of `mljar_golden`'s do), `openfe` and `mljar_golden` on 92 to 94%, `ag_tabprep` and `ag_tabprep_keepcat` on CatBoost on 94%
  (100% on LightGBM), the rest are at 100% (coverage table in `exp04_report.md`; failures above).
- Cost: on the jobs=1 timing pass, `ag_tabprep` and `openfe` multiply the whole pipeline time by tens
  versus fitting the same model on raw columns, mostly through wider downstream inputs
  (`exp04_timing_summary.csv`).

Caveats: per-dataset tests are near-powerless and descriptive only (plan 7.5); the AUC differences sit in
the third decimal; results are for defaults-no-retune downstream, not for tuned pipelines.

Test details. The per-dataset verdicts use the Nadeau-Bengio corrected t-test with the ratio
n_test / n_train taken over the fit rows (`n_train` is the fit rows, not the outer train): the fit rows are
68% of a dataset (outer train 80% minus the 15% inner split), the test fold 20%, so the ratio is 0.294
(median over units, range 0.293 to 0.296); with the outer train it would be 0.25. `share_win` and
`share_loss` in `exp04_summary.csv` are shares of the `n_datasets` datasets that have data for that arm
(under `complete_cases` a dataset with no complete split is not counted), not of all primary datasets;
the `coverage` column is the mean over all listed datasets (a dataset with no complete split counts as 0).
The median-effect figures mark every (task, model, arm) cell where fewer than 80% of the units are ok with a
hollow marker and the coverage written next to it; the win/tie/loss figures write the coverage label only.
