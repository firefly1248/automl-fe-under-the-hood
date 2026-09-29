# Progress

## 2026-09-29: Phases 0 and 1 done, A1 approved

**What ran** (locked envs, this laptop): `make env env-check data exp01 exp03`. All six envs import
(`results/ENVIRONMENT.md`). `h2o` env deferred (no Java; exp06 is optional).

**Env fixes** (lockfiles updated)
- `openfe`: added `matplotlib` (imported by openfe, not declared). OpenFE uses LightGBM callbacks, so LightGBM 4.7 is fine.
- `lama`: `numpy<2`, `pandas<3`, `scipy<1.16`. statsmodels 0.14.0 (LAMA dep) fails to import with scipy 1.16
  (`_lazywhere` removed) and with pandas 3 (`deprecate_kwarg` signature).

**Datasets** (`results/datasets.md`): 35 frames. Suites: 12 binary, 6 multiclass, 12 regression; 14 with categoricals
(no swaps needed). Typed: adult (did 1590), Amazon_employee_access (+ `_int`), diamonds, house_sales (did 42092);
bank-marketing and credit-g were already selected from CC18.

**For A1: corrections to plan section 1 and decisions**
1. 1.2 is wrong on selection: with default `selection_method="random"`, `ArithmeticFeatureGenerator` does not apply
   `corr_threshold` (only `spearman_selection` does). Row `subsample` is implemented despite the TODO comment.
2. 1.2/1.3 confirmed by `exp01_prep_configs.csv` / `exp01_presets.csv`: 14 cpu `GBM_PREP` configs
   (7 both, `_r47/_r27/_r37` Arithmetic, `_r1/_r19/_r14` CatInt+TE, `_r21` empty); gpu portfolio has 5.
3. 1.4 confirmed and extended (`exp01_types.csv`): `const` dropped as useless; `dup_num` and `uid` dropped as unused;
   int-coded `city_code` and `date_int` stay int64; `date_str` -> int64 + year/month/day/dayofweek;
   two-valued `cat_binary` and `near_const` -> int8 (`bool` special type); `text_short` -> category;
   `num_as_str` -> category (not numeric). New: `mixed` (numbers with "n/a") is typed as a datetime on every seed;
   the number of date parts kept varies by seed. `text_long` gives 503 columns at n = 5000, 12 at n = 500;
   `cat_rare` with singletons only is dropped.
4. exp01 step 4 (`exp01_tabprep.csv`): Arithmetic recovers the planted `a/b` on `hard_columns()`; the `_r13` group leaves
   no `category` columns (TE replaces them), as in 1.2.
5. 1.5 answered (`exp03_config.yaml`): all GBM algos (lgb, lgb_tuned, cb, cb_tuned) get `LGBAdvancedPipeline`;
   `linear_l2` gets `LinearFeatures`; selection mode 1 (LightGBM gain importance on `LGBSimpleFeatures`, cutoff 0);
   reader `drop_score_co` 0.0 (class default 0.01); `use_groupby` off.
6. exp03 roles (`exp03_roles.csv`): numeric -> category for `city_code`, Amazon ids, adult `age,fnlwgt,capital-gain,capital-loss`,
   bank-marketing `V1,V10,V14`, credit-g `residence_since`. LAMA keeps `dup_num` and `uid` (as ordinal category);
   `near_const` (99.9%) dropped by the constant-rate check; `text_long` becomes a category (no text role without
   a manual role).
7. Typed `adult` resolved by name to did 179 (v1), a binned version. A1 decision: pin CC18 did 1590.
8. Bike_Sharing_Demand has no date-typed column (only integer year/month/hour/weekday). A1 decision: replace it with
   `house_sales` did 42092 (date string `20141013T000000`, int-coded zipcode). LAMA reads its `date` as Datetime and
   turns `zipcode`, `yr_built`, `long` into categories.

**A1 decisions applied**: items 7 and 8; plan sections 1.2, 4.2 and 7.3 corrected. exp01 and exp03 rerun.
Remote: private repo `firefly1248/automl-fe-under-the-hood`; commits authored as firefly1248@gmail.com (repo-local config).

**Next**: Phase 2 (splits, run store, worker, limits, contract tests, kill-and-resume test).
