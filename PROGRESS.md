# Progress

## 2026-09-30: Phase 2 done (splits, run store, worker, limits, tests)

**What was built**
- `make splits`: `results/splits/<key>.parquet` for all 52 frames (one row per row and repeat: `outer_fold`,
  `inner_mask` bit per fold). Stratified per-class round-robin folds, 15% inner per class; numpy only.
- `afe.run`: `unit.py` (hashes), `store.py` (atomic JSON/parquet, skip rules), `limits.py` (tree-RSS watchdog,
  wall clock, free-RAM launch gate), `worker.py` (gen stage in the arm env, fit stage in core), `cli.py`
  (`make exp04-smoke ARGS="--jobs N --retry-failed"`).
- `afe.generators.base` (interface, `BaseView`, `finalize`, `combine`), arm registry, `base` arm;
  `afe.models.downstream` (LightGBM, CatBoost; early stop on inner; metrics of 7.4).
- `experiments/exp04_open_benchmark/config.yaml` (protocol, limits, stages; only `base` so far).
- Tests (`make test`): splits, base view, limits (ok/failed/timeout/oom), downstream (3 tasks x 2 models, missing
  class), contract tests per env, kill-and-resume (orchestrator killed mid-run; rerun skips finished units, 15 of 15
  records, no duplicates). All green.

**Smoke with `base`** (credit-g, cmc, Moneyball, fold 0): 9/9 ok; generator 0.7 s, LightGBM ~2 s, CatBoost ~7 s per unit.

**Deviations and surprises**
- `protocol_hash` hashes the `protocol:` section of the exp config, not the whole file, so editing stage lists or
  limits does not invalidate finished units. Limits are stored in every record instead.
- Free-RAM launch gate is `min_free_gb: 2` in the config: with other apps open only ~7 GB of 24 GB is free, and a
  6 GB gate stalled the smoke run for 9 minutes. The full run needs the laptop mostly free.
- LightGBM 4.7 deprecates `eval_set`; the code uses `eval_X` / `eval_y`.

**Next**: Phase 3 (arm wrappers, exp04 smoke with all arms) -> A2.

## 2026-09-29: plan v3 after a second review (all 14 findings accepted)

Plan changes (A4-level, approved by Ilia): pool extended with OpenML suites 271, 269 (AMLB), 334, 335
(tabular benchmark, categorical); filters for duplicate rows (> 5%), optdigits as image, electricity excluded
(temporal); one dataset per family in the primary aggregate; quotas 20/10/20 with >= 10-level categoricals first;
1 repeat x 5 folds; AUC raw difference instead of gap closed; descriptive per-dataset tests without Holm;
both failure rules with equal weight; timing from a separate jobs=1 pass; jobs = min(cores - 2, 10) with a
free-RAM launch gate; generator cap 3,600 s in smoke/pilot, full cap from pilot p95; CatBoost lr 0.1 / 3,000
iterations; `core_lock_hash` in the fit unit; arm `ag_tabprep_keepcat` added, `ag_tabprep_groupby` dropped,
`cheap_manual` renamed `manual_baseline`; exp05 check 1 restated, OpenFE equivalence test added; figures get
claims entries; improvement vs n new features figure; exp02 600 s caveat; untuned downstream stated as a design choice.

New registry run (`results/datasets.md`): 50 primary datasets (20 binary, 10 multiclass, 20 regression) + 2 case
studies (`Amazon_employee_access_int`, `house_sales_42092` with the date string; exp03 now uses the latter).
23 candidates rejected for > 5% duplicate rows (all NASA MDP, spambase, wine sets, PhishingWebsites, ...).
Only 17 primary datasets (34%) have a categorical column with >= 10 levels: every such candidate in the six suites
is already selected, so the 50% target is not reachable with this pool. Decision (Ilia): keep 50 datasets; claims
about categoricals use the 17-dataset subgroup.

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
