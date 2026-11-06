# Progress

## 2026-11-06: record hygiene before `full_r1`

- Home path: `limits` stripped the home path after cutting the error log, so a cut kept a piece of it in 13
  `openfe` records (sick 5, fps_benchmark 5, yeast 3). It now strips before the cut, and `strip_home` also
  removes the pieces a cut leaves at either end; every record rewritten.
- `make test` scans every tracked file, including the records inside `*.jsonl.gz`, for home and temp paths
  (whole or cut), the user name and the host name (`tests/test_privacy.py`).
- Records carry git_sha "...-dirty". `tests/test_records.py`: each record's `code_hash` equals the hash of the
  committed code; 0 of 8,809 records differ, in every arm.
- `openfe` failures on sick, yeast and fps_benchmark reproduce with a direct `OpenFE().fit` without the
  wrapper: tool failures, causes in `experiments/exp04_open_benchmark/README.md`; no rerun.
- README states that the numbers are not final.

## 2026-11-06: exp04 `full_r0`

Repeat 0, 50 datasets x 5 folds x 8 generator arms, 3 jobs. Generator units: 1,869 ok, 75 timeout (all
`autofeat`, 15 datasets), 35 not_applicable (`autofeat` on Amazon_employee_access, connect-4, kr-vs-kp;
`mljar_golden` on Amazon_employee_access, socmob), 16 failed (`openfe` inside the library on sick, yeast and
fps_benchmark, every fold; `autofeat` on one yeast fold), 5 oom (`autofeat`, numerai28.6). Fit units: 4,166
ok, 37 timeout (all CatBoost: `ag_tabprep` and `ag_tabprep_keepcat` on GesturePhaseSegmentationProcessed,
pendigits, texture and wall-robot-navigation, `openfe` on texture), 252 without a
generator output. The timing stage is complete as well (45 ok, 1 oom, 2 not_applicable).

## 2026-11-03: code review after A3 (`/code-review`, `/simplify`)

**openfe fix** (results change: `openfe` and `openfe_transductive` rerun in every stage)
- `GroupByThenRank` on inner and test rows was (below + equal / 2) / n, on fit rows pandas' (below +
  (equal + 1) / 2) / n: each new-row rank 0.5 / n lower than the same value on fit rows (0.025 in a group of
  20), a train/test shift made by the inductive evaluator. Fixed and vectorized (26x faster at 15k rows,
  2,000 groups). Test 7 had encoded the same formula; a new test (fit rows through `transform` get their
  `fit_transform` values) fails on the old code.
- Gain filter rewritten as one pass; kept features and gains equal the pilot records on
  concrete_compressive_strength, cmc and credit-g (test 7). `openfe_transductive` turns an exit() in
  OpenFE's `transform` into an exception, as the search already did.

**PYTHONHASHSEED**: every arm and both downstream models on the contract frames (synthetic, credit-g,
Moneyball), two processes with hash seeds 1 and 2: identical outputs except `autofeat` (3 of 3 frames).
`PYTHONHASHSEED=0` for every child, in the protocol hash. Records of the other arms moved to the new hashes
by unit key (1,016 gen, 2,300 fit); 10 of 10 random moved pilot units rerun from scratch under hash seed 0
matched bitwise (metrics and generator parquet, 6 arms). `autofeat` rerun in every stage.

**Watchdog**: each record keeps the watchdog's marker (`watchdog`: reason, peak tree RSS); `oom` and
`timeout` come only from it, any other SIGKILL is `failed` (before, exit code -9 or 137 counted as `oom`).
The 6 `oom` of pilot and timing are all `autofeat` on numerai28.6, killed by the watchdog at a peak of
6,195-6,952 MB (cap 6,144).

**Other**: `pack` merges into the existing archive (tests: a repeated pack adds nothing, `*.json.tmp` is
left alone); hashed files must have distinct names (asserted in `code_hash`); slow arms (`autofeat`,
`openfe`) are launched first; the report scripts share `cli.records`, outputs byte-identical except the
check 3 heading (now repeat 0) and `exp04_smoke.md` (older than the smoke records it summarized).
exp05 results and the estimate below are recomputed after the `openfe` and `autofeat` reruns (2026-11-06):
per-unit metrics moved by up to a few thousandths, no check changed its verdict; check 5 byte-identical.

**Known limitations, not fixed (each would rerun everything)**
- Seeds: `repeat_seed(0) = tool_seed(0, 0) = 1000` and `repeat_seed(1) = tool_seed(0, 1)`, so one seed draws
  the outer folds of a repeat and the inner rows (and exp05 permuted labels) of a fold. No test row leaks;
  the draws are not independent streams.
- `splits_hash` covers the whole splits file: adding a repeat changes every unit hash (handled at A3 by
  moving records by unit key).
- `code_hash` hashes file names without their directories; distinct names are now asserted.
- Records before this change took `oom` from the exit code; all 6 were watchdog kills (above).

## 2026-11-03: exp04 pilot, exp05 leakage checks, full-run estimate -> A3

**A3 decisions (Ilia), applied**
- `autofeat`: limits not raised, arm kept; failures are results (section 7.6: base fallback main, complete
  cases secondary, coverage alongside).
- connect-4: base CatBoost timed out at 1800 s on 5 of 5 pilot folds, so connect-4 x CatBoost is in the config
  `exclude` list (not run, not compared). Fit limit 3600 s for all datasets.
- The `not_applicable` bug fixed (a skipped generator wrote no `describe()`); status `not_applicable`, counted as
  base in aggregates, its own coverage row.
- `code_hash` per arm: the arm module + shared code that affects results (`generators/base.py`,
  `run/worker.py`, `data/loaders.py`, `data/splits.py`; fit units add `models/`). Before migration 25 of 25
  random pilot units rerun from scratch matched bitwise (metrics and generator parquet); records moved to the
  new hashes, stale ones removed.
- OpenFE deviations accepted; `openfe_transductive` built as a separate exp05 arm, its time not published.
- LAMA pandas 2 bug: minimal reproduction in `experiments/exp03_lama_probe/pandas2_bug.py` and its README.

**Found on the way**
- `autofeat` output depended on the per-process str/set hash seed: 3 of 3 rerun units differed (14 vs 28
  features on one concrete fold). Contract test 2 runs both fits in one process, so it missed this. Every
  child now gets `PYTHONHASHSEED` = the unit's tool seed; repeat runs are bitwise identical. `autofeat`
  records were not migrated and were rerun.
- `lama_adv` on Moneyball: digit-valued categoricals read as Numeric by LAMA's reader; the arm coerces them.

**Pilot** (10 datasets x 5 folds): 370 of 400 generator units ok. `autofeat`: timeout on albert, kick,
mfeat-fourier (15), oom on numerai28.6 (5), not applicable on connect-4 (5); `mljar_golden` not applicable on
connect-4 (5). All 805 fit units ok, including the two mfeat-fourier CatBoost fits that needed more than 1800 s.

**exp05** (`results/tidy/exp05_report.md`)
1. Permuted labels (6 datasets x 5 folds): every arm x model x task has a 95% bootstrap CI containing 0 or
   below it.
2. TE delivery: median gap fit - test about 0 for `ag_tabprep` and `lama_adv`; flags sit on the four datasets
   under 1,500 rows, where raw columns flag at a similar rate, and none on the large ones (kick, 1,063 levels:
   max gap 0.048). Only `lama_adv` on credit-g is above the raw-column level (median gap +0.024).
3. Transductive OpenFE (repeat 0, 5 folds): the same columns as inductive in 50 of 50 units; differences from
   the transform only, up to +0.045 AUC (kick, LightGBM: +0.033 to +0.045 in every fold) and +0.041 R2
   (house_prices, CatBoost); 0 on mfeat-fourier (row-wise operators).
4. Planted signals (4 kinds x 3 seeds x 5 folds, oracle R2 0.76-0.80). The true feature is created on ratio
   and product by every arm except `ag_default` and `lama_adv`; on cat_interaction by `ag_tabprep`, `lama_adv`
   and `openfe` (2 of 3 units); on group_dev by `openfe` always and `manual_baseline` in 47%. Base CatBoost
   reaches the oracle R2 (within 1%) on ratio in 73% of folds, so reaching it is a weak signal there; on
   group_dev only `openfe` with LightGBM (73%) and `manual_baseline` (47-53%) reach it often; `openfe` with
   CatBoost reaches it in 20%, with the feature created in every unit.
5. OpenFE operators: 23 of 23 equal to the reference on fit rows.

**Estimate** (`results/tidy/exp04_estimate.md`): per-dataset interpolation of pilot times in rows x columns;
about 20 unit-hours per fold, 3 jobs: 34 h for 5 splits, 68 h for 10 (24 h / 58 h left after the pilot).
`autofeat` is a third of it and the least certain part.

**After A3 (Ilia)**
- 5 x 2 restored (plan v3 had cut it to 1 repeat; section 7.1): `full_r0`, commit, `full_r1`, commit;
  per-dataset test back to J = 10. Splits regenerated: repeat 0 identical in 64 of 64 files. Protocol and
  splits hashes changed, unit results did not; records moved to the new hashes by unit key, statuses per
  stage unchanged; 10 of 10 random moved units rerun from scratch matched bitwise.
- `openfe_transductive` extended to the 5 folds of repeat 0, run alongside `full_r0`.
- Removed records are kept in `results/stale/` (git-ignored) until the full run ends. A cleanup script
  had first moved live record directories as well; they were moved back, and every exp05 output and the
  estimate recomputed from them match the pre-incident files byte for byte; all record hashes match.
- `experiments/exp05_leakage_checks/README.md` defines "found" and "reach" for check 4.
- Run records are committed as `results/raw/<exp>/{gen,fit}.jsonl.gz` (`make pack`, 20 MB of JSON ->
  1.5 MB); per-unit JSON is the git-ignored working store (`make unpack` restores it byte for byte).

## 2026-11-01: Phase 3 done (arm wrappers, exp04 smoke) -> A2

**What was built**
- Plan v4 run-store changes: `code_hash` in gen and fit units, `arm` and `view` in fit units, `not_applicable`
  status (`applicable()`), `--retry-status`, RAM-budgeted jobs (3 on this laptop), stdout into the error log,
  first traceback kept in error text, SIGKILL outside the watchdog = `oom`, missing `out.json` = `failed`.
- Arms: `ag_default`, `ag_tabprep` (+ fit-only `ag_tabprep_keepcat`, new columns by provenance), `lama_adv`
  (exp03 step 4), `openfe` (inductive evaluator), `autofeat`, `mljar_golden`, `manual_baseline`.
- Contract tests rewritten to plan section 6 (synthetic frame + credit-g); test 7 in `test_openfe_equivalence.py`.
- `registry.save` recomputes categorical stats per variant; `make data` rerun: only the
  `Amazon_employee_access_int` row changed (`max_cat_levels` 7518 -> 0, `hicard` -> False).
- `smoke_report.py` -> `results/tidy/exp04_smoke.md`.

**Tests**: core 35 passed; autogluon 22; mljar 10; autofeat 10; openfe 12 (incl. test 7); lama 11 of 12
(test 6 fails on credit-g, see below). `make env-check` green.

**Smoke** (credit-g, cmc, Moneyball, fold 0, 9 arms, 2 models): 78 of 78 units ok. Slowest generator: autofeat
on credit-g, 72 s; slowest fit: CatBoost on the 1,981 `ag_tabprep` columns of Moneyball, 83 s, 0.84 GB.

**Deviations and surprises**
- `openfe` env pins `scikit-learn<1.6`: OpenFE 0.0.12 calls `mean_squared_error(squared=False)`, so every
  regression unit failed in the first smoke run. The contract frames are binary only, so tests did not catch it.
- `autofeat` env pins `scikit-learn<1.8` (`force_all_finite`). The wrapper drops constant columns: autofeat
  standardizes every column and a zero std turns its selection input into NaN.
- LightAutoML 0.4.2 bug under pandas 2: `OrdinalEncoder.fit` reads the `count` column of
  `value_counts().reset_index()` instead of the values (`transformers/categorical.py:1053`), so every string
  categorical is NaN in `LGBSimpleFeatures` and selection drops them all. `lama_adv` on credit-g keeps 4
  features; test AUC 0.66 vs 0.79 for `base`. A plain `TabularAutoML` run shows the same selection. The same
  encoder feeds the reader's role scores, so `exp03_roles.csv` was computed with the bug.
- `ag_default`: unseen values of a two-valued column map to 0, not NaN (AutoGluon's bool mapping).
- `manual_baseline`: the key rule (categorical with the highest MI) picks near-unique columns, because
  discrete MI grows with the number of levels (`uid` on the synthetic frame; credit-g picks `checking_status`).

**A2 decisions (Ilia), applied after commit bfbedb1**
1. `lama` env pins `pandas<2` (1.5.3), where LAMA's own code path is correct. `lama_adv` passes 12 of 12
   contract tests; credit-g keeps 16 selected features with OOF encodings of category intersections. exp03
   rerun: the encoding subtype of 10 categorical columns changed in `exp03_roles.csv`; the numeric ->
   category changes and dropped columns did not. The bug itself is a finding for post 3.
2. `manual_baseline` key candidates: categoricals with at most 50% unique values on fit rows.
3. Contract tests also run on Moneyball (regression).
4. The 9 Phase 2 `base` records in the old unit format were removed in bfbedb1.

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
