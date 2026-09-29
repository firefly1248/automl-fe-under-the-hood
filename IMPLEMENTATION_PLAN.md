# Implementation plan: `automl-fe-under-the-hood` companion repo (v2, after review)

Owner: Ilia (firefly1248). Implementer: coding agent on a laptop.
Purpose: one repo that holds every experiment and every number used in the LinkedIn series
"What automated feature engineering really does under the hood". Every figure in a post must be
reproducible from this repo with one command.

Read the whole plan before writing code. Sections 0 and 1 are hard rules and starting facts.

---

## 0. Hard rules

1. **Only open data and open code.** Datasets come from OpenML or are generated in the repo.
   No company data, no results that did not come out of this repo's own runs, no references to
   unpublished work. Every external claim in a README cites a public link (paper, docs, source file).
2. **Every published number has a source.** Each number in a post is listed in
   `results/claims.yaml` with script, result file and query (section 9). No number is typed by hand.
3. **Inductive by default.** A generator sees only the rows it is allowed to fit on (section 7.2).
   Test rows are transformed with statistics fitted on train. A tool that needs test rows runs in a
   separate "transductive" arm, never in the main ranking.
4. **Failures are results.** Crash, timeout, memory kill: written as a record with the error text.
   Never dropped silently. Aggregation rule for failures is in section 7.6.
5. **Resumable.** Any run can be killed and restarted with the same command; finished units are
   skipped (section 5).
6. **Pinned environments.** One lockfile per tool env. No unpinned installs.
7. **Privacy of the machine.** Records never store the hostname or user name. `$HOME` is replaced
   with `~` in every stored error text, log and `ENVIRONMENT.md`. Store CPU model, core count,
   RAM and OS only.
8. **Repo stays private** until Phase 6 review by Ilia. Push to a private remote after each phase.
9. **Approval points** (the only ones; wait for Ilia at each):
   - A1: after Phase 1 probes (exp01, exp03 steps 1-2) and the dataset table `results/datasets.md`;
   - A2: after the exp04 smoke run (protocol check);
   - A3: after the exp04 pilot, with the full-run time estimate and any fallback you propose;
   - A4: before exp06 (optional experiment) and before any change of a section 7 parameter or arm list.

---

## 1. Starting facts (from reading autogluon 1.6.3 wheels; exp01 must re-confirm by script)

Nothing in this section may go into a post until exp01 reproduces it inside the repo.

### 1.1 AutoGluon default preprocessing
`AutoMLPipelineFeatureGenerator` (applied before every model; models may add their own
`_preprocess` on top) consists of: pre steps (`FillNa`, `AsType` type enforcement, drop useless
constant columns), a generator group (`IdentityFeatureGenerator` for int/float,
`CategoryFeatureGenerator` with `minimum_cat_count=2`, `DatetimeFeatureGenerator` giving the
timestamp plus `year, month, day, dayofweek`, `TextSpecialFeatureGenerator`,
`TextNgramFeatureGenerator` with CountVectorizer `min_df=30, ngram_range=(1,3), max_features=10000`,
vision identity / IsNan), post steps (`DropUnique`, drop duplicates). It builds no arithmetic,
interaction or target-encoding features.

### 1.2 TabPrep-style generators exist, but only behind `GBM_PREP`
Generators in `autogluon.features.generators`:
- `ArithmeticFeatureGenerator` (`arithmetic/preprocessor.py`): `selection_method="random"`,
  `max_order=3`, `max_base_feats=150`, `max_new_feats=2000`, `interaction_types=["/","*","-","+"]`,
  `corr_threshold=0.95`, `min_cardinality=3`, `nan_threshold=0.95`, `mode_imbalance_threshold=0.9`,
  `cat_as_num=False`, `subsample=100000` (marked TODO in source, not implemented). No model-based
  selection: random choice of base features plus a correlation filter.
- `CategoricalInteractionFeatureGenerator` (`cat_int.py`): `max_order=3`, `max_new_feats=100`, `min_count=2`.
- `OOFTargetEncodingFeatureGenerator` (`oof_target_encoder.py`): `n_splits=5`, `alpha=10`,
  `keep_original=False`. Out-of-fold values come only from `fit_transform`; `transform` on the
  training rows gives in-sample encodings.
- `GroupByFeatureGenerator` (`groupby.py`): `max_features=500`, mean and pct_rank, relative ratio.
- `RandomSubsetFeatureCompressionGenerator` (`rsfc.py`), plus `frequency.py`, `binned.py`, `cat_as_num.py`.

They are used only by model key `GBM_PREP` (`PrepLGBModel`, `models/tabprep/prep_mixin.py`) through
`ag.prep_params`. The typical config (`_r13`) is
`[[[Arithmetic], [CatInteraction(passthrough=True), OOFTE]]]`: one group with **two parallel
branches whose outputs are concatenated**. Arithmetic does not feed CatInteraction or TE; only
CatInteraction feeds OOF-TE. The group is wrapped in a `BulkFeatureGenerator` with
`remove_unused_features="false_recursive"`, `post_drop_duplicates=True`, `passthrough=True` and
`passthrough_types={"invalid_raw_types": ["category","object"]}`. Together with `keep_original=False`
this means raw categoricals are replaced by their target encodings; LightGBM inside `GBM_PREP`
never sees native categoricals.

The 14 `GBM_PREP` configs in the cpu 2025-12-18 portfolio differ: 7 use both branches;
`_r47, _r27, _r37` are Arithmetic only; `_r1, _r19, _r14` are CatInteraction + TE only;
`_r21` has no prep (empty list).

### 1.3 Which presets reach `GBM_PREP`
- `GBM_PREP` is only in portfolios `zeroshot_2025_12_18_cpu` and `zeroshot_2025_12_18_gpu`.
- Presets using them: `best_quality_v150`, `high_quality_v150` (cpu portfolio),
  `extreme_quality_v150` (gpu portfolio). Aliases: `best_v150`, `high_v150`, `extreme_v150`.
- `best_quality` and `high_quality` use `"zeroshot"` = the 2023 portfolio (no `GBM_PREP`).
- `extreme_quality` uses `commercial_2026_08_05` (no `GBM_PREP`).
- The default preset is `medium_quality` (`hyperparameters="default"`, no `GBM_PREP`).
- `best_quality_v150` sets `num_stack_levels=0` and an early-stopping callback; `best_quality`
  stacks with `dynamic_stacking="auto"`.

Hypothesis to test: with default presets, AutoGluon builds almost no features; its gains come from
model diversity, bagging and stacking.

### 1.4 Preliminary probe (sandbox, one run, NOT for posts until exp01 reproduces it)
Input 9 columns gave 11 columns. All-unique string id dropped; duplicate numeric dropped;
int-coded category stayed int64; integer date `20210530` stayed int64 (not detected as a date);
ISO string date became int64 plus year/month/day/dayofweek; free text with 4 distinct values
became category; a two-valued column came out as a non-category dtype (to confirm).

### 1.5 LightAutoML 0.4.2 (not yet run; verify from source in exp03)
- `Reader` guesses roles with Normalized Gini over several encodings per column plus rules
  (`lightautoml/reader/guess_roles.py`); a numeric column can become categorical.
- `LGBSimpleFeatures` vs `LGBAdvancedPipeline`: frequency encoding, OOF target encoding for high
  cardinality, categorical intersections, datetime differences and seasonality, optional group-by
  (`use_groupby`, off by default).
- To find out: which pipeline `TabularAutoML` uses per model by default, and the default feature
  selection mode.

### 1.6 Other tools (verified from wheels by the reviewer; re-check in wrappers)
- OpenFE 0.0.12: `transform(X_train, X_test, features)` concatenates train and test. Operators that
  need column-wide statistics: `freq`, `Combine` (codes factorized on the concatenation),
  `CombineThenFreq`, `GroupByThen{Min,Max,Mean,Median,Std,Rank,Freq,NUnique}`. Row-wise operators:
  `abs, log, sqrt, square, sigmoid, round, residual, min, max, +, -, *, /`. Selection uses LightGBM.
  `tmp_save_path` defaults to `./openfe_tmp_data.feather`; `_trans` calls `exit()` on error.
- autofeat 2.1.3: returns original plus new columns with sanitized names; needs `numpy<2`,
  Python < 3.13; has `max_gb`; accepts `categorical_cols`. Selection with L1 linear models.
- mljar-supervised 1.3.2: `supervised.preprocessing.goldenfeatures_transformer.GoldenFeaturesTransformer(results_path, ml_task, features_count, n_jobs)`
  has standalone `fit(X, y)` / `transform(X)`. Ops: diff, ratio both ways, sum, multiply; numeric
  columns only; scored with a depth-3 decision tree; keeps `min(100, max(10, 0.1*n_cols))` features;
  split seed hard-coded (`random_state=1`); unseeded `np.random.choice` above 250k pairs;
  **loads `results_path/golden_features.json` if it exists** (use a fresh temp dir per unit);
  `n_jobs` default -1.

---

## 2. Repo layout

```
automl-fe-under-the-hood/
  README.md  IMPLEMENTATION_PLAN.md  PROGRESS.md  LICENSE (MIT)  Makefile
  envs/{core,autogluon,lama,openfe,autofeat,mljar,h2o}/pyproject.toml + uv.lock
  src/afe/
    data/{registry.py, loaders.py, synthetic.py, splits.py}
    generators/{base.py, identity.py, ag_default.py, ag_tabprep.py, lama_pipeline.py,
                openfe_gen.py, autofeat_gen.py, mljar_golden.py, cheap_manual.py}
    models/downstream.py
    run/{unit.py, store.py, worker.py, limits.py, cli.py}
    analysis/{aggregate.py, stats.py, figures.py, claims.py}
  experiments/exp01_autogluon_probe/ ... exp06_automl_end_to_end/
    each: config.yaml, run.py, README.md (question, method, how to read, findings, caveats)
  results/
    raw/<exp>/gen/<gen_hash>.json      # generator-stage records
    raw/<exp>/fit/<fit_hash>.json      # downstream-stage records
    cache/<exp>/<gen_hash>/{train,inner,test}.parquet   # NOT committed (gitignored)
    splits/<dataset>.parquet           # committed fold assignments
    tidy/<exp>.parquet  figures/<exp>/  claims.yaml  datasets.md  ENVIRONMENT.md
  tests/
  posts/README.md                      # post -> experiments -> claim ids
```

Raw JSON records, tidy parquet, splits and figures are committed. Dataset downloads live in
`~/.cache/afe/`; generated-feature parquet cache is gitignored (can be rebuilt).

---

## 3. Environments

- `uv`, one project per folder in `envs/`, Python 3.11 everywhere. `src/afe` installed editable in
  each env. `base.py`, `data/`, `run/unit.py` import only numpy/pandas/stdlib.
- `core`: orchestrator, analysis, downstream LightGBM and CatBoost (pinned), `cheap_manual`, `base`.
  Downstream always runs in `core` so every arm is scored with identical model versions.
- Arm envs: `autogluon` (`autogluon.tabular==1.6.3`), `lama` (`lightautoml==0.4.2`),
  `openfe` (`openfe==0.0.12`, pin pandas/lightgbm that it works with), `autofeat`
  (`autofeat==2.1.3`, `numpy<2`), `mljar` (`mljar-supervised==1.3.2`), `h2o` (optional, Java 17).
- LightAutoML 0.4.2 hard deps include `torch`, `tabicl` (needs `torch>=2.2`, `scikit-learn>=1.3`),
  `tabm`, `xgboost>=2,<3`, `catboost`, `autowoe`, `optuna`, `SQLAlchemy>=2`,
  `statsmodels<=0.14.0` (forces `numpy<2`). Use CPU torch:
  ```toml
  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  [tool.uv.sources]
  torch = { index = "pytorch-cpu" }   # Linux; on macOS PyPI torch is already CPU
  ```
  Extra pins (`pandas<2.3`, `scikit-learn<1.6`) only as fallbacks if the resolver or imports fail.
  Note: a plain `pip install lightautoml` was OOM-killed on an 8 GB machine; CPU torch avoids that.
- macOS: `brew install libomp` for LightGBM.
- `make env-check` imports each tool in its env, writes versions, CPU, RAM, OS to `ENVIRONMENT.md`.

---

## 4. Data

### 4.1 Synthetic frames (`data/synthetic.py`)
`hard_columns(n, seed)` with a known generating process:
`num`, `dup_num` (copy), `const`, `near_const` (99.9% one value), `city_code` (int 0..39 acting as a
category, non-monotone effect), `date_int` (YYYYMMDD), `date_str` (ISO), `ts_unix`, `cat_low` (3
levels), `cat_rare` (3 frequent levels + 300 levels seen once or twice), `cat_binary`, `uid`,
`text_short` (4 phrases), `text_long` (~500-word vocabulary), `num_as_str`, `mixed` (numbers with "n/a").
Binary target from city effect + month of `date_int` + ratio `a/b` + a word in `text_long` + noise.
Store the formula in metadata.

`planted(kind, n, seed)` for exp05: target driven by a ratio, a product, deviation from a group
mean, or a category interaction; also returns the oracle feature.

### 4.2 Open benchmark datasets (`data/registry.py`)
- Pool: `openml.study.get_suite(99).data` (OpenML-CC18) and `get_suite(353).data` (OpenML-CTR23).
  Take the target from the suite's task, not from the dataset default.
- Filters: 1,000 <= rows <= 100,000; 4 <= columns <= 100; classification with <= 20 classes; no
  image/pixel data (name list: mnist_784, Fashion-MNIST, CIFAR_10, Devnagari-Script, plus any
  dataset where > 90% of columns match `pixel|^p\d+$`).
- Dedupe on (normalized name, n_rows, n_features, target name).
- Deterministic fill: sort remaining candidates by OpenML data id; fill quotas in that order:
  12 binary, 6 multiclass, 12 regression, then check that >= 8 of the selected have categorical
  columns; if not, replace the highest-id datasets without categoricals by the next candidates with
  categoricals (log every swap).
- Typed extra set (resolve by name through the API, keep only if target and size match):
  `adult`, `bank-marketing`, `credit-g`, `Amazon_employee_access` (plus a variant with its
  nominal columns cast to int, so the int-coded case really happens), `diamonds`,
  `Bike_Sharing_Demand` (check what date-like columns it really has; if none, say so).
- For exp01 step 4 and exp03 step 3 (Phase 1, before the registry is final) use `adult` and
  `diamonds`.
- Checksums: store the md5 of the file as downloaded from OpenML and
  `pd.util.hash_pandas_object(df, index=True).sum()` of the loaded frame.
- Folds: generated once by `splits.py`, saved to `results/splits/<dataset>.parquet`
  (columns: row id, repeat, outer fold, inner flag), committed.
- `results/datasets.md`: id, version, name, task, rows, cols, n categorical, checksums. Part of A1.

---

## 5. Run store and resumability

Two stages, so a generator runs once per fold and is reused by both downstream models.

- Generator unit: `(exp, dataset, repeat, fold, arm, arm_params)` plus
  `protocol_hash` (hash of the exp `config.yaml`), `data_hash`, `splits_hash`, `env_lock_hash`
  (hash of the arm's `uv.lock`). `gen_hash = sha1(json.dumps(unit, sort_keys=True))[:16]`.
  Output: cached parquet + record.
- Fit unit: `(gen_hash, downstream_model, model_params, seed)` -> `fit_hash`.
- Skip a unit if its record exists with status in `{ok, failed, timeout, oom}`; `--retry-failed`
  reruns non-ok. If a cache parquet is missing but the gen record is ok, regenerate it.
- Atomic writes: `*.tmp`, fsync, rename.
- Gen record: unit, status, error (<= 5 kB, `$HOME` stripped), n features in / new / final,
  n infinite values replaced, fit time, transform time (train, inner, test), peak RSS, arm env
  versions, git sha, machine info (rule 7), timestamps.
- Fit record: test metrics, inner-validation metric, best iteration, fit time, peak RSS.
- Limits (`run/limits.py`): generator wall clock 900 s; downstream wall clock 1,800 s; memory cap
  6 GB enforced by a psutil RSS watchdog that kills the child process tree (Linux and macOS; do not
  use `RLIMIT_AS`, it kills torch/numba/LightGBM early). On Linux `systemd-run --user --scope -p
  MemoryMax=6G` is an allowed alternative.
- Threads: in every worker set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, LightGBM `num_threads=1`, CatBoost `thread_count=1`, and `n_jobs=1`
  in every tool (OpenFE, GoldenFeaturesTransformer, autofeat, LAMA `cpu_limit=1`).
- Parallelism: `jobs = min(physical_cores - 1, floor((RAM_GB - 4) / 6))`, overridable by `--jobs`.
- Per-unit temp dir (OpenFE `tmp_save_path`, MLJAR `results_path`, LAMA/AG output paths), deleted
  after the unit.
- `make commit-results`: aggregate, git add raw + tidy + figures, commit with exp, stage, coverage.

---

## 6. Generator interface

```python
class FeatureGenerator:
    name: str
    env: str
    transductive: bool = False
    replaces_input: bool = False      # True: downstream uses only this output (e.g. ag_default)

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        """The ONLY source of features for the rows it was fitted on.
        Supervised encodings must be out-of-fold here."""
    def transform(self, X) -> pd.DataFrame:
        """For rows not used in fit (inner validation and test). Same columns as fit_transform."""
    def describe(self) -> dict:       # created features, ops used, wrapper decisions
```

Downstream input = `base_view(original) + new columns`, or the arm output alone if `replaces_input`.

`base_view` (applied identically in every arm, also to `base`): object/string -> `category`;
datetime -> int64 ns; bool -> int8; nothing else. LightGBM gets `category` dtype; CatBoost gets
the same columns as `cat_features` with NaN filled by the string `"__nan__"`.

Contract tests (`tests/test_generators_contract.py`, run on the synthetic frame and one small real
dataset for every arm):
1. Output index/rows equal input; unique names; numeric or category dtypes; inf replaced by NaN and counted.
2. Determinism: `fit_transform` twice with the same seed gives identical output.
3. Test independence: replacing test X and test y with random values leaves the fit-rows output unchanged.
4. Row locality: shuffling test rows and transforming gives the same per-row features.
5. Unseen categories in test do not crash and map to NaN (or the tool's documented default).
6. For arms with target encoding: fit-rows output differs from `transform(fit rows)` (proves OOF is used).

Arms, main ranking (all inductive):
| arm | env | what |
|---|---|---|
| `base` | core | `base_view` only |
| `ag_default` | autogluon | `AutoMLPipelineFeatureGenerator` output; `replaces_input=True` |
| `ag_tabprep` | autogluon | the exact `_r13` `ag.prep_params` group, built as AutoGluon builds it (section 1.2), applied after `AutoMLPipelineFeatureGenerator`. Output rule: **as AutoGluon does it** (categoricals replaced by TE), `replaces_input=True`. Record this in the README; it is the one arm that does not keep native categoricals |
| `ag_tabprep_groupby` | autogluon | same plus `GroupByFeatureGenerator`; only if it runs standalone. Cite the TabPrep paper (arXiv 2606.02384) as the source of this variant |
| `lama_adv` | lama | LAMA reader + advanced LGB feature pipeline extracted as a transformer (exp03.4); dropped if not reliable |
| `openfe` | openfe | inductive wrapper (below) |
| `autofeat` | autofeat | `feateng_steps=2`, `categorical_cols` passed, `max_gb` set, NaN imputed by the wrapper with train medians (recorded); originals stripped, names mapped back |
| `mljar_golden` | mljar | `GoldenFeaturesTransformer`, fresh `results_path` per unit, `n_jobs=1` |
| `cheap_manual` | core | exact rules below |

Separate arm: `openfe_transductive` (library default `transform(train, test)`), only in exp05.

OpenFE inductive wrapper: reimplement the column-wide operators (`freq`, `Combine`,
`CombineThenFreq`, `GroupByThen*`) with statistics fitted on fit rows only and mapped to other rows
(unseen keys -> NaN); row-wise operators are applied as is. If reimplementation is not done in
reasonable time, restrict the inductive arm to row-wise operators and say so in the README.

`cheap_manual` rules (all statistics on fit rows only):
- frequency encoding of every categorical column (share of rows);
- top-5 numeric columns by mutual information with y (`mutual_info_*`, `random_state=seed`);
  for each pair: `a/b` and `b/a` with denominator `|b| < 1e-12` -> NaN, and `a*b`;
- key = the categorical column with the highest mutual information with y (if none: decile bins of
  the top-MI numeric column); for each top-5 numeric: group mean and `x - group_mean`
  (unseen key -> global mean).

---

## 7. Benchmark protocol (exp04; confirmed at A2)

### 7.1 Folds
5-fold outer CV (stratified for classification, plain KFold for regression), 2 repeats. Seed of a
repeat r: `1000 + r`; tool seed for (repeat r, fold k): `1000 + 10*r + k`.

### 7.2 Who sees what
Inside each outer train, split off 15% as inner validation first (stratified for classification).
Generators fit on outer train minus inner (`fit_transform`), then `transform` inner and test.
Downstream trains on the fit rows, early-stops on inner, is scored on test. No refit on full train.

### 7.3 Downstream models (both always)
- LightGBM: `n_estimators=5000, learning_rate=0.05`, early stopping 200 rounds; other params default.
- CatBoost: `iterations=5000, learning_rate=0.05`, early stopping 200 rounds; other params default.
- Early-stopping metric: log loss (classification), RMSE (regression).
Why two models: the tools select features with different models (OpenFE: LightGBM; MLJAR: depth-3
tree; autofeat: L1 linear; AutoGluon Arithmetic: random + correlation filter). Two downstream
families show whether gains carry over.

### 7.4 Metrics
- Binary: ROC AUC and log loss (both co-primary).
- Multiclass: log loss (primary), accuracy.
- Regression: RMSE (primary), R2.
- Also: n new features, generator time, total time.

### 7.5 Comparison to `base` (same model, same folds)
Per dataset, per arm and model, on the 10 folds:
- relative improvement for losses on per-dataset means: `(base - arm) / base`;
- AUC: gap closed on per-dataset means `(arm - base) / (1 - base)`; datasets with base AUC >= 0.995
  are excluded from this ratio and listed;
- paired test: corrected repeated k-fold t-test (Nadeau-Bengio, Bouckaert-Frank form: J = 10,
  correction n_test/n_train with n_train = outer train size, df = 9). If all fold differences are 0
  (for example the arm created no features), record a tie with p = 1;
- win / tie / loss at alpha 0.05 with Holm across arms within the dataset. This is low-power by
  design; say so in every report.
Across datasets:
- median relative improvement with bootstrap 95% CI over datasets (10,000 resamples);
- Wilcoxon signed-rank on per-dataset means, Holm-corrected across arms;
- share of datasets with a significant win and with a significant loss.
Raw metric differences are reported per dataset only (units differ across datasets).
Primary conclusions are the across-dataset numbers. Report by task type and for datasets with categoricals.

### 7.6 Failures in aggregates
- Primary: a failed or timed-out unit takes the `base` score for that fold (what a user gets with no
  new features).
- Secondary: complete cases only.
- Coverage (ok / planned) shown next to both, per arm.

### 7.7 Cost
Median and p90 of generator time, and of `(generator + downstream) / base downstream` time.
Timing only from runs on the same machine, idle, laptop on power.

### 7.8 Runtime and fallbacks (decided at A3)
Rough full-run cost: about 36 datasets x 10 folds x 8 arms, generator stage cached. Estimated
300-500 CPU-hours (CatBoost single-threaded and OpenFE/autofeat near the cap dominate), i.e. about
2-4 days of wall time at 4-7 jobs; worst case about a week. The pilot prints a measured estimate.
Fallbacks to propose at A3 if needed, in this order: CatBoost `learning_rate=0.1`; row cap 50,000
(stratified subsample, recorded); 1 repeat for OpenFE and autofeat; drop `ag_tabprep_groupby`.
Keep the laptop awake (`caffeinate -i` on macOS, `systemd-inhibit` on Linux).

---

## 8. Experiments

### exp01_autogluon_probe (post 2) — env autogluon, minutes
1. `AutoMLPipelineFeatureGenerator` on `hard_columns()`, 3 seeds, verbosity 3. Save
   `input column -> output columns -> dtype -> action` to `results/tidy/exp01_types.csv` and
   `feature_metadata`.
2. Variants: `cat_rare` with singletons only; `text_long` at n = 500 and 5,000 (when n-grams appear).
3. `inventory.py`: all generator classes with `inspect.signature` defaults; for every preset and
   alias, the resolved portfolio, whether it contains `GBM_PREP`, and every `ag.prep_params` chain
   with its suffix. Writes `exp01_presets.csv`, `exp01_prep_configs.csv`. If output differs from
   section 1, the script wins and section 1 is corrected.
4. `ArithmeticFeatureGenerator` and the full `_r13` group on `hard_columns()`, `adult`, `diamonds`:
   candidates built, kept after the correlation filter, time; does it recover the planted `a/b`?

### exp02_autogluon_ablation (post 2) — env autogluon, many hours, runs alone (uses all cores)
12 datasets (mixed tasks, half with categoricals), outer folds of repeat 0 (5 folds).
- A: `GBM` with the `_r13` hyperparameters and no `ag.prep_params`, no bagging;
- B: `GBM_PREP` `_r13`, no bagging (B - A = effect of feature construction only);
- B2: `GBM_PREP` `_r21` (portfolio's own no-prep control), for reference;
- C: `medium_quality`;
- D: `best_quality` (2023 portfolio, stacking);
- D0: `best_quality` with `num_stack_levels=0` (separates stacking from portfolio);
- E: `best_quality_v150` (2025-12-18 portfolio with `GBM_PREP`, no stacking, early-stop callback).
Contrasts: B - A (features), D - A (models + ensembling), D - D0 (stacking), E - D0 (new portfolio).
For C, D, D0, E use a fixed `time_limit=600` and record the trained model list, fit time and the
machine; state that these arms depend on CPU speed. Record resolved hyperparameters from the predictor.

### exp03_lama_probe (post 3) — env lama
1. Default config of `TabularAutoML` for 0.4.2: feature pipeline per model, `auto_unique_co`,
   intersection settings, `use_groupby`, selection mode. Save `exp03_config.yaml`.
2. Reader alone on `hard_columns()` and the typed set: `column -> dtype -> LAMA role -> reason`
   (`exp03_roles.csv`). Highlight numeric -> categorical and the reverse.
3. `TabularAutoML(task, timeout=600, cpu_limit=...)` on `adult`, `diamonds`, `Amazon_employee_access`:
   features each model received, selected/dropped features, OOF score. Compare simple vs advanced
   pipeline if reachable via config.
4. Try to extract reader + advanced pipeline as a `FeatureGenerator` (`lama_adv`). If the internal
   API makes it unreliable, document the attempt, keep the code, drop the arm from exp04.

### exp04_open_benchmark (posts 4 and 5)
Stages: `smoke` (3 datasets, 1 fold, all arms, both models, contract checks) -> A2 -> `pilot`
(8 datasets, 10 folds, time estimate) -> A3 -> `full`.
Outputs: `results/tidy/exp04.parquet`, tables by task type, figures, README with findings and caveats.

### exp05_leakage_checks (posts 4 and 5) — after the pilot, before or alongside full
1. Permuted labels (6 datasets): permute y on fit rows. Check that the downstream inner-validation
   metric and the train-rows metric stay at chance level for every arm; a generator that lifts the
   fit-rows metric under permuted labels encodes y in-sample.
2. TE delivery check: for arms with target encoding, AUC (or |corr| for regression) of each TE
   column with y on fit rows as delivered vs on test rows. Large gap = in-sample encoding.
3. `openfe` vs `openfe_transductive` on the same folds.
4. Planted signals (`planted()` x 4 kinds x 3 seeds): does each arm reach the oracle's metric, and is
   the true feature (or an equivalent) among the created ones.

### exp06_automl_end_to_end (post 6, optional, A4)
AutoGluon `best_quality` vs LightAutoML `TabularAutoML` vs H2O AutoML with and without
`preprocessing=["target_encoding"]`, 10 datasets, 600 s each, same folds.

---

## 9. Analysis, figures, claims

- `make aggregate` builds tidy parquet with coverage columns.
- Figures (matplotlib, SVG + PNG, fixed colour per arm across all figures, legible at 1200x1500
  and 1200x628): exp01 type-mapping table; exp02 contrasts with CIs; exp03 role changes; exp04
  median relative improvement with CI by task type and model, win/tie/loss bars, improvement vs
  generator time (log x); exp05 permuted-label results and planted-signal grid.
- `results/claims.yaml`:
  ```yaml
  - id: p4_openfe_reg_median_lgbm
    post: 4
    text: "median relative RMSE improvement of OpenFE on regression, LightGBM downstream"
    source: results/tidy/exp04.parquet
    query: "task=='regression' & arm=='openfe' & model=='lgbm'"
    stat: median_rel_improvement
    failure_rule: base_fallback
    value: null        # filled by `make claims`
    ci: null
    coverage: null
    approved_value: null   # set ONLY by Ilia
    approved_git_sha: null
    tolerance: 0.001
  ```
  `make claims` recomputes values and fails if any value differs from `approved_value` by more than
  `tolerance`. The agent never writes `approved_*` fields.

---

## 10. Makefile targets

```
make env env-check data test
make exp01 exp02 exp03
make exp04-smoke exp04-pilot exp04-full
make exp05 exp06
make aggregate figures claims report commit-results
```

---

## 11. Phases

| phase | work | done when |
|---|---|---|
| 0 | skeleton, envs, `env-check` | all envs import; versions in `ENVIRONMENT.md`; LAMA recipe documented |
| 1 | exp01, exp03 steps 1-2, dataset registry + `datasets.md` | tables saved, section 1 confirmed or corrected -> **A1** |
| 2 | splits, store, worker, limits, contract tests | `make test` green; kill-and-resume test (kill mid-run, rerun: no duplicates, no lost units) |
| 3 | all arm wrappers, exp04 smoke | smoke report -> **A2** |
| 4 | exp04 pilot, exp05 | time estimate, leakage report -> **A3**, then exp04 full; coverage >= 95% or failures explained per arm |
| 5 | exp02, exp03 steps 3-4 | READMEs with findings |
| 6 | aggregate, figures, claims, README | `make claims` passes; fresh clone reproduces exp01 and exp04 smoke |

After each phase: commit, push to the private remote, add a `PROGRESS.md` entry (date, what ran,
coverage, surprises, open questions). This file is how Ilia and later sessions resume the work.

---

## 12. Likely problems

- OpenFE may break on new pandas: pin in its env. Catch `exit()` in `_trans` via the worker timeout.
- autofeat: NaN and memory; impute and set `max_gb`, or record the failure.
- AutoGluon generators need AutoGluon feature metadata: run `AutoMLPipelineFeatureGenerator` first
  inside `ag_tabprep`.
- OOF-TE with many classes: one column per class; the 20-class filter keeps this bounded.
- `ag_tabprep` can add up to ~2,000 columns; the downstream cap (1,800 s) turns slow fits into
  timeout records. Always show n features next to cost.
- Laptop timing noise: idle machine, on power, medians only, one machine.

---

## 13. What to hand back to Ilia

- A1: exp01 and exp03 tables, corrections to section 1, `datasets.md`.
- A2: smoke report (per arm: ok/failed, n features, times, contract test results).
- A3: pilot results, leakage report, measured full-run estimate, proposed fallbacks.
- End: repo link, `claims.yaml` with values, figures, findings per post in plain words with claim ids.
