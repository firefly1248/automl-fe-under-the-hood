# Implementation plan: `automl-fe-under-the-hood` companion repo (v3, after A1 and the second review)

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
  `cat_as_num=False`, `subsample=100000` (row subsample; implemented despite a TODO comment). No
  model-based selection and, with the default `selection_method="random"`, no correlation filter:
  `corr_threshold` is used only by `selection_method="spearman"` (corrected at A1 from exp01).
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
                openfe_gen.py, autofeat_gen.py, mljar_golden.py, manual_baseline.py}
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
- `core`: orchestrator, analysis, downstream LightGBM and CatBoost (pinned), `manual_baseline`, `base`.
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
- Pool (changed after the plan review): OpenML suites 99 (OpenML-CC18), 353 (OpenML-CTR23),
  271 (AutoML Benchmark classification), 269 (AutoML Benchmark regression), 334 and 335 (tabular
  benchmark, categorical classification and regression, Grinsztajn et al. 2022). Take the target from
  the suite's task, not from the dataset default. Note: suites 334/335 had high-cardinality
  categoricals removed by their authors; they add datasets with categoricals, not high cardinality.
- Filters: 1,000 <= rows <= 100,000; 4 <= columns <= 100; classification with <= 20 classes; no
  image/pixel data (name list: mnist_784, Fashion-MNIST, CIFAR_10, Devnagari-Script, optdigits, plus
  any dataset where > 90% of columns match `pixel|^p\d+$`); exact duplicate rows <= 5%; manual
  exclusions with a reason (electricity: temporal order, random CV over-rewards date features).
- Dedupe on (normalized name, n_rows, n_features, target name).
- Families: datasets that are views or versions of the same data share a `family` (mfeat-*; NASA
  MDP pc/kc/jm/mc/mw; Amazon and its int variant; house_sales and kings_county; satimage and
  Satellite; otherwise the normalized name). Same name, different dataset: key `<name>_<did>`. One dataset per
  family goes into the primary aggregate (one with a >= 10-level categorical first, then lowest id);
  the others are logged. Typed extras of an already present family are case studies, not primary.
- Deterministic fill of quotas: 20 binary, 10 multiclass, 20 regression. Within a task, datasets with
  a categorical column of >= 10 levels come first, then the rest, each in OpenML data id order.
  With this pool every candidate with such a column is selected: 17 of 50 (34%). Decided after the
  plan review: keep 50 datasets rather than shrink the set to reach 50%; claims about categoricals
  are made on that subgroup (17 vs 33).
- Per-task-type breakdowns are reported only for task types with >= 8 primary datasets.
- Typed extra set (resolve by name through the API, keep only if target and size match):
  `adult` (pinned to CC18 did 1590; the name resolves to did 179, a binned version), `bank-marketing`,
  `credit-g`, `Amazon_employee_access` (plus a variant with its nominal columns cast to int, so the
  int-coded case really happens), `diamonds`, `house_sales` (did 42092, real date string; replaced
  `Bike_Sharing_Demand` at A1, which has no date-typed column).
- For exp01 step 4 and exp03 step 3 (Phase 1, before the registry is final) use `adult` and
  `diamonds`.
- Checksums: store the md5 of the file as downloaded from OpenML and
  `pd.util.hash_pandas_object(df, index=True).sum()` of the loaded frame.
- Folds: generated once by `splits.py`, saved to `results/splits/<dataset>.parquet`
  (columns: row id, repeat, outer fold, inner flag), committed.
- `results/datasets.md`: id, version, name, family, primary flag, task, rows, cols, n categorical,
  max categorical levels, duplicate-row share, checksums. Part of A1.

---

## 5. Run store and resumability

Two stages, so a generator runs once per fold and is reused by both downstream models.

- Generator unit: `(exp, dataset, repeat, fold, arm, arm_params)` plus
  `protocol_hash` (hash of the exp `config.yaml`), `data_hash`, `splits_hash`, `env_lock_hash`
  (hash of the arm's `uv.lock`). `gen_hash = sha1(json.dumps(unit, sort_keys=True))[:16]`.
  Output: cached parquet + record.
- Fit unit: `(gen_hash, downstream_model, model_params, seed, core_lock_hash)` -> `fit_hash`
  (`core_lock_hash`: hash of `envs/core/uv.lock`, so a LightGBM/CatBoost bump invalidates fits).
- Skip a unit if its record exists with status in `{ok, failed, timeout, oom}`; `--retry-failed`
  reruns non-ok. If a cache parquet is missing but the gen record is ok, regenerate it.
- Atomic writes: `*.tmp`, fsync, rename.
- Gen record: unit, status, error (<= 5 kB, `$HOME` stripped), n features in / new / final,
  n infinite values replaced, fit time, transform time (train, inner, test), peak RSS, arm env
  versions, git sha, machine info (rule 7), timestamps.
- Fit record: test metrics, inner-validation metric, best iteration, fit time, peak RSS.
- Limits (`run/limits.py`): generator wall clock 3,600 s in smoke and pilot; the full-run cap is set
  at A3 from the pilot's measured p95 per arm, so expensive tools are not starved into timeouts;
  downstream wall clock 1,800 s; memory cap
  6 GB enforced by a psutil RSS watchdog that kills the child process tree (Linux and macOS; do not
  use `RLIMIT_AS`, it kills torch/numba/LightGBM early). On Linux `systemd-run --user --scope -p
  MemoryMax=6G` is an allowed alternative.
- Threads: in every worker set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, LightGBM `num_threads=1`, CatBoost `thread_count=1`, and `n_jobs=1`
  in every tool (OpenFE, GoldenFeaturesTransformer, autofeat, LAMA `cpu_limit=1`).
- Parallelism: `jobs = min(physical_cores - 2, 10)`, overridable by `--jobs`. Memory is guarded at
  launch instead of in the formula: a unit starts only if free RAM >= `min_free_gb` (2 GB; a 6 GB
  gate stalled the queue on this laptop, where other apps leave ~7 GB free); the per-unit 6 GB
  watchdog stays. Close other apps for the full run. (The old formula gave 3 jobs on a 24 GB laptop.)
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
7. `openfe` only: the reimplemented column-wide operators reproduce OpenFE's own values from
   `transform(X_fit, X_fit, features)` on the fit rows (equivalence test, per operator). Without it
   the inductive arm can be dismissed as "not OpenFE".

Arms, main ranking (all inductive):
| arm | env | what |
|---|---|---|
| `base` | core | `base_view` only |
| `ag_default` | autogluon | `AutoMLPipelineFeatureGenerator` output; `replaces_input=True` |
| `ag_tabprep` | autogluon | the exact `_r13` `ag.prep_params` group, built as AutoGluon builds it (section 1.2), applied after `AutoMLPipelineFeatureGenerator`. Output rule: **as AutoGluon does it** (categoricals replaced by TE), `replaces_input=True`. Record this in the README; it is the one arm that does not keep native categoricals. In posts call it "the `_r13` configuration", not "TabPrep", and say that most of its features are random, unfiltered arithmetic (exp01: 894 of 894 on `hard_columns`, 934 on diamonds) that the portfolio's own tuned LightGBM config, not a default LightGBM, was built to absorb |
| `ag_tabprep_keepcat` | autogluon (fit stage only) | `base_view(original)` + the `ag_tabprep` group's new columns, native categoricals kept. Reuses the cached `ag_tabprep` generator output (no extra generator run); separates "new features" from "losing native categoricals", which matters most for CatBoost |
| `lama_adv` | lama | LAMA reader + advanced LGB feature pipeline extracted as a transformer (exp03.4); dropped if not reliable |
| `openfe` | openfe | inductive wrapper (below) |
| `autofeat` | autofeat | `feateng_steps=2`, `categorical_cols` passed, `max_gb` set, NaN imputed by the wrapper with train medians (recorded); originals stripped, names mapped back |
| `mljar_golden` | mljar | `GoldenFeaturesTransformer`, fresh `results_path` per unit, `n_jobs=1` |
| `manual_baseline` | core | exact rules below |

Separate arm: `openfe_transductive` (library default `transform(train, test)`), only in exp05.

OpenFE inductive wrapper: reimplement the column-wide operators (`freq`, `Combine`,
`CombineThenFreq`, `GroupByThen*`) with statistics fitted on fit rows only and mapped to other rows
(unseen keys -> NaN); row-wise operators are applied as is. If reimplementation is not done in
reasonable time, restrict the inductive arm to row-wise operators and say so in the README.

`manual_baseline` rules (all statistics on fit rows only):
- frequency encoding of every categorical column (share of rows);
- top-5 numeric columns by mutual information with y (`mutual_info_*`, `random_state=seed`);
  for each pair: `a/b` and `b/a` with denominator `|b| < 1e-12` -> NaN, and `a*b`;
- key = the categorical column with the highest mutual information with y (if none: decile bins of
  the top-MI numeric column); for each top-5 numeric: group mean and `x - group_mean`
  (unseen key -> global mean).

---

## 7. Benchmark protocol (exp04; confirmed at A2)

### 7.1 Folds
5-fold outer CV (stratified for classification, plain KFold for regression), 1 repeat (r = 0).
Across-dataset power grows with the number of datasets, not repeats, so the budget goes to ~50
datasets instead (section 4.2). Seed of a repeat r: `1000 + r`; tool seed for (repeat r, fold k):
`1000 + 10*r + k`.

### 7.2 Who sees what
Inside each outer train, split off 15% as inner validation first (stratified for classification).
Generators fit on outer train minus inner (`fit_transform`), then `transform` inner and test.
Downstream trains on the fit rows, early-stops on inner, is scored on test. No refit on full train.

### 7.3 Downstream models (both always)
- LightGBM: `n_estimators=5000, learning_rate=0.05`, early stopping 200 rounds; other params default.
- CatBoost: `iterations=3000, learning_rate=0.1`, early stopping 200 rounds; other params default.
- Early-stopping metric: log loss (classification), RMSE (regression).
Why two models: the tools select features with different models (OpenFE: LightGBM; MLJAR: depth-3
tree; autofeat: L1 linear; AutoGluon Arithmetic: random choice, no filter). Two downstream
families show whether gains carry over.
Design choice, stated up front in every report and post: downstream models are untuned defaults and
the same for every arm. The question is "plug the tool's features into a standard GBM, no retune".
Default LightGBM (`feature_fraction=1`, `min_child_samples=20`) with ~1,000 random features on small
data can overfit; that is part of what is measured, not a rigged baseline.

### 7.4 Metrics
- Binary: ROC AUC and log loss (both co-primary).
- Multiclass: log loss (primary), accuracy.
- Regression: RMSE (primary), R2.
- Also: n new features, generator time, total time.

### 7.5 Comparison to `base` (same model, same folds)
Per dataset, per arm and model, on the 5 folds:
- AUC: paired raw difference of per-dataset means `arm - base` (AUC is already unit-free; the
  gap-closed ratio is dropped because it explodes near AUC 1);
- losses (log loss, RMSE): relative improvement on per-dataset means `(base - arm) / base`; for log
  loss also the raw difference, since datasets with log loss near 0 dominate relative numbers;
- descriptive win / tie / loss: corrected k-fold t-test (Nadeau-Bengio, J = 5, correction
  n_test/n_train with n_train = outer train size, df = 4) at alpha 0.05, no multiplicity correction
  within a dataset. If all fold differences are 0 (e.g. the arm created no features), record a tie
  with p = 1. Per-dataset tests are near-powerless; they are descriptive only, say so.
Across datasets (primary conclusions; primary set = one dataset per family, section 4.2):
- median effect (AUC difference or relative improvement) with bootstrap 95% CI over datasets
  (10,000 resamples);
- Wilcoxon signed-rank on per-dataset means, Holm-corrected across arms;
- share of datasets with a descriptive win and with a descriptive loss.
Raw metric differences other than AUC and log loss are reported per dataset only (units differ).
Breakdowns: by task type only where >= 8 primary datasets exist; with vs without a categorical
column of >= 10 levels. Case-study datasets (not primary) are reported separately.

### 7.6 Failures in aggregates
Two rules, reported side by side with equal weight:
- base fallback: a failed or timed-out unit takes the `base` score for that fold ("what a user gets");
- complete cases only ("what the tool does when it runs").
Coverage (ok / planned) is shown next to both, per arm. A claim that holds under one rule only says so.

### 7.7 Cost
Median and p90 of generator time, and of `(generator + downstream) / base downstream` time.
Timing claims come from a separate timing pass with `jobs=1` on ~6 primary datasets x 1 fold (all
arms), machine idle and on power; the main run is parallel and never idle, so its times are for
budgeting only.

### 7.8 Runtime and fallbacks (decided at A3)
Rough full-run cost: about 50 datasets x 5 folds x 9 arms (7 generator arms, generator stage
cached) x 2 models. Estimated 200-400 CPU-hours (OpenFE/autofeat generators and CatBoost on wide
outputs dominate), i.e. about 1-2 days of wall time at 10 jobs. The pilot prints a measured estimate.
Fallbacks to propose at A3 if needed, in this order: row cap 50,000 (stratified subsample,
recorded); full-run generator cap from the pilot p95 (never below the time the tool needs on the
median dataset); drop the slowest arm on datasets it cannot finish, reported as coverage.
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
Caveat for the README and post: with `time_limit=600` on this laptop, D and E will not finish their
portfolios on the larger datasets; D - A and D - D0 measure "what fits in 600 s on 14 cores", not the
full presets. The per-fold trained-model list shows how much of each portfolio ran.

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
(8 primary datasets, 5 folds, time estimate) -> A3 -> `full`; plus the `jobs=1` timing pass (7.7).
Outputs: `results/tidy/exp04.parquet`, tables by task type, figures, README with findings and caveats.

### exp05_leakage_checks (posts 4 and 5) — after the pilot, before or alongside full
1. Permuted labels (6 datasets): permute y on fit rows. The downstream inner-validation and test
   metrics must stay at chance level for every arm (the train-rows metric of a GBM is above chance
   even without leakage, so it is not the check). In-sample target encoding is detected by check 2.
2. TE delivery check: for arms with target encoding, AUC (or |corr| for regression) of each TE
   column with y on fit rows as delivered vs on test rows. Large gap = in-sample encoding.
3. `openfe` vs `openfe_transductive` on the same folds.
4. Planted signals (`planted()` x 4 kinds x 3 seeds): does each arm reach the oracle's metric, and is
   the true feature (or an equivalent) among the created ones.
5. OpenFE operator equivalence (contract test 7) reported per operator: max abs difference between
   the inductive wrapper and OpenFE's own values on fit rows.

### exp06_automl_end_to_end (post 6, optional, A4)
AutoGluon `best_quality` vs LightAutoML `TabularAutoML` vs H2O AutoML with and without
`preprocessing=["target_encoding"]`, 10 datasets, 600 s each, same folds.

---

## 9. Analysis, figures, claims

- `make aggregate` builds tidy parquet with coverage columns.
- Figures (matplotlib, SVG + PNG, fixed colour per arm across all figures, legible at 1200x1500
  and 1200x628): exp01 type-mapping table; exp02 contrasts with CIs; exp03 role changes; exp04
  median effect with CI by model (and by task type where >= 8 primary datasets), win/tie/loss bars,
  improvement vs generator time (log x), improvement vs n new features (log x; do gains just come
  from more columns?); exp05 permuted-label results and planted-signal grid. In figures and posts
  the manual arm is labelled `manual_baseline`.
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
- Figures are claims too: every figure used in a post has an entry with `kind: figure`, `source`,
  `script`, the output path, a hash of the rendered data table and `approved_git_sha`; `make claims`
  fails if the recomputed data hash differs from the approved one. Number entries have `kind: number`.

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
- Generator timeouts on large datasets would bias results against expensive tools: pilot cap
  3,600 s, full-run cap from measured p95 (section 5), and both failure rules reported (7.6).
- Laptop timing noise: idle machine, on power, medians only, one machine.

---

## 13. What to hand back to Ilia

- A1: exp01 and exp03 tables, corrections to section 1, `datasets.md`.
- A2: smoke report (per arm: ok/failed, n features, times, contract test results).
- A3: pilot results, leakage report, measured full-run estimate, proposed fallbacks.
- End: repo link, `claims.yaml` with values, figures, findings per post in plain words with claim ids.
