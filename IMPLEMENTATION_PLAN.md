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
8. **Repo stays private** until Phase 6 review by Ilia. Commit and push only after Ilia approves that exact commit or push (rule 9).
9. **Approval points** (the only ones; wait for Ilia at each):
   - A1: after Phase 1 probes (exp01, exp03 steps 1-2) and the dataset table `results/datasets.md`;
   - A2: after the exp04 smoke run (protocol check);
   - A3: after the exp04 pilot, with the full-run time estimate, proposed caps for arms that timed out
     (section 5) and any fallback you propose;
   - A4: before exp02 (dataset list, settings, measured time of one dataset) and before exp06 (optional
     experiment); before any change of a section 7 parameter or arm list.
   - every commit and every push: show the file list and wait. Nothing is committed automatically.

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
became category; two-valued columns became int8 with the `bool` special type (answered at A1, PROGRESS.md).

### 1.5 LightAutoML 0.4.2 (answered at A1 from exp03_config.yaml)
- `Reader` guesses roles with Normalized Gini over several encodings per column plus rules
  (`lightautoml/reader/guess_roles.py`); a numeric column can become categorical.
- `LGBSimpleFeatures` vs `LGBAdvancedPipeline`: frequency encoding, OOF target encoding for high
  cardinality, categorical intersections, datetime differences and seasonality, optional group-by
  (`use_groupby`, off by default).
- Answered at A1: all GBM models get `LGBAdvancedPipeline`, `linear_l2` gets `LinearFeatures`;
  selection mode 1 (LightGBM gain importance, cutoff 0); `use_groupby` off.

### 1.6 Other tools (verified from wheels by the reviewer; re-check in wrappers)
- OpenFE 0.0.12: `transform(X_train, X_test, features)` concatenates train and test. Operators that
  need column-wide statistics: `freq`, `Combine` (codes factorized on the concatenation),
  `CombineThenFreq`, `GroupByThen{Min,Max,Mean,Median,Std,Rank,NUnique}` (`GroupByThenFreq` has code in
  `FeatureGenerator.py` but is not in the candidate lists). Row-wise operators:
  `abs, log, sqrt, square, sigmoid, round, residual, min, max, +, -, *, /`. Selection uses LightGBM.
  `fit` defaults `tmp_save_path` to `./openfe_tmp_data_xx.feather` (its docstring says
  `openfe_tmp_data.feather`); `transform` requires `n_jobs`. `exit()` is called on error in `_trans`,
  `_evaluate`, `_calculate_multiprocess` and `_calculate_and_evaluate_multiprocess`.
- autofeat 2.1.3: returns original plus new columns with sanitized names; needs `numpy<2`,
  Python < 3.13; has `max_gb`; accepts `categorical_cols`. Selection with L1 linear models.
- mljar-supervised 1.3.2: `supervised.preprocessing.goldenfeatures_transformer.GoldenFeaturesTransformer(results_path, ml_task, features_count, n_jobs)`
  has standalone `fit(X, y)` / `transform(X)`. Ops: diff, ratio both ways (0 where
  the denominator is 0), sum, multiply; `fit` takes every column it is given (MLJAR's own caller passes
  numeric columns only, `preprocessing.py`); scored with a depth-3 decision tree; keeps `min(100, max(10, 0.1*n_cols))` features;
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
    raw/<exp>/gen/<gen_hash>.json      # generator-stage records, working store, gitignored
    raw/<exp>/fit/<fit_hash>.json      # downstream-stage records, working store, gitignored
    raw/<exp>/{gen,fit}.jsonl.gz       # committed records (make pack / make unpack, decided at A3)
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
  each env. `generators/base.py`, `data/` (except `registry.py`, which needs `openml` and runs in
  `core`) and `run/unit.py` import only numpy/pandas/stdlib.
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
  Pins in use: `lama` needs `numpy<2`, `pandas<2`, `scipy<1.16` (PROGRESS.md; `pandas<2` decided at A2:
  LAMA 0.4.2's `OrdinalEncoder` is wrong under pandas 2). `autofeat` passes
  `force_all_finite` to scikit-learn, which 1.9.1 no longer accepts: pin `scikit-learn<1.8` in the
  `autofeat` env, and make `env-check` fit and transform on `hard_columns` to confirm it.
  `openfe` needs `scikit-learn<1.6` (it calls `mean_squared_error(squared=False)`; found in the
  Phase 3 smoke, regression only). Other extra pins only as fallbacks if the resolver or imports fail.
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
  are made on that subgroup within task type (binary 8 vs 12, regression 8 vs 12; multiclass has one
  and is not used for them), descriptively: with 8 datasets and 8 arms per Holm family no Wilcoxon
  test can reach significance (minimum two-sided p 0.0078 > 0.05 / 8).
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
  (columns: row id, repeat, outer fold, `inner_mask` bit per fold), committed.
- `results/datasets.md`: id, version, name, family, primary flag, task, rows, cols, n categorical,
  max categorical levels, duplicate-row share, checksums. Part of A1. Variants (e.g.
  `Amazon_employee_access_int`) get their own `n_categorical`, `max_cat_levels` and `hicard`; the
  current table copies the parent's: fix `registry.save` in Phase 3, rerun `make data`, reconfirm at A2.

---

## 5. Run store and resumability

Two stages, so a generator runs once per fold and is reused by both downstream models.

- Generator unit: `(exp, dataset, repeat, fold, arm, arm_params)` plus
  `protocol_hash` (hash of the `protocol:` section of the exp `config.yaml` and `PYTHONHASHSEED`; limits are
  not in it and are stored in each record), `data_hash`, `splits_hash`, `env_lock_hash` (hash of the arm's `uv.lock`),
  `code_hash` per arm: the arm's own module and its registry entry, plus the shared code that affects
  results (`generators/base.py` with the base view, `run/worker.py`, `data/loaders.py`, `data/splits.py`).
  Orchestration, limits and logging (`run/cli.py`, `run/limits.py`, `run/store.py`, the registry of other
  arms) are not in it, so a fix to one wrapper reruns only that arm (decided at A3).
  `gen_hash = sha1(json.dumps(unit, sort_keys=True))[:16]`. Output: cached parquet + record.
- Fit unit: `(gen_hash, arm, view, downstream_model, model_params, seed, core_lock_hash, code_hash)`
  -> `fit_hash`. `arm` and `view` keep `ag_tabprep_keepcat` apart from `ag_tabprep`, which share a
  `gen_hash`; `core_lock_hash` is the hash of `envs/core/uv.lock` (a LightGBM/CatBoost bump invalidates
  fits); the fit `code_hash` also covers `src/afe/models/` (downstream).
- A generator unit whose `applicable(X_fit)` is False writes a `not_applicable` record through
  `out.json`, no cache; its fit units are not run.
- Skip a unit if its record exists with status in `{ok, failed, timeout, oom, not_applicable}`; `--retry-failed`
  reruns `failed`, `timeout`, `oom`; `--retry-status timeout` reruns only timeouts (after a cap is raised). If a cache parquet is missing but the gen record is ok, regenerate it.
- Atomic writes: `*.tmp`, fsync, rename.
- Gen record: unit, status, error (<= 5 kB, `$HOME` stripped), caps, `jobs`, n features in / new / final,
  n infinite values replaced, fit time, transform time (train, inner, test), peak RSS, arm env
  versions, git sha, machine info (rule 7), timestamps.
- Fit record: test metrics, inner-validation metric, best iteration, fit time, peak RSS.
- Limits (`run/limits.py`): generator wall clock 3,600 s, downstream wall clock 3,600 s (1,800 s up to
  the pilot; raised at A3 for all datasets). `exclude` in the exp config lists (dataset, model) pairs that
  are not run and are left out of every comparison on that model, with the reason: connect-4 on
  CatBoost (`base` timed out at 1,800 s on 5 of 5 pilot folds, decided at A3). Caps are
  stored in each record. A run under a cap cannot measure times above it, so the pilot cannot set a
  higher cap by itself: if an arm times out in the pilot, A3 decides a higher cap for that arm and
  stage, the pilot timeouts are rerun with it (`--retry-status timeout`), and the full run uses the
  decided caps. Memory cap 6 GB enforced by a psutil RSS watchdog that kills the child process tree (Linux and macOS; do not
  use `RLIMIT_AS`, it kills torch/numba/LightGBM early). The record keeps the watchdog's marker (`watchdog`:
  reason and peak tree RSS); `oom` and `timeout` come only from it, any other kill is `failed`. On Linux `systemd-run --user --scope -p
  MemoryMax=6G` is an allowed alternative.
- Threads: in every worker set `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=NUMBA_NUM_THREADS=1`,
  `torch.set_num_threads(1)`, LightGBM `num_threads=1`, CatBoost `thread_count=1`, and `n_jobs=1`
  in every tool (OpenFE, GoldenFeaturesTransformer, autofeat, LAMA `cpu_limit=1`).
- `PYTHONHASHSEED=0` for every child process, in the protocol hash (decided after the A3 code review; it
  was the unit's tool seed before). autofeat's selection depends on str/set hash order, which Python
  randomizes per process: on one concrete fold two hash seeds gave 14 and 28 features. Contract test 2
  runs in one process and could not see it. Every arm and both downstream models were run on the contract
  frames in two processes with hash seeds 1 and 2: only autofeat differed. Its units are rerun; the
  records of the other arms were moved to the new hashes, checked by rerunning a sample.
- Parallelism: `jobs = min(physical_cores - 2, floor((ram_gb - 4) / mem_gb))`; `--jobs` may lower it.
  The memory term reserves `mem_gb` for each running unit and leaves 4 GB for the system and other apps.
  On the 24 GB laptop with 14 cores and `mem_gb: 6` this gives 3. The pilot records peak RSS per arm and
  stage; A3 may replace the single `mem_gb` by per-stage caps from those peaks, and the scheduler then
  reserves each running unit's own cap. The launch gate `min_free_gb` (2 GB) stays as a check against
  other apps. On macOS the RSS watchdog does not see compressed memory, so a child that dies by SIGKILL
  (-9) when neither the watchdog nor the timeout killed it is recorded as `oom`. The child's stdout goes
  to the same log as stderr (OpenFE prints its tracebacks to stdout). Close other apps for the full run.
- Per-unit temp dir (OpenFE `tmp_save_path`, MLJAR `results_path`, LAMA/AG output paths), deleted
  after the unit.
- `make commit-results`: aggregate, stage raw + tidy + figures + coverage, show the staged file list and
  diff. It does not commit; commit and push need Ilia's approval (section 0 rule 9).

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
    def applicable(self, X) -> bool:  # False if the tool cannot run on this input (e.g. < 2 numeric
                                      # columns for mljar_golden); recorded as `not_applicable`
```

Downstream input = `base_view(original) + new columns`, or the arm output alone if `replaces_input`.

`base_view` (applied identically in every arm, also to `base`): object/string -> `category`;
datetime -> int64 ns; bool -> int8; nothing else. LightGBM gets `category` dtype; CatBoost gets
the same columns as `cat_features` with NaN filled by the string `"__nan__"`.

Contract tests (`tests/test_generators_contract.py`, run on the synthetic frame and two small real
datasets, credit-g (binary) and Moneyball (regression, added at A2), for every arm):
1. Output index/rows equal input; unique names; numeric or category dtypes; inf replaced by NaN and counted.
2. Determinism: `fit_transform` twice with the same seed gives identical output.
3. Isolation: running the arm on fold 0 and then fold 1 in one process and temp root gives the same fold-1
   output as fold 1 alone (no state carried through files, caches or globals, e.g. MLJAR's
   `golden_features.json`).
4. Batch independence: `transform(test)` equals `transform` of its two halves concatenated, and of single
   rows for a 20-row sample (catches statistics computed over the transformed batch).
5. Unseen categories in test do not crash, and the test asserts the value: NaN or the tool's documented
   default.
6. For arms with target encoding: fit-rows output differs from `transform(fit rows)` (proves OOF is used).
7. `openfe` only: the reimplemented column-wide operators reproduce OpenFE's own values computed on the
   fit rows alone (OpenFE's operator code on `X_fit` only; not `transform(X_fit, X_fit, ...)`, which
   duplicates the rows and doubles counts), keeping OpenFE's conventions: raw counts, `std` with
   ddof=1, pct rank (equivalence test, per operator). Without it
   the inductive arm can be dismissed as "not OpenFE". On other rows OpenFE has no inductive reference
   (its `transform` is transductive): there each operator is checked against a direct pandas computation
   of the same statistic from fit rows.

Arms, main ranking (all inductive):
| arm | env | what |
|---|---|---|
| `base` | core | `base_view` only |
| `ag_default` | autogluon | `AutoMLPipelineFeatureGenerator` output; `replaces_input=True` |
| `ag_tabprep` | autogluon | the exact `_r13` `ag.prep_params` group, built as AutoGluon builds it (section 1.2), applied after `AutoMLPipelineFeatureGenerator`. Output rule: **as AutoGluon does it** (categoricals replaced by TE), `replaces_input=True`. Record this in the README; it is the one arm that does not keep native categoricals. In posts call it "the `_r13` configuration", not "TabPrep", and say that most of its features are random, unfiltered arithmetic (counts in `results/tidy/exp01_tabprep.csv`; posts cite them through `claims.yaml`) that the portfolio's own tuned LightGBM config, not a default LightGBM, was built to absorb |
| `ag_tabprep_keepcat` | core (fit stage only) | `base_view(original)` + the `ag_tabprep` group's new columns, native categoricals kept. New = columns produced by the group's generators (Arithmetic, CatInteraction, OOF-TE), identified per generator by the wrapper, not by name; pass-through columns are not new; the list is stored in the `ag_tabprep` record. Fit units with `view="keepcat"` on the `ag_tabprep` `gen_hash` (no extra generator run; when `ag_tabprep` fails, keepcat has the same gap); separates "new features" from "losing native categoricals", which matters most for CatBoost |
| `lama_adv` | lama | LAMA reader + advanced LGB feature pipeline extracted as a transformer (exp03.4), as LAMA does it by default: reader roles, `LGBAdvancedPipeline` encodings with LAMA's own OOF folds, selection mode 1 included; `replaces_input=True` (its output already encodes the originals). If the extraction cannot include selection, say so at A2. Removed at A2 if the extraction is not reliable |
| `openfe` | openfe | inductive wrapper (below). `fit` with library defaults (`n_data_blocks=8`, `min_candidate_features=2000`, `stage1_metric='predictive'`, `stage2_metric='gain_importance'`), `task` passed explicitly (`binary` and `multiclass` map to OpenFE's `classification`), `seed` from 7.1, `n_jobs=1`, `tmp_save_path` in the unit temp dir. `fit` returns every stage-1 survivor ranked by stage-2 gain, with no cut-off: keep the features whose gain in `new_features_scores_list` is > 0 (the positive-score rule OpenFE itself applies at stage 1), hard cap 2,000 (= `min_candidate_features`, = `ag_tabprep` `max_new_feats`). Candidates before and after the cut, and whether the cap fired, go into `describe()` and the record |
| `autofeat` | autofeat | `feateng_steps=2`, `feateng_cols` = numeric columns only; categoricals with <= 50 levels are one-hot encoded by the wrapper on fit-row levels and passed as numeric 0/1 columns with `categorical_cols=None` (autofeat's own encoder refits on every `transform` batch); wider categoricals are left out and listed in `describe()`; dummies are inputs, not new features; `max_gb=4` (below `mem_gb`), and the row subsample autofeat takes above it is recorded; `np.random.seed(seed)` before `fit` (autofeat permutes with the global generator); not applicable without numeric columns; NaN imputed by the wrapper with train medians (recorded); originals stripped, names mapped back |
| `mljar_golden` | mljar | `GoldenFeaturesTransformer`, fresh `results_path` per unit, `n_jobs=1`, `features_count` library default (10 on every dataset here, all have <= 100 columns); the wrapper passes numeric columns only with NaN filled by fit-row medians, as MLJAR's own caller does, and calls `transform` on a copy (it writes into its input); not applicable below 2 numeric columns |
| `manual_baseline` | core | exact rules below |

Separate arm: `openfe_transductive` (library default `transform(train, test)`), only in exp05. It does not
fit the interface: a separate worker function takes the feature list fitted by the `openfe` arm on the
same fold and calls `transform(X_fit, X_other)` with X_other = inner and test rows together, so exp05
check 3 differs only in how statistics are computed. As built (accepted at A3): the arm repeats the
OpenFE search with the same seed instead of reading the `openfe` feature list (the search is
deterministic, contract test 2, so the features are the same; the cost is a second search), and runs on
the 5 folds of repeat 0 of the pilot datasets (fold 0 at A3, extended alongside `full_r0`). Its time is
not published.

Applicability: `mljar_golden` needs >= 2 numeric columns, `autofeat` >= 1; every other arm always
applies (an arm that builds nothing returns 0 new columns, a tie).

OpenFE inductive wrapper: reimplement the column-wide operators (`freq`, `Combine`,
`CombineThenFreq`, `GroupByThen*`) with statistics fitted on fit rows only and mapped to other rows
(unseen keys -> NaN); row-wise operators are applied as is. `GroupByThenRank` on a new row = share of
fit-row values in its group below the value plus half of those equal to it. `Combine` gives a category
with the fit-row levels; unseen combinations -> NaN.

`manual_baseline` rules (all statistics on fit rows only):
- frequency encoding of every categorical column (share of rows);
- top-5 numeric columns by mutual information with y (`mutual_info_*`, `random_state=seed`, on a copy
  with NaN filled by fit-row medians);
  for each pair: `a/b` and `b/a` with denominator `|b| < 1e-12` -> NaN, and `a*b`;
- key = the categorical column with the highest mutual information with y among those with at most
  50% unique values on fit rows (decided at A2: discrete MI grows with the number of levels and picked
  ID-like columns) (if none: decile bins of the top-MI numeric column); for each top-5 numeric: group mean and `x - group_mean`
  (unseen key -> global mean; NaN in the key is its own level).

---

## 7. Benchmark protocol (exp04; confirmed at A2)

### 7.1 Folds
5-fold outer CV (stratified for classification, plain KFold for regression), 2 repeats: 10 splits
per dataset (restored at A3; plan v3 had cut it to 1 repeat). Run as two stages, `full_r0` then
`full_r1`, with a commit after each. Each repeat has its own seeds, so adding repeat 1 left the repeat 0
splits unchanged. Seed of a repeat r: `1000 + r`; tool seed for (repeat r, fold k): `1000 + 10*r + k`.

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
Per dataset, per arm and model, on the 10 splits:
- AUC: paired raw difference of per-dataset means `arm - base` (AUC is already unit-free; the
  gap-closed ratio is dropped because it explodes near AUC 1);
- losses (log loss, RMSE): relative improvement on per-dataset means `(base - arm) / base`; for log
  loss also the raw difference, since datasets with log loss near 0 dominate relative numbers;
- descriptive win / tie / loss: corrected repeated k-fold t-test (Nadeau-Bengio, J = 10, correction
  n_test/n_train with n_train = fit rows, df = 9) at alpha 0.05, no multiplicity correction
  within a dataset. If all fold differences are 0 (e.g. the arm created no features), record a tie
  with p = 1. Per-dataset tests are near-powerless; they are descriptive only, say so.
Across datasets (primary conclusions; primary set = one dataset per family, section 4.2):
- within each task type: median effect with bootstrap 95% CI over datasets (10,000 resamples), on AUC
  difference and relative log loss improvement (binary), relative log loss improvement (multiclass),
  relative RMSE improvement (regression). Effects in different units are never pooled;
- Wilcoxon signed-rank (`zero_method="pratt"`) on the same per-dataset effects. Family: all arms within
  one (task type, model, primary metric, failure rule); Holm within each family. Subgroups and
  secondary metrics are descriptive and labelled so;
- across task types only dimensionless numbers: share of datasets with a descriptive win and with a
  descriptive loss.
Raw metric differences other than AUC and log loss are reported per dataset only (units differ).
A task type with < 8 primary datasets is reported descriptively only (none at present: 20 / 10 / 20).
Breakdown: with vs without a categorical column of >= 10 levels. Case-study datasets (not primary) are reported separately.

### 7.6 Failures in aggregates
Two rules (decided at A3: base fallback is the main one, complete cases the secondary one):
- base fallback: a failed or timed-out unit takes the `base` score for that fold ("what a user gets");
- complete cases only ("what the tool does when it runs"): `base` is restricted to the same folds; a
  dataset where the arm has no ok fold drops out of that arm's comparison and shows in coverage.
Coverage (ok / planned) is shown next to both, per arm, split by stage: generator failure or timeout vs
downstream timeout; `not_applicable` units are a separate coverage row, not counted as failures. A
`not_applicable` unit takes the `base` score in aggregates and is left out of complete cases.
Failures are results: limits are not raised for an arm that fails (autofeat, decided at A3). A claim that holds under one rule
only says so.

### 7.7 Cost
Median and range (with n) of generator time, and of `(generator + downstream) / base downstream` time.
Timing claims come from a separate timing pass with `jobs=1` on the 6 per-task smallest and largest
pilot datasets x 1 fold (all arms), run under its own experiment key `exp04_timing` so its units never reuse main-run records,
machine idle and on power; the main run is parallel and never idle, so its times are for
budgeting only.

### 7.8 Runtime and fallbacks (decided at A3)
Full-run cost: about 50 datasets x 10 splits x 9 arms (7 generator arms, generator stage cached) x 2
models. Measured from the pilot (`exp04_open_benchmark/estimate.py`, A3): about 20 unit-hours per split,
so about 34 h of wall time per repeat at 3 jobs (the 24 GB laptop, section 5); `autofeat` is a third of
it and the least certain part.
Fallbacks to propose at A3 if needed, in this order: row cap 50,000 (stratified subsample,
recorded); drop the slowest arm on datasets it cannot finish, reported as coverage. Caps are decided
as in section 5.
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
   candidates built, kept, time (no correlation filter with the default `selection_method`, 1.2); does it
   recover the planted `a/b`?

### exp02_autogluon_ablation (post 2) — env autogluon, runs alone, A4 before the run
12 primary datasets with <= 60,000 rows, fixed in the exp02 config before the run: per task type the
first 2 with a categorical of >= 10 levels and the first 2 without, in OpenML data id order; where a
task type has fewer than 2 such datasets (multiclass has one, okcupid-stem), the rest come from those
without. Outer folds of repeat 0 (5 folds). One unit at a time, without the section 5 thread variables;
AutoGluon gets `num_cpus = physical_cores - 2`; memory watchdog at `ram_gb - 4`. Every arm fits on the
full outer train (fit and inner rows) and validates as AutoGluon does by default (holdout or bagging);
`time_limit=600` for every arm. Contrasts are reported per dataset (mean over folds, fold bootstrap
CI) and as the median within task type, descriptively (4 datasets per task type).
- A: `GBM` with the `_r13` hyperparameters and no `ag.prep_params`, no bagging;
- B: `GBM_PREP` `_r13`, no bagging (B - A = new features plus TE replacing native categoricals, 1.2;
  not feature construction alone);
- B2: `GBM_PREP` `_r21` (portfolio's own no-prep control), for reference;
- C: `medium_quality`;
- D: `best_quality` with `dynamic_stacking=False` (2023 portfolio, stacking; DyStack off so D and D0
  both fit models for the full 600 s, not after a detection run that takes 1/4 of `time_limit`);
- D0: `best_quality` with `num_stack_levels=0` (separates stacking from portfolio);
- E: `best_quality_v150` (2025-12-18 portfolio with `GBM_PREP`, no stacking, early-stop callback).
Contrasts: B - A (features + TE instead of native categoricals), D - A (models + bagging + ensembling), D - D0
(stacking), E - D0 (new portfolio + early-stop callback; the README lists every resolved difference).
Record the trained model list, fit time and the machine; state that C, D, D0, E depend on CPU speed. Record resolved hyperparameters from the predictor.
Caveat for the README and post: with `time_limit=600` on this laptop, D and E will not finish their
portfolios on the larger datasets; D - A and D - D0 measure "what fits in 600 s on 12 cores", not the
full presets. The per-fold trained-model list shows how much of each portfolio ran.

### exp03_lama_probe (post 3) — env lama
1. Default config of `TabularAutoML` for 0.4.2: feature pipeline per model, `auto_unique_co`,
   intersection settings, `use_groupby`, selection mode. Save `exp03_config.yaml`.
2. Reader alone on `hard_columns()` and the typed set: `column -> dtype -> LAMA role -> reason`
   (`exp03_roles.csv`). Highlight numeric -> categorical and the reverse.
3. `TabularAutoML(task, timeout=600, cpu_limit=...)` on `adult`, `diamonds`, `Amazon_employee_access`:
   features each model received, selected/dropped features, OOF score. Compare simple vs advanced
   pipeline if reachable via config.
4. Try to extract reader + advanced pipeline as a `FeatureGenerator` (`lama_adv`); run in Phase 3,
   before the smoke run. If the internal API makes it unreliable, document the attempt, keep the code,
   and propose removing the arm at A2.

### exp04_open_benchmark (posts 4 and 5)
Stages: `smoke` (3 datasets, 1 fold, all arms, both models, contract checks) -> A2 -> `pilot`
(10 primary datasets, 5 folds of repeat 0, time estimate) -> A3 -> `full_r0` -> commit -> `full_r1` ->
commit; plus the `jobs=1` timing pass (7.7).
Pilot datasets, fixed in config: per task type the smallest and the largest by rows; then, over all
task types, the two largest with a categorical of >= 10 levels and the two widest by columns, each not
already chosen, so the pilot includes the slow cases.
Outputs: `results/tidy/exp04.parquet`, tables by task type, figures, README with findings and caveats.

### exp05_leakage_checks (posts 4 and 5) — after the pilot, finished before A3
1. Permuted labels (6 datasets: the 3 smoke datasets; the first 2 primary datasets with a categorical
   of >= 10 levels not already chosen, by data id; the first primary multiclass dataset not already
   chosen), under its own experiment key `exp05_permuted`: permute y on fit and inner rows, so early
   stopping also sees noise. The downstream test metric must stay at chance level for every arm: per arm
   and task type, the 95% bootstrap CI over units of AUC - 0.5 (binary), of prior entropy (class shares
   of the fit rows) - log loss (multiclass) and of R2
   (regression) contains 0 or lies below it (the train-rows metric of a GBM is above chance
   even without leakage, so it is not the check). In-sample target encoding is detected by check 2.
2. TE delivery check: for arms with target encoding, AUC (or |corr| for regression) of each TE
   column with y on fit rows as delivered vs on test rows. A gap above 0.05 in AUC (0.1 in |corr|) flags
   in-sample encoding.
3. `openfe` vs `openfe_transductive` on the same folds.
4. Planted signals (`planted()` x 4 kinds x 3 seeds, n = 5,000, registered as `planted_<kind>_s<seed>`,
   not primary, own experiment key `exp05_planted`, splits as for any dataset): does each arm reach the
   oracle's metric (within 1% relative), and is the true feature, or one with |corr| >= 0.95 with it on
   test rows, among the created ones.
5. OpenFE operator equivalence (contract test 7) reported per operator: max abs difference between
   the inductive wrapper and OpenFE's own values on fit rows.

### exp06_automl_end_to_end (post 6, optional, A4)
AutoGluon `best_quality` vs LightAutoML `TabularAutoML` vs H2O AutoML with and without
`preprocessing=["target_encoding"]`, 10 datasets, 600 s each, same folds.

---

## 9. Analysis, figures, claims

- `make aggregate` builds tidy parquet: one row per fit unit with exp, stage, dataset, family, primary,
  task, hicard, repeat, fold, arm, model, status and failing stage, the metrics of 7.4, n new features,
  generator and fit time, caps; plus coverage columns. `make claims` filters `primary` unless a claim
  says otherwise. `stat` names: `median_effect`, `share_win`, `share_loss`, `wilcoxon_p_holm`, `coverage`
  (exp04); `value` and `count` (single numbers from the tidy tables of exp01-03 and exp05); `contrast`
  (exp02); `median_time` (7.7). Each exp04 tidy row also carries the `base` metric of the same dataset,
  fold and model, and each claim names its `metric`.
- Figures (matplotlib, SVG + PNG, fixed colour per arm across all figures, legible at 1200x1500
  and 1200x628): exp01 type-mapping table; exp02 contrasts with CIs; exp03 role changes; exp04
  median effect with CI by task type and model, win/tie/loss bars,
  improvement vs generator time (log x), improvement vs n new features (log x; do gains just come
  from more columns?); exp05 permuted-label results and planted-signal grid. In figures and posts
  the manual arm is labelled `manual_baseline`.
- `results/claims.yaml`:
  ```yaml
  - id: p4_openfe_reg_median_lgbm
    post: 4
    text: "median relative RMSE improvement of OpenFE on regression, LightGBM downstream"
    source: results/tidy/exp04.parquet
    query: "primary & task=='regression' & arm=='openfe' & model=='lgbm'"
    stat: median_effect
    metric: rmse
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
make aggregate figures claims commit-results
```

---

## 11. Phases

| phase | work | done when |
|---|---|---|
| 0 | skeleton, envs, `env-check` | all envs import; versions in `ENVIRONMENT.md`; LAMA recipe documented |
| 1 | exp01, exp03 steps 1-2, dataset registry + `datasets.md` | tables saved, section 1 confirmed or corrected -> **A1** |
| 2 | splits, store, worker, limits, contract tests | `make test` green; kill-and-resume test (kill mid-run, rerun: no duplicates, no lost units) |
| 3 | all arm wrappers (incl. `lama_adv` from exp03 step 4), registry fix for variants (4.2), exp04 smoke | smoke report -> **A2** |
| 4 | exp04 pilot, exp05 | time estimate, leakage report -> **A3**, then exp04 full; coverage >= 95% or failures explained per arm |
| 5 | exp02, exp03 step 3 | READMEs with findings |
| 6 | aggregate, figures, claims, README, `posts/README.md` | `make claims` passes; fresh clone reproduces exp01 and exp04 smoke |

After each phase: add a `PROGRESS.md` entry (date, what ran, coverage, surprises, open questions).
Commit and push only after Ilia approves them. This file is how Ilia and later sessions resume the work.

---

## 12. Likely problems

- OpenFE may break on new pandas: pin in its env. Its `exit()` calls (section 1.6) can end a unit without
  an `out.json`: the OpenFE wrapper catches `SystemExit` and raises `RuntimeError`, and `_launch` records a
  missing `out.json` as `failed`, never `ok`.
- autofeat: NaN and memory; impute and set `max_gb`, or record the failure.
- AutoGluon generators need AutoGluon feature metadata: run `AutoMLPipelineFeatureGenerator` first
  inside `ag_tabprep`.
- OOF-TE with many classes: one column per class; the 20-class filter keeps this bounded.
- `ag_tabprep` can add up to ~2,000 columns; CatBoost on that input can hit the downstream cap; A3 can
  raise it per arm (section 5), and fit timeouts are shown apart from generator failures (7.6). Always show n features next to cost.
- Generator timeouts on large datasets would bias results against expensive tools: caps of arms that
  time out in the pilot are raised at A3 (section 5), and both failure rules are reported (7.6).
- Laptop timing noise: idle machine, on power, medians only, one machine.

---

## 13. What to hand back to Ilia

- A1: exp01 and exp03 tables, corrections to section 1, `datasets.md`.
- A2: smoke report (per arm: ok/failed, n features, times, contract test results).
- A3: pilot results, leakage report, measured full-run estimate, proposed caps and fallbacks.
- End: repo link, `claims.yaml` with values, figures, findings per post in plain words with claim ids.
