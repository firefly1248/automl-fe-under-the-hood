# exp03: LightAutoML probe (post 3)

**Question.** Which feature pipeline does `TabularAutoML` 0.4.2 give each model by default, and how
does its `Reader` assign column roles?

**Method.** Steps 1-2 of plan section 8. `run.py` step 1 reads the resolved defaults of
`TabularAutoML` per task; step 2 runs `PandasToPandasReader` with the `TabularAutoML` reader params on
`hard_columns()` and the typed set, recording the stat tables of the reader's own rule functions.

**Outputs.** `results/tidy/exp03_config.yaml`, `exp03_roles.csv`, `exp03_role_stats.csv`.

**How to read.** `exp03_roles.csv`: `simple_role` is the first-pass guess (castable to float ->
numeric; parsable as date -> datetime; else category), `lama_role` the final role, `change` marks
numeric -> category and the reverse, `reason` names the rules that fired.

**Findings.**
- Defaults (`exp03_config.yaml`): every GBM algorithm (lgb, lgb_tuned, cb, cb_tuned) gets
  `LGBAdvancedPipeline`; `linear_l2` gets `LinearFeatures`. Selection mode 1: LightGBM gain importance on
  `LGBSimpleFeatures`, cutoff 0. Reader `drop_score_co` is 0.0 (class default 0.01); `use_groupby` is off.
- Roles (`exp03_roles.csv`): numeric -> category for `city_code`, the Amazon ids, adult
  `age, fnlwgt, capital-gain, capital-loss`, bank-marketing `V1, V10, V14`, credit-g `residence_since`,
  house_sales `zipcode, yr_built, long`. LAMA keeps `dup_num` and `uid` (as ordinal category), drops
  `near_const` by the constant-rate check, and reads `text_long` as a category (no text role without a
  manual role). house_sales `date` is read as Datetime.
- Step 4 (`lama_adv`, `src/afe/generators/lama_pipeline.py`): the arm wires LAMA's own classes as
  `TabularAutoML` does for its GBMs: `PandasToPandasReader` with the preset reader params, a `FoldsIterator`
  over the reader's folds, `ImportanceCutoffSelector(LGBSimpleFeatures, BoostLGBM, ModelBasedImportanceEstimator)`
  (selection mode 1, cutoff 0, fit on the first fold as holdout), then `LGBAdvancedPipeline` with the preset
  GBM params and the selector as `feats_imp`. Selection is included. Folds: the reader's 5-fold split
  (stratified for classification) with `random_state` = the unit seed (LAMA default 42); fit rows get the
  pipeline's `fit_transform` output, so target encodings are out-of-fold over those folds; other rows get
  `transform`. Selector LightGBM: `num_threads=1`, seed = unit seed.
- LAMA 0.4.2 bug under pandas >= 2: `OrdinalEncoder.fit` reads the counts column of
  `value_counts().reset_index()` instead of the values (`transformers/categorical.py:1053`), so every string
  categorical is NaN in `LGBSimpleFeatures` and selection drops all of them (credit-g under pandas 2.3.3:
  4 features left, `TabularAutoML` itself selects the same). The env pins `pandas<2` (1.5.3), where LAMA's
  own code path is correct; credit-g then keeps 16 selected features with OOF target encodings of category
  intersections. The bug also feeds the reader's role scores: under pandas 1.5.3 the encoding subtype
  (auto, freq, ordinal) of 10 categorical columns changed in `exp03_roles.csv`; the numeric -> category
  changes and the dropped columns above did not.
  Minimal reproduction, `pandas2_bug.py`: 1,000 rows, a five-level string column that sets the target, run
  through `PandasToPandasReader` and `LGBSimpleFeatures`. Share of NaN in the encoded column: 0.00 under
  pandas 1.5.3 (`envs/lama`), 1.00 under pandas 2.3.3 (same LAMA 0.4.2, numpy 1.26.4, scipy 1.15.3,
  statsmodels 0.14.0, torch 2.14.0, LightGBM 4.7.0).
- Step 3 (end-to-end runs) is not run yet.
