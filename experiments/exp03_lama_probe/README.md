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
- Steps 3 (end-to-end runs) and 4 (`lama_adv` extraction) are not run yet.
