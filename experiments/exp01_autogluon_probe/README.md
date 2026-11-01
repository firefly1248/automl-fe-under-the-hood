# exp01: AutoGluon probe (post 2)

**Question.** What does AutoGluon 1.6.3 do to raw columns by default, and which presets reach the
TabPrep generators (`GBM_PREP`)?

**Method.** `run.py`: `AutoMLPipelineFeatureGenerator` on `hard_columns()` (3 seeds, plus
`cat_rare` singletons-only and n = 500 variants); `ArithmeticFeatureGenerator` alone and the `_r13`
group (built by AutoGluon's own `get_preprocessor_tabprep`) on `hard_columns()`, `adult`, `diamonds`.
`inventory.py`: generator defaults, preset -> portfolio resolution, every `ag.prep_params` chain.

**Outputs.** `results/tidy/exp01_types.csv`, `exp01_feature_metadata.json`, `exp01_generators.csv`,
`exp01_presets.csv`, `exp01_prep_configs.csv`, `exp01_tabprep.csv`; AutoGluon logs in `results/raw/exp01/`.

**How to read.** `exp01_types.csv`: one row per input column: action (`kept`, `retyped`, `expanded`,
`dropped`) and the stage that dropped it.

**Findings.**
- Default preprocessing builds no new features; it retypes and drops (`exp01_types.csv`). `const` is
  dropped as useless, `dup_num` and `uid` as unused. Int-coded `city_code` and `date_int` stay int64;
  `date_str` becomes int64 plus year/month/day/dayofweek. Two-valued columns become int8 (`bool` special
  type); `text_short` and `num_as_str` become category. `mixed` (numbers with "n/a") is typed as a
  datetime on every seed. `text_long` gives n-gram columns only at the larger n; `cat_rare` with
  singletons only is dropped.
- Only `GBM_PREP` reaches the TabPrep generators: 14 cpu configs (7 both branches, 3 Arithmetic only,
  3 CatInteraction + TE only, `_r21` empty), reached only by the `*_v150` presets
  (`exp01_prep_configs.csv`, `exp01_presets.csv`).
- `ArithmeticFeatureGenerator` with the default `selection_method="random"` applies no correlation
  filter; it recovers the planted `a/b` on `hard_columns()`. The `_r13` group leaves no `category`
  columns: TE replaces them (`exp01_tabprep.csv`).

**Caveats.** Feature counts depend on the input width after `AutoMLPipelineFeatureGenerator` (n-grams
count as inputs). Row `subsample` is implemented despite its TODO comment.
