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

**Findings.** Pending (not yet run).

**Caveats.** Source-reading notes (to confirm by the run): with the default `selection_method="random"`,
`ArithmeticFeatureGenerator` does not apply `corr_threshold` (only the `spearman` path does,
`arithmetic/preprocessor.py`); row `subsample` is implemented despite its TODO comment.
