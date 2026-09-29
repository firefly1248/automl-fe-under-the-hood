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

**Findings.** Pending (not yet run).
