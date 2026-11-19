# Article

One Medium article, "What automated feature engineering really does under the hood". Every number in it is a
claim id in `results/claims.yaml`; every image is a PNG of at least 1400 px in `results/figures/article/` with a
hand-written `<name>.alt.txt` next to it (`make article` writes the PNGs and the data tables they render, as the text of the cells for the table images;
the figure claims hash those tables). Section 4 carries `preview.png` (1500 x 680 px, white margins) in the text.

| section | topic | experiments | claim ids | images (`results/figures/article/`) |
|---|---|---|---|---|
| 1 | Introduction: what the tools promise, what is compared | - | - | - |
| 2 | What the tools do under the hood; AutoGluon | exp01, exp02, probes of exp03, exp05 and exp06, exp09 (types) | `s2_*` | `transform_map_automl`, `transform_map_generators`, `exp02_contrasts`, `exp02_single_model`, `exp02_contrasts_table`, `exp02_single_model_table`, `exp02_hp_groups_table`, `exp02_noise_table`, `exp09_types_table` |
| 3 | LightAutoML | exp03 | `s3_*` | - |
| 4 | Open benchmark: do the generated features help | exp04, exp07, exp09, probe exp08 | `s4_*` | `preview`, `median_effect_lgbm`, `median_effect_catboost`, `win_tie_loss_lgbm`, `win_tie_loss_catboost`, `effect_vs_gen_time`, `effect_vs_n_features`, `coverage_table`, `ctr_ablation_table`, `cat_subgroup_table`, `numcast_table` |
| 5 | Leakage checks and planted signals | exp05 | `s5_*` | `permuted_labels`, `planted_signals` |
| 6 | AutoML end to end | exp06 | `s6_*` | `exp06_contrasts`, `exp06_contrasts_table`, `exp06_lama_diag_table`, `exp06_noise_table` |

Results for the contrasts of sections 2 and 6 are given as "N better of M datasets" (the main claim) with the
per-dataset table beside it; the median over datasets is a secondary claim (`role: secondary`), because four
datasets per task type, or two for multiclass in section 6, are too few for a median to carry a conclusion.

The transformation map is built, not edited: `make article` first runs `afe.analysis.transform_map_build`, which holds
every cell (text, short label, status, evidence), resolves the cited source lines by searching the installed tool envs
(`make env` first) and writes `results/tidy/transform_map.csv` and `transform_map_evidence.csv`; a correction goes into that
script. Evidence is exp01, exp03 and exp05 outputs, the probe tables `exp03_dates_text.csv` and `exp06_h2o_probe.csv`
(run by hand, no make target) and source lines. `afe.analysis.transform_map` then draws two PNGs, `transform_map_automl`
(ag_default, ag_tabprep, lama, h2o) and `transform_map_generators` (openfe, autofeat, mljar_golden, manual_baseline),
each with a CSV of the rendered cells. Only the cell colour carries a status (green done, amber partial, grey not
done); whether a cell rests on a run output or on source code is the `status` column of the evidence table (two values,
plus not confirmed), which the `s2_map_*` claims count; unverified parts of a cell are in its `note` column. The `h2o`
column covers the default run and the run with target encoding together, and some `lama` cells come from the `lama_adv`
arm of exp05.

The main exp02 table (`exp02_contrasts_table`, `exp02_contrasts`) is anchored on `A_def`, AutoGluon's default LightGBM
(`B_def` over `A_def`, `D` over `A_def`, then `D` over `D0`, `E` over `D0`, `B` over `B2`); `B` over `A` and `D` over `A`
sit in `exp02_single_model_table` and `exp02_single_model` ("`_r13` in a single model"). For `C`, `D`, `D0`, `E` (exp02)
and `lama` over `ag_best` (exp06) the text may say "clearly better" or "clearly worse" (author's decision 2026-11-19): the noise
measurement (pol and credit-g) shows AutoGluon and LAMA repeating to within 0.4 points, a caveat to state because it covers two
datasets. For `h2o` (5 to 10 points between runs, `exp02_noise_table`, `exp06_noise_table`) the text uses sign counts only. The
tables and dots colour by the fold CI, which covers folds only.

In the contrast tables `better (sign)` counts datasets whose mean effect is above zero; `clearly better` and
`clearly worse` count datasets whose fold CI excludes zero. Dots in the contrast figures use the same colours.
