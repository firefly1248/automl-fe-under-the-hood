# exp02 AutoGluon ablation

What in AutoGluon's gain comes from new features, from model diversity, from stacking, and from the
portfolio. 12 datasets (4 per task type, rule in `config.yaml`), repeat 0, 5 outer folds, 7 arms
(`worker.py` docstring), `time_limit=600`, log loss or RMSE as the eval metric, one unit at a time on the
whole machine. 420 units, all ok. `make exp02` runs it, `make aggregate` writes the tables and
`results/figures/exp02/contrasts.*`.

Contrast = relative improvement of the first arm over the second on the eval metric (positive favours the
first), mean over folds per dataset with a fold-bootstrap 95% CI, then the median over the 4 datasets of a
task type. Numbers live in `results/tidy/exp02_contrasts_summary.csv` and `results/claims.yaml` (`p2_*`).

## Findings (descriptive; 4 datasets per task type)

- Stack and portfolio dominate: `D` over the single-LightGBM `A` is the largest contrast in every task type
  (`p2_d_a_*`). It bundles many models, bagging and stacking, so on its own it does not say which of them helps.
- Stacking over the same portfolio (`D` over `D0`) is close to zero on binary, small and positive on
  multiclass and regression (`p2_d_d0_*`).
- The 2025 portfolio with `GBM_PREP` and no stacking (`E` over `D0`) is slightly worse on binary and
  regression, level on multiclass (`p2_e_d0_*`).
- TabPrep inside one LightGBM (`B` over `A`) is a small gain on binary and multiclass and a larger one on
  regression (`p2_b_a_*`); against its own no-prep control (`B` over `B2`) the sign flips on binary
  (`p2_b_b2_*`), so part of the `B` - `A` gap is the config, not the new features.
- Cost: mean wall per unit `A` 23 s, `B` 40 s, `B2` 21 s, `C` 83 s, `E` 346 s, `D0` 527 s, `D` 554 s;
  `D` and `D0` sit at the 600 s budget on most sets (`results/tidy/exp02.parquet`).

Caveats: 4 datasets per task type make the medians descriptive; the CIs are over folds of one dataset
and say nothing about other datasets. `D` and `D0` turn dynamic stacking off so that both spend the full
budget; AutoGluon's default would stop earlier. Timing is one machine with all cores and no other job (the
run was paused four times for other jobs; each paused unit was recomputed from scratch).
