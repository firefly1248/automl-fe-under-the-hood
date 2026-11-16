# exp06 AutoML end to end

AutoGluon `best_quality` (`ag_best`), LightAutoML `TabularAutoML` (`lama`) and H2O AutoML without (`h2o`)
and with (`h2o_te`) `preprocessing=["target_encoding"]`, each given 600 s on the whole machine, same folds
as exp02 (repeat 0, 5 outer folds). 10 datasets: the exp02 list minus mfeat-fourier and pendigits (pure
numeric multiclass, where the target-encoding contrast is moot), so multiclass has 2 datasets. 200 units, all
ok. `make exp06` runs it, `make aggregate` writes `results/tidy/exp06*.{parquet,csv}` and
`results/figures/exp06/contrasts.*`; numbers for posts are the `p6_*` entries of `results/claims.yaml`.

Contrast as in exp02: relative improvement of the first arm over the second on log loss (classification)
or RMSE (regression), mean over folds per dataset with a fold-bootstrap 95% CI, median over the datasets of a
task type.

## Findings (descriptive; 4 binary, 2 multiclass, 4 regression datasets)

- `ag_best` is the strongest arm: `lama` and `h2o` are worse than it on every binary dataset, and `h2o` on
  every classification dataset (`p6_lama_ag_*`, `p6_h2o_ag_*`); on regression `lama` is better than
  `ag_best` on one dataset of four, `h2o` on none.
- The median `lama` log loss is about 40% above `ag_best` on binary; on multiclass the two are level.
- Target encoding in H2O does not help: the median contrast is about zero on binary and negative on
  multiclass and regression (`p6_h2o_te_h2o_*`).
- Budget use differs: `ag_best` (mean 588 s), `h2o` (611 s) and `h2o_te` (609 s) run to the limit, `lama`
  stops on its own at a mean of 213 s (52 to 450 s). Part of the `lama` gap is therefore a smaller spent
  budget, not only a weaker method.

Caveats: 2 multiclass datasets only; per-dataset CIs are over folds of one dataset; default settings of
each tool, no tuning, one machine, one repeat. `lama` has no `best_quality` equivalent in this setup, and the
H2O and `ag_best` units overshoot the budget by up to about 30 s.
