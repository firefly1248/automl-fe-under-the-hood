# exp05: leakage checks (plan section 8)

```
make exp05                                   # permuted, transductive, planted runs
uv run --project envs/openfe python experiments/exp05_leakage_checks/openfe_ops.py       # check 5
uv run --project envs/core python experiments/exp05_leakage_checks/planted_oracle.py     # oracle for check 4
uv run --project envs/core python experiments/exp05_leakage_checks/report.py             # checks 1-4
```

Results: `results/tidy/exp05_report.md` and `exp05_*.csv`.

## Check 3: transductive against inductive OpenFE

To reproduce: `make exp05` (the runs), `report.py`; the kick refit below is `experiments/exp05_leakage_checks/kick_missing_level.py`, run by hand.

`openfe_transductive` (OpenFE's own `transform(train, test)`, statistics over the fit, inner and test rows together) against the repo's
inductive `openfe` on the 5 folds of repeat 0 of the 10 pilot datasets, both downstream models. `exp05_transductive.csv` has, per
dataset and model, the mean over folds of the difference in the primary metric signed so that positive is better (AUC and R2 higher,
log loss lower) with an uncorrected fold bootstrap 95% CI over 5 folds (10,000 draws); per-fold values are in
`exp05_transductive_folds.csv`; claims `s5_transductive_*`. The kept feature lists are identical in 50 of 50 units. Besides the statistics there is one coding
difference: OpenFE's own transform gives a missing `Combine` pair the code -1, which is the level "-1.0" here, while the inductive arm keeps it
missing; LightGBM treats a level differently from missing, while CatBoost's downstream code turns missing into one level in both arms
(so on CatBoost a pair that is unseen on fit rows joins the trained missing level in the inductive arm). Check 5 (`openfe_ops.py`) compares
category codes (-1 for missing in both), so it does not see this difference. OpenFE's own transform returns its `Combine` features as `category` columns with float levels, which the
parquet cache turns into numbers; `afe.generators.openfe_transductive` renames the levels to strings, so these columns reach the model
as categoricals, like the inductive arm's (the first run lacked this and was repeated on 2026-11-20).

Result: every difference is small. LightGBM (10 datasets): 3 better by sign, 2 of them clearly (connect-4, gain +0.0007, i.e. log loss lower by that much; diamonds,
+0.00015 R2, lower CI bound 7e-6), 1 clearly worse (kick, -0.0032 AUC, CI [-0.0056, -0.0009], negative in 4 of 5 folds), 1 identical
(mfeat-fourier: row-wise operators). CatBoost (9 datasets): 3 better by sign, 1 clearly (albert, +0.0010 AUC), none clearly worse,
1 identical. The largest differences, house_prices_nominal on CatBoost (+0.010 R2), credit-g on CatBoost (-0.009 AUC) and cmc on
CatBoost (log loss higher by 0.006), have intervals that contain zero. Most of the kick result on LightGBM comes from the -1.0 level: with it set back to
missing, a refit from the stored caches (`kick_missing_level.py`, `exp05_transductive_kick_refit.csv`, claim `s5_transductive_lgbm_kick_missing_level`) gives
-0.0005 AUC, CI [-0.0016, +0.0004]. So the transductive transform gives no consistent gain or loss. The clear counts are descriptive: with 5 folds a percentile bootstrap calls a few of the 17 non-identical cells clear by chance, and
there are 4 here (3 without the kick level).

## Check 4 definitions

Unit: one generator unit (planted dataset, repeat 0, fold, arm). `ag_tabprep_keepcat` reuses the
`ag_tabprep` generator, so it has the same created columns.

**Oracle feature**: the true feature the target was built from, before standardization
(`afe.data.synthetic.planted`): `a / b` (ratio), `a * b` (product), `x` minus the true mean of its group
`g` (group_dev, 20 groups), the cell value of the 6 x 6 table of `(c1, c2)` with zero main effects
(cat_interaction). y = standardized oracle + normal noise with sd 0.5.

**Created columns**: the generator's `describe()["new_columns"]`, or, if it does not list them, the output
columns not in the input frame.

**Found**: on the outer test rows of the fold, as delivered by `transform`, at least one created column has
a relation >= 0.95 with the oracle feature, where relation is
- numeric column: |Pearson correlation| over the rows where the column is finite;
- categorical or non-numeric column: correlation ratio eta of the oracle on the column's levels (NaN level
  excluded), sqrt(between-level sum of squares / total sum of squares).

A column that is NaN on most test rows is judged on the rest; `best_relation` and `best_column` in
`exp05_planted.csv` show which column passed.

**Reach**: the unit's test R2 >= 0.99 x the oracle R2, where the oracle R2 is the same downstream model on
the same fold with the base view plus the oracle feature (`planted_oracle.py`). Weak where the downstream
model approximates the feature itself: base CatBoost reaches it on ratio in 73% of folds.
