# exp05: leakage checks (plan section 8)

```
make exp05                                   # permuted, transductive, planted runs
uv run --project envs/openfe python experiments/exp05_leakage_checks/openfe_ops.py       # check 5
uv run --project envs/core python experiments/exp05_leakage_checks/planted_oracle.py     # oracle for check 4
uv run --project envs/core python experiments/exp05_leakage_checks/report.py             # checks 1-4
```

Results: `results/tidy/exp05_report.md` and `exp05_*.csv`.

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
