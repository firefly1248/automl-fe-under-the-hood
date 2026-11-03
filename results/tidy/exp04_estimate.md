# exp04 full-run estimate

From 10 pilot datasets; 50 primary datasets; jobs = 3. Unit-hours per fold over all arms and stages: 20; the pilot took 30 over 5 folds. Ten pilot points per arm and stage, so a rough budget: autofeat time depends on more than rows x columns, and every primary dataset it reaches near the cap adds up to an hour per fold.

| splits | unit-hours, all | wall clock, all | unit-hours left after the pilot | wall clock left |
|---|---|---|---|---|
| 5 | 102 | 34 h | 73 | 24 h |
| 10 | 205 | 68 h | 176 | 59 h |

Per arm and stage (unit-hours per fold, summed over primary datasets):

| arm                | stage    |   pilot_h |   unit_h_per_fold |   unit_h_per_fold_not_in_pilot |   datasets_near_cap |
|:-------------------|:---------|----------:|------------------:|-------------------------------:|--------------------:|
| autofeat           | gen      |     15.25 |              6.69 |                           3.64 |                   4 |
| ag_tabprep_keepcat | catboost |      3.74 |              4.05 |                           3.37 |                   0 |
| ag_tabprep         | catboost |      3.48 |              3.8  |                           3.15 |                   0 |
| openfe             | catboost |      1.85 |              1.84 |                           1.47 |                   0 |
| openfe             | gen      |      2.18 |              1.1  |                           0.67 |                   0 |
| ag_tabprep_keepcat | lgbm     |      0.69 |              0.6  |                           0.46 |                   0 |
| ag_tabprep         | lgbm     |      0.67 |              0.53 |                           0.4  |                   0 |
| manual_baseline    | catboost |      0.36 |              0.32 |                           0.26 |                   0 |
| mljar_golden       | catboost |      0.31 |              0.31 |                           0.25 |                   0 |
| base               | catboost |      0.27 |              0.25 |                           0.2  |                   0 |
| openfe             | lgbm     |      0.33 |              0.21 |                           0.15 |                   0 |
| ag_default         | catboost |      0.11 |              0.12 |                           0.1  |                   0 |
| lama_adv           | catboost |      0.1  |              0.12 |                           0.1  |                   0 |
| autofeat           | catboost |      0.05 |              0.11 |                           0.09 |                   0 |
| mljar_golden       | gen      |      0.07 |              0.07 |                           0.06 |                   0 |
| lama_adv           | gen      |      0.07 |              0.06 |                           0.05 |                   0 |
| ag_tabprep         | gen      |      0.06 |              0.04 |                           0.03 |                   0 |
| manual_baseline    | lgbm     |      0.06 |              0.04 |                           0.03 |                   0 |
| lama_adv           | lgbm     |      0.06 |              0.04 |                           0.02 |                   0 |
| mljar_golden       | lgbm     |      0.03 |              0.03 |                           0.03 |                   0 |
| base               | lgbm     |      0.05 |              0.03 |                           0.02 |                   0 |
| ag_default         | lgbm     |      0.05 |              0.03 |                           0.02 |                   0 |
| autofeat           | lgbm     |      0.01 |              0.02 |                           0.02 |                   0 |
| manual_baseline    | gen      |      0.03 |              0.02 |                           0.02 |                   0 |
| ag_default         | gen      |      0.02 |              0.02 |                           0.01 |                   0 |
| base               | gen      |      0.01 |              0.01 |                           0    |                   0 |
