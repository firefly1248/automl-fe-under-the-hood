# exp05 leakage checks

Written by `experiments/exp05_leakage_checks/report.py`.

## 1. Permuted labels

Unit statuses: {'ok': 518, 'timeout': 12, 'not_applicable': 10}. Statistic per unit: binary AUC - 0.5; multiclass prior entropy - log loss; regression R2. Pass: the 95% bootstrap CI over units contains 0 or lies below it.

| arm                | model    | task       |   n_units |    mean |   ci_lo |   ci_hi | pass   |
|:-------------------|:---------|:-----------|----------:|--------:|--------:|--------:|:-------|
| ag_default         | catboost | binary     |        10 |  0.0103 | -0.037  |  0.0558 | True   |
| ag_default         | catboost | multiclass |        10 | -0.008  | -0.0137 | -0.0028 | True   |
| ag_default         | catboost | regression |        10 | -0.0059 | -0.0259 |  0.0209 | True   |
| ag_default         | lgbm     | binary     |        10 |  0.0096 | -0.0194 |  0.0362 | True   |
| ag_default         | lgbm     | multiclass |        10 | -0.0025 | -0.0051 |  0      | True   |
| ag_default         | lgbm     | regression |        10 | -0.0022 | -0.0161 |  0.0181 | True   |
| ag_tabprep         | catboost | binary     |        10 | -0.0058 | -0.0408 |  0.0294 | True   |
| ag_tabprep         | catboost | multiclass |        10 | -0.0106 | -0.0185 | -0.0035 | True   |
| ag_tabprep         | catboost | regression |        10 | -0.0207 | -0.0369 | -0.0029 | True   |
| ag_tabprep         | lgbm     | binary     |        10 | -0.0145 | -0.0309 | -0.0002 | True   |
| ag_tabprep         | lgbm     | multiclass |        10 | -0.0038 | -0.0071 | -0.0004 | True   |
| ag_tabprep         | lgbm     | regression |        10 | -0.0134 | -0.0336 |  0.0145 | True   |
| ag_tabprep_keepcat | catboost | binary     |        10 | -0.0001 | -0.0306 |  0.0317 | True   |
| ag_tabprep_keepcat | catboost | multiclass |        10 | -0.0133 | -0.0235 | -0.0044 | True   |
| ag_tabprep_keepcat | catboost | regression |        10 | -0.0233 | -0.0476 | -0.0017 | True   |
| ag_tabprep_keepcat | lgbm     | binary     |        10 | -0.0166 | -0.0318 | -0.0029 | True   |
| ag_tabprep_keepcat | lgbm     | multiclass |        10 | -0.0037 | -0.007  | -0.0004 | True   |
| ag_tabprep_keepcat | lgbm     | regression |        10 | -0.021  | -0.0402 | -0.0005 | True   |
| autofeat           | catboost | binary     |         9 | -0.0077 | -0.0489 |  0.0293 | True   |
| autofeat           | catboost | multiclass |         5 | -0.001  | -0.0087 |  0.0059 | True   |
| autofeat           | catboost | regression |        10 | -0.0294 | -0.0534 | -0.0069 | True   |
| autofeat           | lgbm     | binary     |         9 | -0.0072 | -0.0426 |  0.0255 | True   |
| autofeat           | lgbm     | multiclass |         5 | -0.0011 | -0.0039 |  0.0016 | True   |
| autofeat           | lgbm     | regression |        10 | -0.0039 | -0.0186 |  0.0171 | True   |
| base               | catboost | binary     |        10 |  0.0028 | -0.046  |  0.0537 | True   |
| base               | catboost | multiclass |        10 | -0.0036 | -0.0094 |  0.0018 | True   |
| base               | catboost | regression |        10 | -0.0321 | -0.0557 | -0.0101 | True   |
| base               | lgbm     | binary     |        10 |  0.0096 | -0.0194 |  0.0362 | True   |
| base               | lgbm     | multiclass |        10 | -0.0025 | -0.0051 |  0      | True   |
| base               | lgbm     | regression |        10 | -0.0041 | -0.0188 |  0.0168 | True   |
| lama_adv           | catboost | binary     |        10 | -0.0125 | -0.0415 |  0.0135 | True   |
| lama_adv           | catboost | multiclass |        10 | -0.0047 | -0.0096 |  0      | True   |
| lama_adv           | catboost | regression |        10 | -0.0194 | -0.0435 |  0.0004 | True   |
| lama_adv           | lgbm     | binary     |        10 | -0.0047 | -0.0346 |  0.027  | True   |
| lama_adv           | lgbm     | multiclass |        10 | -0.0007 | -0.0041 |  0.0026 | True   |
| lama_adv           | lgbm     | regression |        10 | -0.0036 | -0.014  |  0.0084 | True   |
| manual_baseline    | catboost | binary     |        10 | -0.0189 | -0.0603 |  0.0256 | True   |
| manual_baseline    | catboost | multiclass |        10 | -0.0009 | -0.0038 |  0.0019 | True   |
| manual_baseline    | catboost | regression |        10 | -0.0227 | -0.0458 | -0.0032 | True   |
| manual_baseline    | lgbm     | binary     |        10 | -0.0135 | -0.0423 |  0.011  | True   |
| manual_baseline    | lgbm     | multiclass |        10 | -0.0024 | -0.0042 | -0.0007 | True   |
| manual_baseline    | lgbm     | regression |        10 | -0.0151 | -0.0369 |  0.0057 | True   |
| mljar_golden       | catboost | binary     |        10 | -0.0077 | -0.0586 |  0.0396 | True   |
| mljar_golden       | catboost | multiclass |        10 | -0.0035 | -0.0069 | -0.0005 | True   |
| mljar_golden       | catboost | regression |         5 | -0.0281 | -0.0571 | -0.0049 | True   |
| mljar_golden       | lgbm     | binary     |        10 | -0.001  | -0.0395 |  0.0401 | True   |
| mljar_golden       | lgbm     | multiclass |        10 | -0.002  | -0.0045 |  0.0001 | True   |
| mljar_golden       | lgbm     | regression |         5 | -0.0018 | -0.0096 |  0.0052 | True   |
| openfe             | catboost | binary     |        10 |  0.013  | -0.0292 |  0.057  | True   |
| openfe             | catboost | multiclass |        10 | -0.0069 | -0.0132 |  0.0001 | True   |
| openfe             | catboost | regression |        10 | -0.0395 | -0.0996 |  0.0082 | True   |
| openfe             | lgbm     | binary     |        10 |  0.0021 | -0.016  |  0.017  | True   |
| openfe             | lgbm     | multiclass |        10 | -0.0056 | -0.0102 | -0.0016 | True   |
| openfe             | lgbm     | regression |        10 | -0.0138 | -0.0288 | -0.0009 | True   |

## 2. Target-encoding delivery (exp04 pilot)

Gap = association with y on fit rows (as delivered) - on test rows; flag above 0.05 (AUC) or 0.1 (|corr|). kind raw: the arm's untouched numeric input columns, the gap from sampling noise alone.

| arm        | task       | kind   |   columns |   flagged_share |   max_gap |   median_gap |
|:-----------|:-----------|:-------|----------:|----------------:|----------:|-------------:|
| ag_tabprep | binary     | raw    |       330 |          0.0152 |    0.0688 |      -0.0007 |
| ag_tabprep | binary     | te     |      1680 |          0.0345 |    0.1068 |      -0.0018 |
| ag_tabprep | multiclass | raw    |       405 |          0.0346 |    0.1305 |      -0.0035 |
| ag_tabprep | multiclass | te     |      2340 |          0.0188 |    0.1188 |      -0.0016 |
| ag_tabprep | regression | raw    |       262 |          0.0802 |    0.1467 |      -0.0024 |
| ag_tabprep | regression | te     |       735 |          0.0313 |    0.1575 |      -0.0085 |
| lama_adv   | binary     | raw    |       344 |          0.0145 |    0.0688 |      -0.0005 |
| lama_adv   | binary     | te     |       165 |          0.0848 |    0.1146 |       0      |
| lama_adv   | multiclass | raw    |       568 |          0.0317 |    0.1305 |      -0.0009 |
| lama_adv   | regression | raw    |       206 |          0.0728 |    0.1467 |      -0.0023 |
| lama_adv   | regression | te     |       120 |          0.05   |    0.1352 |      -0.0073 |

## 3. openfe_transductive - openfe (exp04 pilot, fold 0)

| dataset                       | task       |   fold | model    | metric   | status_transductive   | status_inductive   |   transductive |   inductive |    diff |
|:------------------------------|:-----------|-------:|:---------|:---------|:----------------------|:-------------------|---------------:|------------:|--------:|
| credit-g                      | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.7861 |      0.7914 | -0.0054 |
| credit-g                      | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.7723 |      0.7713 |  0.001  |
| numerai28.6                   | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.5233 |      0.5236 | -0.0004 |
| numerai28.6                   | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.5272 |      0.5243 |  0.003  |
| cmc                           | multiclass |      0 | lgbm     | logloss  | ok                    | ok                 |         0.9853 |      0.9791 |  0.0062 |
| cmc                           | multiclass |      0 | catboost | logloss  | ok                    | ok                 |         0.9387 |      0.9406 | -0.0019 |
| connect-4                     | multiclass |      0 | lgbm     | logloss  | ok                    | ok                 |         0.3569 |      0.3393 |  0.0176 |
| concrete_compressive_strength | regression |      0 | lgbm     | r2       | ok                    | ok                 |         0.9213 |      0.9262 | -0.0048 |
| concrete_compressive_strength | regression |      0 | catboost | r2       | ok                    | ok                 |         0.9379 |      0.9365 |  0.0014 |
| diamonds                      | regression |      0 | lgbm     | r2       | ok                    | ok                 |         0.9826 |      0.9824 |  0.0002 |
| diamonds                      | regression |      0 | catboost | r2       | ok                    | ok                 |         0.9823 |      0.9824 | -0.0001 |
| kick                          | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.795  |      0.7499 |  0.0451 |
| kick                          | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.7954 |      0.7951 |  0.0003 |
| albert                        | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.7    |      0.7122 | -0.0122 |
| albert                        | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.7112 |      0.7309 | -0.0197 |
| house_prices_nominal          | regression |      0 | lgbm     | r2       | ok                    | ok                 |         0.8089 |      0.8129 | -0.004  |
| house_prices_nominal          | regression |      0 | catboost | r2       | ok                    | ok                 |         0.8511 |      0.8055 |  0.0456 |
| mfeat-fourier                 | multiclass |      0 | lgbm     | logloss  | ok                    | ok                 |         0.5175 |      0.5175 |  0      |
| mfeat-fourier                 | multiclass |      0 | catboost | logloss  | ok                    | ok                 |         0.4501 |      0.4501 |  0      |

## 4. Planted signals

Share of units that reach the oracle R2 within 1% relative, and share of generator units where a created column has |corr| (numeric) or correlation ratio (categorical) >= 0.95 with the true feature on test rows.

| kind            | arm                | model    |   units |   reach |   found |
|:----------------|:-------------------|:---------|--------:|--------:|--------:|
| cat_interaction | ag_default         | catboost |      15 |   0     |   0     |
| cat_interaction | ag_default         | lgbm     |      15 |   0     |   0     |
| cat_interaction | ag_tabprep         | catboost |      15 |   0.533 |   1     |
| cat_interaction | ag_tabprep         | lgbm     |      15 |   0.8   |   1     |
| cat_interaction | ag_tabprep_keepcat | catboost |      15 |   0.733 |   1     |
| cat_interaction | ag_tabprep_keepcat | lgbm     |      15 |   0.8   |   1     |
| cat_interaction | autofeat           | catboost |      15 |   0.533 |   0     |
| cat_interaction | autofeat           | lgbm     |      15 |   0     |   0     |
| cat_interaction | base               | catboost |      15 |   0.4   |   0     |
| cat_interaction | base               | lgbm     |      15 |   0     |   0     |
| cat_interaction | lama_adv           | catboost |      15 |   0.867 |   1     |
| cat_interaction | lama_adv           | lgbm     |      15 |   1     |   1     |
| cat_interaction | manual_baseline    | catboost |      15 |   0.733 |   0     |
| cat_interaction | manual_baseline    | lgbm     |      15 |   0     |   0     |
| cat_interaction | mljar_golden       | catboost |      15 |   0.267 |   0     |
| cat_interaction | mljar_golden       | lgbm     |      15 |   0     |   0     |
| cat_interaction | openfe             | catboost |      15 |   0.6   |   0.667 |
| cat_interaction | openfe             | lgbm     |      15 |   0.867 |   0.667 |
| group_dev       | ag_default         | catboost |      15 |   0     |   0     |
| group_dev       | ag_default         | lgbm     |      15 |   0     |   0     |
| group_dev       | ag_tabprep         | catboost |      15 |   0     |   0     |
| group_dev       | ag_tabprep         | lgbm     |      15 |   0     |   0     |
| group_dev       | ag_tabprep_keepcat | catboost |      15 |   0     |   0     |
| group_dev       | ag_tabprep_keepcat | lgbm     |      15 |   0.067 |   0     |
| group_dev       | autofeat           | catboost |      15 |   0     |   0     |
| group_dev       | autofeat           | lgbm     |      15 |   0.2   |   0     |
| group_dev       | base               | catboost |      15 |   0.067 |   0     |
| group_dev       | base               | lgbm     |      15 |   0.267 |   0     |
| group_dev       | lama_adv           | catboost |      15 |   0     |   0     |
| group_dev       | lama_adv           | lgbm     |      15 |   0     |   0     |
| group_dev       | manual_baseline    | catboost |      15 |   0.467 |   0.467 |
| group_dev       | manual_baseline    | lgbm     |      15 |   0.533 |   0.467 |
| group_dev       | mljar_golden       | catboost |      15 |   0     |   0     |
| group_dev       | mljar_golden       | lgbm     |      15 |   0.067 |   0     |
| group_dev       | openfe             | catboost |      15 |   0.2   |   1     |
| group_dev       | openfe             | lgbm     |      15 |   0.8   |   1     |
| product         | ag_default         | catboost |      15 |   0.267 |   0     |
| product         | ag_default         | lgbm     |      15 |   0     |   0     |
| product         | ag_tabprep         | catboost |      15 |   0.733 |   1     |
| product         | ag_tabprep         | lgbm     |      15 |   1     |   1     |
| product         | ag_tabprep_keepcat | catboost |      15 |   0.533 |   1     |
| product         | ag_tabprep_keepcat | lgbm     |      15 |   1     |   1     |
| product         | autofeat           | catboost |      15 |   1     |   1     |
| product         | autofeat           | lgbm     |      15 |   1     |   1     |
| product         | base               | catboost |      15 |   0.533 |   0     |
| product         | base               | lgbm     |      15 |   0     |   0     |
| product         | lama_adv           | catboost |      15 |   0.133 |   0     |
| product         | lama_adv           | lgbm     |      15 |   0     |   0     |
| product         | manual_baseline    | catboost |      15 |   0.733 |   1     |
| product         | manual_baseline    | lgbm     |      15 |   1     |   1     |
| product         | mljar_golden       | catboost |      15 |   1     |   1     |
| product         | mljar_golden       | lgbm     |      15 |   1     |   1     |
| product         | openfe             | catboost |      15 |   1     |   1     |
| product         | openfe             | lgbm     |      15 |   1     |   1     |
| ratio           | ag_default         | catboost |      15 |   0.6   |   0     |
| ratio           | ag_default         | lgbm     |      15 |   0.333 |   0     |
| ratio           | ag_tabprep         | catboost |      15 |   0.867 |   1     |
| ratio           | ag_tabprep         | lgbm     |      15 |   1     |   1     |
| ratio           | ag_tabprep_keepcat | catboost |      15 |   0.8   |   1     |
| ratio           | ag_tabprep_keepcat | lgbm     |      15 |   1     |   1     |
| ratio           | autofeat           | catboost |      15 |   0.867 |   1     |
| ratio           | autofeat           | lgbm     |      15 |   1     |   1     |
| ratio           | base               | catboost |      15 |   0.733 |   0     |
| ratio           | base               | lgbm     |      15 |   0.4   |   0     |
| ratio           | lama_adv           | catboost |      15 |   0.6   |   0     |
| ratio           | lama_adv           | lgbm     |      15 |   0.4   |   0     |
| ratio           | manual_baseline    | catboost |      15 |   0.867 |   1     |
| ratio           | manual_baseline    | lgbm     |      15 |   0.933 |   1     |
| ratio           | mljar_golden       | catboost |      15 |   1     |   1     |
| ratio           | mljar_golden       | lgbm     |      15 |   1     |   1     |
| ratio           | openfe             | catboost |      15 |   0.867 |   1     |
| ratio           | openfe             | lgbm     |      15 |   1     |   1     |

## 5. OpenFE operator equivalence (fit rows)

| operator           |   n_features |   max_abs_diff |   nan_mismatch |
|:-------------------|-------------:|---------------:|---------------:|
| *                  |            6 |              0 |              0 |
| +                  |            6 |              0 |              0 |
| -                  |            6 |              0 |              0 |
| /                  |            6 |              0 |              0 |
| Combine            |            6 |              0 |              0 |
| CombineThenFreq    |            6 |              0 |              0 |
| GroupByThenMax     |           15 |              0 |              0 |
| GroupByThenMean    |           15 |              0 |              0 |
| GroupByThenMedian  |           15 |              0 |              0 |
| GroupByThenMin     |           15 |              0 |              0 |
| GroupByThenNUnique |           12 |              0 |              0 |
| GroupByThenRank    |           15 |              0 |              0 |
| GroupByThenStd     |           15 |              0 |              0 |
| abs                |            4 |              0 |              0 |
| freq               |            8 |              0 |              0 |
| log                |            4 |              0 |              0 |
| max                |            6 |              0 |              0 |
| min                |            6 |              0 |              0 |
| residual           |            4 |              0 |              0 |
| round              |            4 |              0 |              0 |
| sigmoid            |            4 |              0 |              0 |
| sqrt               |            4 |              0 |              0 |
| square             |            4 |              0 |              0 |
