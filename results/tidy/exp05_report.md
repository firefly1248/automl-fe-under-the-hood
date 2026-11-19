# exp05 leakage checks

Written by `experiments/exp05_leakage_checks/report.py`.

## 1. Permuted labels

Unit statuses: {'ok': 518, 'timeout': 12, 'not_applicable': 10}. Statistic per unit: binary AUC - 0.5; multiclass prior entropy - log loss; regression R2. Pass: the 95% bootstrap CI over units contains 0 or lies below it.

| arm                | model    | task       |   n_units |    mean |   ci_lo |   ci_hi | pass   |
|:-------------------|:---------|:-----------|----------:|--------:|--------:|--------:|:-------|
| ag_default         | catboost | binary     |        10 |  0.0167 | -0.0299 |  0.0664 | True   |
| ag_default         | catboost | multiclass |        10 | -0.0044 | -0.0102 |  0.0005 | True   |
| ag_default         | catboost | regression |        10 | -0.0283 | -0.0614 | -0.003  | True   |
| ag_default         | lgbm     | binary     |        10 |  0.0093 | -0.0126 |  0.0301 | True   |
| ag_default         | lgbm     | multiclass |        10 | -0.0029 | -0.0073 |  0.0009 | True   |
| ag_default         | lgbm     | regression |        10 | -0.0168 | -0.0323 | -0.0032 | True   |
| ag_tabprep         | catboost | binary     |        10 | -0.0058 | -0.0408 |  0.0294 | True   |
| ag_tabprep         | catboost | multiclass |        10 | -0.0106 | -0.0185 | -0.0035 | True   |
| ag_tabprep         | catboost | regression |        10 | -0.0207 | -0.0369 | -0.0029 | True   |
| ag_tabprep         | lgbm     | binary     |        10 | -0.0145 | -0.0309 | -0.0002 | True   |
| ag_tabprep         | lgbm     | multiclass |        10 | -0.0038 | -0.0071 | -0.0004 | True   |
| ag_tabprep         | lgbm     | regression |        10 | -0.0134 | -0.0336 |  0.0145 | True   |
| ag_tabprep_keepcat | catboost | binary     |        10 | -0.0001 | -0.0306 |  0.0317 | True   |
| ag_tabprep_keepcat | catboost | multiclass |        10 | -0.0133 | -0.0235 | -0.0044 | True   |
| ag_tabprep_keepcat | catboost | regression |        10 | -0.0233 | -0.0476 | -0.0017 | True   |
| ag_tabprep_keepcat | lgbm     | binary     |        10 | -0.008  | -0.0245 |  0.0087 | True   |
| ag_tabprep_keepcat | lgbm     | multiclass |        10 | -0.0038 | -0.0071 | -0.0004 | True   |
| ag_tabprep_keepcat | lgbm     | regression |        10 | -0.0116 | -0.0283 |  0.0033 | True   |
| autofeat           | catboost | binary     |         9 | -0.0141 | -0.0568 |  0.0271 | True   |
| autofeat           | catboost | multiclass |         5 | -0.0021 | -0.0107 |  0.0059 | True   |
| autofeat           | catboost | regression |        10 | -0.0253 | -0.0512 | -0.0004 | True   |
| autofeat           | lgbm     | binary     |         9 |  0.0033 | -0.0158 |  0.0179 | True   |
| autofeat           | lgbm     | multiclass |         5 | -0.0018 | -0.0091 |  0.0043 | True   |
| autofeat           | lgbm     | regression |        10 | -0.0145 | -0.0303 | -0.0009 | True   |
| base               | catboost | binary     |        10 |  0.0028 | -0.046  |  0.0537 | True   |
| base               | catboost | multiclass |        10 | -0.0036 | -0.0094 |  0.0018 | True   |
| base               | catboost | regression |        10 | -0.0321 | -0.0557 | -0.0101 | True   |
| base               | lgbm     | binary     |        10 |  0.0093 | -0.0126 |  0.0301 | True   |
| base               | lgbm     | multiclass |        10 | -0.0029 | -0.0073 |  0.0009 | True   |
| base               | lgbm     | regression |        10 | -0.0168 | -0.0323 | -0.0032 | True   |
| lama_adv           | catboost | binary     |        10 | -0.0125 | -0.0415 |  0.0135 | True   |
| lama_adv           | catboost | multiclass |        10 | -0.0047 | -0.0096 |  0      | True   |
| lama_adv           | catboost | regression |        10 | -0.0194 | -0.0435 |  0.0004 | True   |
| lama_adv           | lgbm     | binary     |        10 | -0.0047 | -0.0346 |  0.027  | True   |
| lama_adv           | lgbm     | multiclass |        10 | -0.0007 | -0.0041 |  0.0026 | True   |
| lama_adv           | lgbm     | regression |        10 | -0.0036 | -0.014  |  0.0084 | True   |
| manual_baseline    | catboost | binary     |        10 | -0.0189 | -0.0603 |  0.0256 | True   |
| manual_baseline    | catboost | multiclass |        10 | -0.0009 | -0.0038 |  0.0019 | True   |
| manual_baseline    | catboost | regression |        10 | -0.0227 | -0.0458 | -0.0032 | True   |
| manual_baseline    | lgbm     | binary     |        10 | -0.0134 | -0.0394 |  0.0121 | True   |
| manual_baseline    | lgbm     | multiclass |        10 | -0.0025 | -0.0042 | -0.0008 | True   |
| manual_baseline    | lgbm     | regression |        10 | -0.013  | -0.026  |  0.0008 | True   |
| mljar_golden       | catboost | binary     |        10 | -0.0077 | -0.0586 |  0.0396 | True   |
| mljar_golden       | catboost | multiclass |        10 | -0.0035 | -0.0069 | -0.0005 | True   |
| mljar_golden       | catboost | regression |         5 | -0.0281 | -0.0571 | -0.0049 | True   |
| mljar_golden       | lgbm     | binary     |        10 | -0.0076 | -0.0336 |  0.019  | True   |
| mljar_golden       | lgbm     | multiclass |        10 | -0.0025 | -0.0049 | -0.0001 | True   |
| mljar_golden       | lgbm     | regression |         5 | -0.0136 | -0.0299 | -0.0015 | True   |
| openfe             | catboost | binary     |        10 |  0.0079 | -0.0421 |  0.0557 | True   |
| openfe             | catboost | multiclass |        10 | -0.0067 | -0.0128 |  0.0002 | True   |
| openfe             | catboost | regression |        10 | -0.036  | -0.0849 |  0.0051 | True   |
| openfe             | lgbm     | binary     |        10 |  0.0049 | -0.0099 |  0.0178 | True   |
| openfe             | lgbm     | multiclass |        10 | -0.0055 | -0.0101 | -0.0016 | True   |
| openfe             | lgbm     | regression |        10 | -0.015  | -0.0285 | -0.0016 | True   |

## 2. Target-encoding delivery (exp04 pilot)

Gap = association with y on fit rows (as delivered) - on test rows; flag above 0.05 (AUC) or 0.1 (|corr|). kind raw: the arm's untouched numeric input columns, the gap from sampling noise alone.

| arm        | task       | kind   |   column_fold_instances |   flagged_share |   max_gap |   median_gap |
|:-----------|:-----------|:-------|------------------------:|----------------:|----------:|-------------:|
| ag_tabprep | binary     | raw    |                     330 |          0.0152 |    0.0688 |      -0.0007 |
| ag_tabprep | binary     | te     |                    1680 |          0.0345 |    0.1068 |      -0.0018 |
| ag_tabprep | multiclass | raw    |                     405 |          0.0346 |    0.1305 |      -0.0035 |
| ag_tabprep | multiclass | te     |                    2340 |          0.0188 |    0.1188 |      -0.0016 |
| ag_tabprep | regression | raw    |                     262 |          0.0802 |    0.1467 |      -0.0024 |
| ag_tabprep | regression | te     |                     735 |          0.0313 |    0.1575 |      -0.0085 |
| lama_adv   | binary     | raw    |                     344 |          0.0145 |    0.0688 |      -0.0005 |
| lama_adv   | binary     | te     |                     165 |          0.0848 |    0.1146 |       0      |
| lama_adv   | multiclass | raw    |                     568 |          0.0317 |    0.1305 |      -0.0009 |
| lama_adv   | regression | raw    |                     206 |          0.0728 |    0.1467 |      -0.0023 |
| lama_adv   | regression | te     |                     120 |          0.05   |    0.1352 |      -0.0073 |

## 3. openfe_transductive - openfe (exp04 pilot, repeat 0)

| dataset                       | task       |   fold | model    | metric   | status_transductive   | status_inductive   |   transductive |   inductive |    diff |
|:------------------------------|:-----------|-------:|:---------|:---------|:----------------------|:-------------------|---------------:|------------:|--------:|
| credit-g                      | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.7919 |      0.773  |  0.0189 |
| credit-g                      | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.7723 |      0.7711 |  0.0012 |
| credit-g                      | binary     |      1 | lgbm     | auc      | ok                    | ok                 |         0.7651 |      0.7815 | -0.0164 |
| credit-g                      | binary     |      1 | catboost | auc      | ok                    | ok                 |         0.7854 |      0.8019 | -0.0165 |
| credit-g                      | binary     |      2 | lgbm     | auc      | ok                    | ok                 |         0.717  |      0.7133 |  0.0037 |
| credit-g                      | binary     |      2 | catboost | auc      | ok                    | ok                 |         0.729  |      0.7079 |  0.0212 |
| credit-g                      | binary     |      3 | lgbm     | auc      | ok                    | ok                 |         0.7912 |      0.7881 |  0.0031 |
| credit-g                      | binary     |      3 | catboost | auc      | ok                    | ok                 |         0.7944 |      0.8265 | -0.0321 |
| credit-g                      | binary     |      4 | lgbm     | auc      | ok                    | ok                 |         0.7488 |      0.7604 | -0.0115 |
| credit-g                      | binary     |      4 | catboost | auc      | ok                    | ok                 |         0.7586 |      0.7454 |  0.0132 |
| numerai28.6                   | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.5233 |      0.5236 | -0.0004 |
| numerai28.6                   | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.5272 |      0.5243 |  0.003  |
| numerai28.6                   | binary     |      1 | lgbm     | auc      | ok                    | ok                 |         0.5268 |      0.5226 |  0.0041 |
| numerai28.6                   | binary     |      1 | catboost | auc      | ok                    | ok                 |         0.5223 |      0.5243 | -0.002  |
| numerai28.6                   | binary     |      2 | lgbm     | auc      | ok                    | ok                 |         0.5265 |      0.5303 | -0.0038 |
| numerai28.6                   | binary     |      2 | catboost | auc      | ok                    | ok                 |         0.5341 |      0.533  |  0.0011 |
| numerai28.6                   | binary     |      3 | lgbm     | auc      | ok                    | ok                 |         0.5274 |      0.5301 | -0.0027 |
| numerai28.6                   | binary     |      3 | catboost | auc      | ok                    | ok                 |         0.5272 |      0.5259 |  0.0013 |
| numerai28.6                   | binary     |      4 | lgbm     | auc      | ok                    | ok                 |         0.5311 |      0.5299 |  0.0012 |
| numerai28.6                   | binary     |      4 | catboost | auc      | ok                    | ok                 |         0.5349 |      0.5318 |  0.0031 |
| cmc                           | multiclass |      0 | lgbm     | logloss  | ok                    | ok                 |         0.9793 |      0.984  | -0.0047 |
| cmc                           | multiclass |      0 | catboost | logloss  | ok                    | ok                 |         0.9387 |      0.9406 | -0.002  |
| cmc                           | multiclass |      1 | lgbm     | logloss  | ok                    | ok                 |         0.9583 |      0.9527 |  0.0056 |
| cmc                           | multiclass |      1 | catboost | logloss  | ok                    | ok                 |         0.9288 |      0.9374 | -0.0085 |
| cmc                           | multiclass |      2 | lgbm     | logloss  | ok                    | ok                 |         0.9034 |      0.9069 | -0.0034 |
| cmc                           | multiclass |      2 | catboost | logloss  | ok                    | ok                 |         0.867  |      0.8552 |  0.0118 |
| cmc                           | multiclass |      3 | lgbm     | logloss  | ok                    | ok                 |         0.8734 |      0.8894 | -0.0161 |
| cmc                           | multiclass |      3 | catboost | logloss  | ok                    | ok                 |         0.8684 |      0.8633 |  0.005  |
| cmc                           | multiclass |      4 | lgbm     | logloss  | ok                    | ok                 |         0.9598 |      0.9458 |  0.014  |
| cmc                           | multiclass |      4 | catboost | logloss  | ok                    | ok                 |         0.9172 |      0.9039 |  0.0132 |
| connect-4                     | multiclass |      0 | lgbm     | logloss  | ok                    | ok                 |         0.3555 |      0.3376 |  0.0179 |
| connect-4                     | multiclass |      1 | lgbm     | logloss  | ok                    | ok                 |         0.3627 |      0.3403 |  0.0224 |
| connect-4                     | multiclass |      2 | lgbm     | logloss  | ok                    | ok                 |         0.3566 |      0.3424 |  0.0142 |
| connect-4                     | multiclass |      3 | lgbm     | logloss  | ok                    | ok                 |         0.353  |      0.3355 |  0.0175 |
| connect-4                     | multiclass |      4 | lgbm     | logloss  | ok                    | ok                 |         0.3462 |      0.3264 |  0.0198 |
| concrete_compressive_strength | regression |      0 | lgbm     | r2       | ok                    | ok                 |         0.9213 |      0.9271 | -0.0058 |
| concrete_compressive_strength | regression |      0 | catboost | r2       | ok                    | ok                 |         0.9379 |      0.9366 |  0.0013 |
| concrete_compressive_strength | regression |      1 | lgbm     | r2       | ok                    | ok                 |         0.9416 |      0.9392 |  0.0024 |
| concrete_compressive_strength | regression |      1 | catboost | r2       | ok                    | ok                 |         0.9415 |      0.9479 | -0.0065 |
| concrete_compressive_strength | regression |      2 | lgbm     | r2       | ok                    | ok                 |         0.9366 |      0.9395 | -0.0029 |
| concrete_compressive_strength | regression |      2 | catboost | r2       | ok                    | ok                 |         0.947  |      0.9448 |  0.0022 |
| concrete_compressive_strength | regression |      3 | lgbm     | r2       | ok                    | ok                 |         0.8809 |      0.8786 |  0.0024 |
| concrete_compressive_strength | regression |      3 | catboost | r2       | ok                    | ok                 |         0.8883 |      0.8987 | -0.0104 |
| concrete_compressive_strength | regression |      4 | lgbm     | r2       | ok                    | ok                 |         0.9308 |      0.9291 |  0.0017 |
| concrete_compressive_strength | regression |      4 | catboost | r2       | ok                    | ok                 |         0.9389 |      0.9402 | -0.0013 |
| diamonds                      | regression |      0 | lgbm     | r2       | ok                    | ok                 |         0.9825 |      0.9823 |  0.0002 |
| diamonds                      | regression |      0 | catboost | r2       | ok                    | ok                 |         0.9823 |      0.9825 | -0.0002 |
| diamonds                      | regression |      1 | lgbm     | r2       | ok                    | ok                 |         0.9831 |      0.9832 | -0.0002 |
| diamonds                      | regression |      1 | catboost | r2       | ok                    | ok                 |         0.9831 |      0.9833 | -0.0001 |
| diamonds                      | regression |      2 | lgbm     | r2       | ok                    | ok                 |         0.9827 |      0.9828 | -0.0001 |
| diamonds                      | regression |      2 | catboost | r2       | ok                    | ok                 |         0.9832 |      0.9831 |  0.0001 |
| diamonds                      | regression |      3 | lgbm     | r2       | ok                    | ok                 |         0.9832 |      0.9833 | -0.0001 |
| diamonds                      | regression |      3 | catboost | r2       | ok                    | ok                 |         0.9837 |      0.9832 |  0.0005 |
| diamonds                      | regression |      4 | lgbm     | r2       | ok                    | ok                 |         0.9823 |      0.9822 |  0.0001 |
| diamonds                      | regression |      4 | catboost | r2       | ok                    | ok                 |         0.9827 |      0.9828 | -0.0001 |
| kick                          | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.7794 |      0.7483 |  0.0311 |
| kick                          | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.7954 |      0.7951 |  0.0003 |
| kick                          | binary     |      1 | lgbm     | auc      | ok                    | ok                 |         0.7722 |      0.7524 |  0.0198 |
| kick                          | binary     |      1 | catboost | auc      | ok                    | ok                 |         0.7817 |      0.7833 | -0.0015 |
| kick                          | binary     |      2 | lgbm     | auc      | ok                    | ok                 |         0.7766 |      0.7552 |  0.0214 |
| kick                          | binary     |      2 | catboost | auc      | ok                    | ok                 |         0.793  |      0.7959 | -0.0029 |
| kick                          | binary     |      3 | lgbm     | auc      | ok                    | ok                 |         0.7744 |      0.7537 |  0.0207 |
| kick                          | binary     |      3 | catboost | auc      | ok                    | ok                 |         0.7827 |      0.7882 | -0.0055 |
| kick                          | binary     |      4 | lgbm     | auc      | ok                    | ok                 |         0.7817 |      0.7669 |  0.0148 |
| kick                          | binary     |      4 | catboost | auc      | ok                    | ok                 |         0.7945 |      0.7971 | -0.0026 |
| albert                        | binary     |      0 | lgbm     | auc      | ok                    | ok                 |         0.6993 |      0.7122 | -0.0129 |
| albert                        | binary     |      0 | catboost | auc      | ok                    | ok                 |         0.7112 |      0.73   | -0.0188 |
| albert                        | binary     |      1 | lgbm     | auc      | ok                    | ok                 |         0.6724 |      0.7067 | -0.0343 |
| albert                        | binary     |      1 | catboost | auc      | ok                    | ok                 |         0.6862 |      0.7223 | -0.036  |
| albert                        | binary     |      2 | lgbm     | auc      | ok                    | ok                 |         0.6771 |      0.7047 | -0.0277 |
| albert                        | binary     |      2 | catboost | auc      | ok                    | ok                 |         0.6882 |      0.7205 | -0.0323 |
| albert                        | binary     |      3 | lgbm     | auc      | ok                    | ok                 |         0.6724 |      0.7074 | -0.035  |
| albert                        | binary     |      3 | catboost | auc      | ok                    | ok                 |         0.6908 |      0.721  | -0.0301 |
| albert                        | binary     |      4 | lgbm     | auc      | ok                    | ok                 |         0.6916 |      0.7147 | -0.0231 |
| albert                        | binary     |      4 | catboost | auc      | ok                    | ok                 |         0.7062 |      0.7275 | -0.0213 |
| house_prices_nominal          | regression |      0 | lgbm     | r2       | ok                    | ok                 |         0.8045 |      0.8147 | -0.0102 |
| house_prices_nominal          | regression |      0 | catboost | r2       | ok                    | ok                 |         0.8511 |      0.8106 |  0.0405 |
| house_prices_nominal          | regression |      1 | lgbm     | r2       | ok                    | ok                 |         0.8007 |      0.8044 | -0.0037 |
| house_prices_nominal          | regression |      1 | catboost | r2       | ok                    | ok                 |         0.8092 |      0.8057 |  0.0035 |
| house_prices_nominal          | regression |      2 | lgbm     | r2       | ok                    | ok                 |         0.8841 |      0.8823 |  0.0018 |
| house_prices_nominal          | regression |      2 | catboost | r2       | ok                    | ok                 |         0.8767 |      0.8955 | -0.0188 |
| house_prices_nominal          | regression |      3 | lgbm     | r2       | ok                    | ok                 |         0.8461 |      0.8462 | -0.0002 |
| house_prices_nominal          | regression |      3 | catboost | r2       | ok                    | ok                 |         0.8479 |      0.855  | -0.0071 |
| house_prices_nominal          | regression |      4 | lgbm     | r2       | ok                    | ok                 |         0.9165 |      0.9192 | -0.0027 |
| house_prices_nominal          | regression |      4 | catboost | r2       | ok                    | ok                 |         0.9211 |      0.9203 |  0.0008 |
| mfeat-fourier                 | multiclass |      0 | lgbm     | logloss  | ok                    | ok                 |         0.5175 |      0.5175 |  0      |
| mfeat-fourier                 | multiclass |      0 | catboost | logloss  | ok                    | ok                 |         0.4501 |      0.4501 |  0      |
| mfeat-fourier                 | multiclass |      1 | lgbm     | logloss  | ok                    | ok                 |         0.4678 |      0.4678 |  0      |
| mfeat-fourier                 | multiclass |      1 | catboost | logloss  | ok                    | ok                 |         0.4313 |      0.4313 |  0      |
| mfeat-fourier                 | multiclass |      2 | lgbm     | logloss  | ok                    | ok                 |         0.4801 |      0.4801 |  0      |
| mfeat-fourier                 | multiclass |      2 | catboost | logloss  | ok                    | ok                 |         0.3744 |      0.3744 |  0      |
| mfeat-fourier                 | multiclass |      3 | lgbm     | logloss  | ok                    | ok                 |         0.4295 |      0.4295 |  0      |
| mfeat-fourier                 | multiclass |      3 | catboost | logloss  | ok                    | ok                 |         0.3866 |      0.3866 |  0      |
| mfeat-fourier                 | multiclass |      4 | lgbm     | logloss  | ok                    | ok                 |         0.388  |      0.388  |  0      |
| mfeat-fourier                 | multiclass |      4 | catboost | logloss  | ok                    | ok                 |         0.336  |      0.336  |  0      |

## 4. Planted signals

Share of units that reach the oracle R2 within 1% relative, and share of generator units where a created column has |corr| (numeric) or correlation ratio (categorical) >= 0.95 with the true feature on test rows.

| kind            | arm                | model    |   units |   reach |   found |
|:----------------|:-------------------|:---------|--------:|--------:|--------:|
| cat_interaction | ag_default         | catboost |      15 |   0.4   |   0     |
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
| group_dev       | ag_default         | catboost |      15 |   0.067 |   0     |
| group_dev       | ag_default         | lgbm     |      15 |   0.267 |   0     |
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
| group_dev       | openfe             | lgbm     |      15 |   0.733 |   1     |
| product         | ag_default         | catboost |      15 |   0.533 |   0     |
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
| ratio           | ag_default         | catboost |      15 |   0.733 |   0     |
| ratio           | ag_default         | lgbm     |      15 |   0.4   |   0     |
| ratio           | ag_tabprep         | catboost |      15 |   0.867 |   1     |
| ratio           | ag_tabprep         | lgbm     |      15 |   1     |   1     |
| ratio           | ag_tabprep_keepcat | catboost |      15 |   0.8   |   1     |
| ratio           | ag_tabprep_keepcat | lgbm     |      15 |   1     |   1     |
| ratio           | autofeat           | catboost |      15 |   0.933 |   1     |
| ratio           | autofeat           | lgbm     |      15 |   0.933 |   1     |
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
