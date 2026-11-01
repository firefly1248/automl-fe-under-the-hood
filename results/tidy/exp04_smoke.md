# exp04 smoke report

## Generator units

| dataset   | arm             | stage   | status   |   n_out |   wall_s |   peak_mb | error   |
|:----------|:----------------|:--------|:---------|--------:|---------:|----------:|:--------|
| Moneyball | ag_default      | gen     | ok       |      14 |     1.52 |     222.2 |         |
| cmc       | ag_default      | gen     | ok       |       9 |     1.51 |     216.8 |         |
| credit-g  | ag_default      | gen     | ok       |      20 |     1.75 |     229   |         |
| Moneyball | ag_tabprep      | gen     | ok       |    1981 |     1.97 |     326.8 |         |
| cmc       | ag_tabprep      | gen     | ok       |      52 |     1.31 |     240.7 |         |
| credit-g  | ag_tabprep      | gen     | ok       |     962 |     2.19 |     291.5 |         |
| Moneyball | autofeat        | gen     | ok       |       4 |     4.77 |     421.7 |         |
| cmc       | autofeat        | gen     | ok       |       3 |    27.48 |     361.8 |         |
| credit-g  | autofeat        | gen     | ok       |       8 |    71.67 |     390.3 |         |
| Moneyball | base            | gen     | ok       |       0 |     0.43 |     105.7 |         |
| cmc       | base            | gen     | ok       |       0 |     0.44 |     106.3 |         |
| credit-g  | base            | gen     | ok       |       0 |     0.43 |     111.7 |         |
| Moneyball | lama_adv        | gen     | ok       |      13 |     3.24 |     467.5 |         |
| cmc       | lama_adv        | gen     | ok       |      12 |     3.02 |     473.2 |         |
| credit-g  | lama_adv        | gen     | ok       |       4 |     3.51 |     460.4 |         |
| Moneyball | manual_baseline | gen     | ok       |      46 |     1.3  |     223.2 |         |
| cmc       | manual_baseline | gen     | ok       |      14 |     1.08 |     213.7 |         |
| credit-g  | manual_baseline | gen     | ok       |      53 |     1.3  |     222.5 |         |
| Moneyball | mljar_golden    | gen     | ok       |      10 |     2.62 |     330.6 |         |
| cmc       | mljar_golden    | gen     | ok       |       5 |     2.38 |     331.3 |         |
| credit-g  | mljar_golden    | gen     | ok       |      10 |     2.6  |     331.8 |         |
| Moneyball | openfe          | gen     | ok       |     208 |    10.39 |     495.8 |         |
| cmc       | openfe          | gen     | ok       |      76 |     7.55 |     492.2 |         |
| credit-g  | openfe          | gen     | ok       |     166 |    11.24 |     505.1 |         |

## Fit units (test metric: AUC binary, log loss multiclass, RMSE regression)

| dataset   | arm                | model    | status   | metric   |     value |   n_features |   wall_s |   peak_mb |      base |
|:----------|:-------------------|:---------|:---------|:---------|----------:|-------------:|---------:|----------:|----------:|
| Moneyball | ag_default         | catboost | ok       | rmse     | 21.7966   |           14 |     1.52 |     233.4 | 21.6748   |
| Moneyball | ag_default         | lgbm     | ok       | rmse     | 22.3055   |           14 |     1.07 |     213.7 | 22.03     |
| Moneyball | ag_tabprep         | catboost | ok       | rmse     | 21.3411   |         1981 |    82.75 |     797.9 | 21.6748   |
| Moneyball | ag_tabprep         | lgbm     | ok       | rmse     | 21.4119   |         1981 |    29.62 |     480.3 | 22.03     |
| Moneyball | ag_tabprep_keepcat | catboost | ok       | rmse     | 21.5987   |         1985 |    75.04 |     837.1 | 21.6748   |
| Moneyball | ag_tabprep_keepcat | lgbm     | ok       | rmse     | 21.6128   |         1985 |    14.68 |     477.2 | 22.03     |
| Moneyball | autofeat           | catboost | ok       | rmse     | 20.7659   |           18 |     1.74 |     240.3 | 21.6748   |
| Moneyball | autofeat           | lgbm     | ok       | rmse     | 21.5619   |           18 |     1.08 |     208.8 | 22.03     |
| Moneyball | base               | catboost | ok       | rmse     | 21.6748   |           14 |     1.94 |     247.7 | 21.6748   |
| Moneyball | base               | lgbm     | ok       | rmse     | 22.03     |           14 |     1.09 |     215.1 | 22.03     |
| Moneyball | lama_adv           | catboost | ok       | rmse     | 20.825    |           13 |     1.29 |     226.7 | 21.6748   |
| Moneyball | lama_adv           | lgbm     | ok       | rmse     | 22.8089   |           13 |     1.09 |     214.1 | 22.03     |
| Moneyball | manual_baseline    | catboost | ok       | rmse     | 21.145    |           60 |     3.89 |     256.4 | 21.6748   |
| Moneyball | manual_baseline    | lgbm     | ok       | rmse     | 21.4047   |           60 |     1.52 |     223   | 22.03     |
| Moneyball | mljar_golden       | catboost | ok       | rmse     | 21.1286   |           24 |     2.17 |     235.6 | 21.6748   |
| Moneyball | mljar_golden       | lgbm     | ok       | rmse     | 21.1837   |           24 |     1.12 |     215.7 | 22.03     |
| Moneyball | openfe             | catboost | ok       | rmse     | 25.1812   |          222 |     5.86 |     284.4 | 21.6748   |
| Moneyball | openfe             | lgbm     | ok       | rmse     | 21.1733   |          222 |     1.75 |     238   | 22.03     |
| cmc       | ag_default         | catboost | ok       | logloss  |  0.934565 |            9 |     1.29 |     228.8 |  0.957806 |
| cmc       | ag_default         | lgbm     | ok       | logloss  |  0.962686 |            9 |     1.08 |     212.8 |  0.962686 |
| cmc       | ag_tabprep         | catboost | ok       | logloss  |  0.959046 |           52 |     2.6  |     244.6 |  0.957806 |
| cmc       | ag_tabprep         | lgbm     | ok       | logloss  |  0.990561 |           52 |     1.3  |     219.4 |  0.962686 |
| cmc       | ag_tabprep_keepcat | catboost | ok       | logloss  |  0.961549 |           56 |     3.67 |     269.5 |  0.957806 |
| cmc       | ag_tabprep_keepcat | lgbm     | ok       | logloss  |  0.990706 |           56 |     1.31 |     218.8 |  0.962686 |
| cmc       | autofeat           | catboost | ok       | logloss  |  0.924628 |           12 |     2.38 |     281.8 |  0.957806 |
| cmc       | autofeat           | lgbm     | ok       | logloss  |  0.972937 |           12 |     1.3  |     219.5 |  0.962686 |
| cmc       | base               | catboost | ok       | logloss  |  0.957806 |            9 |     2.38 |     254   |  0.957806 |
| cmc       | base               | lgbm     | ok       | logloss  |  0.962686 |            9 |     1.1  |     210.5 |  0.962686 |
| cmc       | lama_adv           | catboost | ok       | logloss  |  0.966945 |           12 |     1.31 |     229.4 |  0.957806 |
| cmc       | lama_adv           | lgbm     | ok       | logloss  |  0.973229 |           12 |     1.09 |     210.6 |  0.962686 |
| cmc       | manual_baseline    | catboost | ok       | logloss  |  0.944071 |           23 |     2.59 |     270.6 |  0.957806 |
| cmc       | manual_baseline    | lgbm     | ok       | logloss  |  0.963228 |           23 |     1.31 |     215.6 |  0.962686 |
| cmc       | mljar_golden       | catboost | ok       | logloss  |  0.931859 |           14 |     2.37 |     277.1 |  0.957806 |
| cmc       | mljar_golden       | lgbm     | ok       | logloss  |  0.94711  |           14 |     1.09 |     214.7 |  0.962686 |
| cmc       | openfe             | catboost | ok       | logloss  |  0.940572 |           85 |     3.46 |     282.5 |  0.957806 |
| cmc       | openfe             | lgbm     | ok       | logloss  |  0.979098 |           85 |     1.3  |     222.8 |  0.962686 |
| credit-g  | ag_default         | catboost | ok       | auc      |  0.837143 |           20 |     1.31 |     230.5 |  0.793929 |
| credit-g  | ag_default         | lgbm     | ok       | auc      |  0.81369  |           20 |     1.09 |     213.6 |  0.809881 |
| credit-g  | ag_tabprep         | catboost | ok       | auc      |  0.809643 |          962 |    24.93 |     444.5 |  0.793929 |
| credit-g  | ag_tabprep         | lgbm     | ok       | auc      |  0.804286 |          962 |     4.17 |     336.1 |  0.809881 |
| credit-g  | ag_tabprep_keepcat | catboost | ok       | auc      |  0.810714 |          973 |    25.49 |     592   |  0.793929 |
| credit-g  | ag_tabprep_keepcat | lgbm     | ok       | auc      |  0.807738 |          973 |     4.13 |     336.6 |  0.809881 |
| credit-g  | autofeat           | catboost | ok       | auc      |  0.808095 |           28 |     3.04 |     269.3 |  0.793929 |
| credit-g  | autofeat           | lgbm     | ok       | auc      |  0.804286 |           28 |     1.08 |     216.4 |  0.809881 |
| credit-g  | base               | catboost | ok       | auc      |  0.793929 |           20 |     2.2  |     262   |  0.793929 |
| credit-g  | base               | lgbm     | ok       | auc      |  0.809881 |           20 |     1.33 |     217.9 |  0.809881 |
| credit-g  | lama_adv           | catboost | ok       | auc      |  0.663631 |            4 |     1.09 |     222.7 |  0.793929 |
| credit-g  | lama_adv           | lgbm     | ok       | auc      |  0.707262 |            4 |     1.11 |     196.5 |  0.809881 |
| credit-g  | manual_baseline    | catboost | ok       | auc      |  0.804881 |           73 |     3.45 |     267.1 |  0.793929 |
| credit-g  | manual_baseline    | lgbm     | ok       | auc      |  0.813452 |           73 |     1.29 |     224.1 |  0.809881 |
| credit-g  | mljar_golden       | catboost | ok       | auc      |  0.800595 |           30 |     2.39 |     262.6 |  0.793929 |
| credit-g  | mljar_golden       | lgbm     | ok       | auc      |  0.80631  |           30 |     1.09 |     214.8 |  0.809881 |
| credit-g  | openfe             | catboost | ok       | auc      |  0.77131  |          186 |     3.69 |     276.5 |  0.793929 |
| credit-g  | openfe             | lgbm     | ok       | auc      |  0.791429 |          186 |     1.28 |     229.3 |  0.809881 |

## Test metric by dataset and arm

|                                     |   catboost |      lgbm |
|:------------------------------------|-----------:|----------:|
| ('Moneyball', 'ag_default')         |  21.7966   | 22.3055   |
| ('Moneyball', 'ag_tabprep')         |  21.3411   | 21.4119   |
| ('Moneyball', 'ag_tabprep_keepcat') |  21.5987   | 21.6128   |
| ('Moneyball', 'autofeat')           |  20.7659   | 21.5619   |
| ('Moneyball', 'base')               |  21.6748   | 22.03     |
| ('Moneyball', 'lama_adv')           |  20.825    | 22.8089   |
| ('Moneyball', 'manual_baseline')    |  21.145    | 21.4047   |
| ('Moneyball', 'mljar_golden')       |  21.1286   | 21.1837   |
| ('Moneyball', 'openfe')             |  25.1812   | 21.1733   |
| ('cmc', 'ag_default')               |   0.934565 |  0.962686 |
| ('cmc', 'ag_tabprep')               |   0.959046 |  0.990561 |
| ('cmc', 'ag_tabprep_keepcat')       |   0.961549 |  0.990706 |
| ('cmc', 'autofeat')                 |   0.924628 |  0.972937 |
| ('cmc', 'base')                     |   0.957806 |  0.962686 |
| ('cmc', 'lama_adv')                 |   0.966945 |  0.973229 |
| ('cmc', 'manual_baseline')          |   0.944071 |  0.963228 |
| ('cmc', 'mljar_golden')             |   0.931859 |  0.94711  |
| ('cmc', 'openfe')                   |   0.940572 |  0.979098 |
| ('credit-g', 'ag_default')          |   0.837143 |  0.81369  |
| ('credit-g', 'ag_tabprep')          |   0.809643 |  0.804286 |
| ('credit-g', 'ag_tabprep_keepcat')  |   0.810714 |  0.807738 |
| ('credit-g', 'autofeat')            |   0.808095 |  0.804286 |
| ('credit-g', 'base')                |   0.793929 |  0.809881 |
| ('credit-g', 'lama_adv')            |   0.663631 |  0.707262 |
| ('credit-g', 'manual_baseline')     |   0.804881 |  0.813452 |
| ('credit-g', 'mljar_golden')        |   0.800595 |  0.80631  |
| ('credit-g', 'openfe')              |   0.77131  |  0.791429 |
