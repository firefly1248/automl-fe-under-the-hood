# exp04 smoke report

## Generator units

| dataset   | arm             | stage   | status   |   n_out |   wall_s |   peak_mb | error   |
|:----------|:----------------|:--------|:---------|--------:|---------:|----------:|:--------|
| Moneyball | ag_default      | gen     | ok       |      14 |     1.52 |     221.9 |         |
| cmc       | ag_default      | gen     | ok       |       9 |     1.52 |     223.7 |         |
| credit-g  | ag_default      | gen     | ok       |      20 |     1.53 |     228.8 |         |
| Moneyball | ag_tabprep      | gen     | ok       |    1981 |     1.94 |     329.5 |         |
| cmc       | ag_tabprep      | gen     | ok       |      52 |     1.3  |     240.5 |         |
| credit-g  | ag_tabprep      | gen     | ok       |     962 |     1.98 |     292.8 |         |
| Moneyball | autofeat        | gen     | ok       |       4 |     4.89 |     392.6 |         |
| cmc       | autofeat        | gen     | ok       |       2 |    27.6  |     353.6 |         |
| credit-g  | autofeat        | gen     | ok       |       5 |    73.66 |     371.2 |         |
| Moneyball | base            | gen     | ok       |       0 |     0.43 |     108.1 |         |
| cmc       | base            | gen     | ok       |       0 |     0.44 |     107.9 |         |
| credit-g  | base            | gen     | ok       |       0 |     0.44 |      87.7 |         |
| Moneyball | lama_adv        | gen     | ok       |      19 |     3.23 |     463.1 |         |
| cmc       | lama_adv        | gen     | ok       |      12 |     3.21 |     464.6 |         |
| credit-g  | lama_adv        | gen     | ok       |      28 |     3.46 |     457.3 |         |
| Moneyball | manual_baseline | gen     | ok       |      46 |     1.29 |     224.6 |         |
| cmc       | manual_baseline | gen     | ok       |      14 |     1.07 |     209.9 |         |
| credit-g  | manual_baseline | gen     | ok       |      53 |     1.3  |     222.1 |         |
| Moneyball | mljar_golden    | gen     | ok       |      10 |     2.57 |     330.6 |         |
| cmc       | mljar_golden    | gen     | ok       |       5 |     2.38 |     330.4 |         |
| credit-g  | mljar_golden    | gen     | ok       |      10 |     2.8  |     328.3 |         |
| Moneyball | openfe          | gen     | ok       |     208 |    10.64 |     461.9 |         |
| cmc       | openfe          | gen     | ok       |      76 |     8.21 |     492.6 |         |
| credit-g  | openfe          | gen     | ok       |     166 |    11.4  |     501   |         |

## Fit units (test metric: AUC binary, log loss multiclass, RMSE regression)

| dataset   | arm                | model    | status   | metric   |     value |   n_features |   wall_s |   peak_mb |      base |
|:----------|:-------------------|:---------|:---------|:---------|----------:|-------------:|---------:|----------:|----------:|
| Moneyball | ag_default         | catboost | ok       | rmse     | 21.7966   |           14 |     1.51 |     233.6 | 21.6748   |
| Moneyball | ag_default         | lgbm     | ok       | rmse     | 22.3055   |           14 |     1.08 |     213.6 | 22.03     |
| Moneyball | ag_tabprep         | catboost | ok       | rmse     | 21.3411   |         1981 |    82.73 |     801.9 | 21.6748   |
| Moneyball | ag_tabprep         | lgbm     | ok       | rmse     | 21.4119   |         1981 |    29.74 |     486.3 | 22.03     |
| Moneyball | ag_tabprep_keepcat | catboost | ok       | rmse     | 21.5987   |         1985 |    74.92 |     834.1 | 21.6748   |
| Moneyball | ag_tabprep_keepcat | lgbm     | ok       | rmse     | 21.6128   |         1985 |    14.81 |     478.1 | 22.03     |
| Moneyball | autofeat           | catboost | ok       | rmse     | 20.5246   |           18 |     1.71 |     240.2 | 21.6748   |
| Moneyball | autofeat           | lgbm     | ok       | rmse     | 21.5619   |           18 |     1.27 |     215.3 | 22.03     |
| Moneyball | base               | catboost | ok       | rmse     | 21.6748   |           14 |     1.92 |     247.1 | 21.6748   |
| Moneyball | base               | lgbm     | ok       | rmse     | 22.03     |           14 |     1.08 |     214.8 | 22.03     |
| Moneyball | lama_adv           | catboost | ok       | rmse     | 21.2986   |           19 |     1.51 |     231   | 21.6748   |
| Moneyball | lama_adv           | lgbm     | ok       | rmse     | 22.5177   |           19 |     1.07 |     209.2 | 22.03     |
| Moneyball | manual_baseline    | catboost | ok       | rmse     | 21.145    |           60 |     3.88 |     257.1 | 21.6748   |
| Moneyball | manual_baseline    | lgbm     | ok       | rmse     | 21.4047   |           60 |     1.49 |     222   | 22.03     |
| Moneyball | mljar_golden       | catboost | ok       | rmse     | 21.1286   |           24 |     2.15 |     235.3 | 21.6748   |
| Moneyball | mljar_golden       | lgbm     | ok       | rmse     | 21.1837   |           24 |     1.08 |     208   | 22.03     |
| Moneyball | openfe             | catboost | ok       | rmse     | 25.1401   |          222 |     6.16 |     273.3 | 21.6748   |
| Moneyball | openfe             | lgbm     | ok       | rmse     | 21.0998   |          222 |     1.92 |     235.8 | 22.03     |
| cmc       | ag_default         | catboost | ok       | logloss  |  0.934565 |            9 |     1.3  |     229.1 |  0.957806 |
| cmc       | ag_default         | lgbm     | ok       | logloss  |  0.962686 |            9 |     1.07 |     214.1 |  0.962686 |
| cmc       | ag_tabprep         | catboost | ok       | logloss  |  0.959046 |           52 |     2.59 |     245.8 |  0.957806 |
| cmc       | ag_tabprep         | lgbm     | ok       | logloss  |  0.990561 |           52 |     1.29 |     219.7 |  0.962686 |
| cmc       | ag_tabprep_keepcat | catboost | ok       | logloss  |  0.961549 |           56 |     3.65 |     268.8 |  0.957806 |
| cmc       | ag_tabprep_keepcat | lgbm     | ok       | logloss  |  0.990706 |           56 |     1.31 |     220   |  0.962686 |
| cmc       | autofeat           | catboost | ok       | logloss  |  0.950011 |           11 |     2.39 |     261.2 |  0.957806 |
| cmc       | autofeat           | lgbm     | ok       | logloss  |  0.980181 |           11 |     1.52 |     215.3 |  0.962686 |
| cmc       | base               | catboost | ok       | logloss  |  0.957806 |            9 |     2.38 |     253   |  0.957806 |
| cmc       | base               | lgbm     | ok       | logloss  |  0.962686 |            9 |     1.08 |     208.6 |  0.962686 |
| cmc       | lama_adv           | catboost | ok       | logloss  |  0.966945 |           12 |     1.3  |     229.8 |  0.957806 |
| cmc       | lama_adv           | lgbm     | ok       | logloss  |  0.973229 |           12 |     1.08 |     209.3 |  0.962686 |
| cmc       | manual_baseline    | catboost | ok       | logloss  |  0.944071 |           23 |     2.57 |     270.2 |  0.957806 |
| cmc       | manual_baseline    | lgbm     | ok       | logloss  |  0.963228 |           23 |     1.09 |     215.2 |  0.962686 |
| cmc       | mljar_golden       | catboost | ok       | logloss  |  0.931859 |           14 |     2.37 |     276.7 |  0.957806 |
| cmc       | mljar_golden       | lgbm     | ok       | logloss  |  0.94711  |           14 |     1.08 |     213.9 |  0.962686 |
| cmc       | openfe             | catboost | ok       | logloss  |  0.940626 |           85 |     3.66 |     282.7 |  0.957806 |
| cmc       | openfe             | lgbm     | ok       | logloss  |  0.983585 |           85 |     1.94 |     220.5 |  0.962686 |
| credit-g  | ag_default         | catboost | ok       | auc      |  0.837143 |           20 |     1.31 |     231   |  0.793929 |
| credit-g  | ag_default         | lgbm     | ok       | auc      |  0.81369  |           20 |     1.1  |     215.5 |  0.809881 |
| credit-g  | ag_tabprep         | catboost | ok       | auc      |  0.809643 |          962 |    24.92 |     442.5 |  0.793929 |
| credit-g  | ag_tabprep         | lgbm     | ok       | auc      |  0.804286 |          962 |     4.13 |     336.3 |  0.809881 |
| credit-g  | ag_tabprep_keepcat | catboost | ok       | auc      |  0.810714 |          973 |    25.44 |     588.8 |  0.793929 |
| credit-g  | ag_tabprep_keepcat | lgbm     | ok       | auc      |  0.807738 |          973 |     4.31 |     339.1 |  0.809881 |
| credit-g  | autofeat           | catboost | ok       | auc      |  0.817976 |           25 |     2.37 |     250.6 |  0.793929 |
| credit-g  | autofeat           | lgbm     | ok       | auc      |  0.791786 |           25 |     1.52 |     217   |  0.809881 |
| credit-g  | base               | catboost | ok       | auc      |  0.793929 |           20 |     2.17 |     261.6 |  0.793929 |
| credit-g  | base               | lgbm     | ok       | auc      |  0.809881 |           20 |     1.29 |     214.9 |  0.809881 |
| credit-g  | lama_adv           | catboost | ok       | auc      |  0.78369  |           28 |     1.52 |     232   |  0.793929 |
| credit-g  | lama_adv           | lgbm     | ok       | auc      |  0.777024 |           28 |     1.1  |     212.6 |  0.809881 |
| credit-g  | manual_baseline    | catboost | ok       | auc      |  0.804881 |           73 |     3.45 |     263.1 |  0.793929 |
| credit-g  | manual_baseline    | lgbm     | ok       | auc      |  0.813452 |           73 |     1.3  |     213.3 |  0.809881 |
| credit-g  | mljar_golden       | catboost | ok       | auc      |  0.800595 |           30 |     2.36 |     262   |  0.793929 |
| credit-g  | mljar_golden       | lgbm     | ok       | auc      |  0.80631  |           30 |     1.07 |     215.6 |  0.809881 |
| credit-g  | openfe             | catboost | ok       | auc      |  0.771071 |          186 |     3.66 |     273.5 |  0.793929 |
| credit-g  | openfe             | lgbm     | ok       | auc      |  0.768333 |          186 |     1.51 |     229   |  0.809881 |

## Test metric by dataset and arm

|                                     |   catboost |      lgbm |
|:------------------------------------|-----------:|----------:|
| ('Moneyball', 'ag_default')         |  21.7966   | 22.3055   |
| ('Moneyball', 'ag_tabprep')         |  21.3411   | 21.4119   |
| ('Moneyball', 'ag_tabprep_keepcat') |  21.5987   | 21.6128   |
| ('Moneyball', 'autofeat')           |  20.5246   | 21.5619   |
| ('Moneyball', 'base')               |  21.6748   | 22.03     |
| ('Moneyball', 'lama_adv')           |  21.2986   | 22.5177   |
| ('Moneyball', 'manual_baseline')    |  21.145    | 21.4047   |
| ('Moneyball', 'mljar_golden')       |  21.1286   | 21.1837   |
| ('Moneyball', 'openfe')             |  25.1401   | 21.0998   |
| ('cmc', 'ag_default')               |   0.934565 |  0.962686 |
| ('cmc', 'ag_tabprep')               |   0.959046 |  0.990561 |
| ('cmc', 'ag_tabprep_keepcat')       |   0.961549 |  0.990706 |
| ('cmc', 'autofeat')                 |   0.950011 |  0.980181 |
| ('cmc', 'base')                     |   0.957806 |  0.962686 |
| ('cmc', 'lama_adv')                 |   0.966945 |  0.973229 |
| ('cmc', 'manual_baseline')          |   0.944071 |  0.963228 |
| ('cmc', 'mljar_golden')             |   0.931859 |  0.94711  |
| ('cmc', 'openfe')                   |   0.940626 |  0.983585 |
| ('credit-g', 'ag_default')          |   0.837143 |  0.81369  |
| ('credit-g', 'ag_tabprep')          |   0.809643 |  0.804286 |
| ('credit-g', 'ag_tabprep_keepcat')  |   0.810714 |  0.807738 |
| ('credit-g', 'autofeat')            |   0.817976 |  0.791786 |
| ('credit-g', 'base')                |   0.793929 |  0.809881 |
| ('credit-g', 'lama_adv')            |   0.78369  |  0.777024 |
| ('credit-g', 'manual_baseline')     |   0.804881 |  0.813452 |
| ('credit-g', 'mljar_golden')        |   0.800595 |  0.80631  |
| ('credit-g', 'openfe')              |   0.771071 |  0.768333 |
