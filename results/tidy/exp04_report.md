# exp04 report

Written by `experiments/exp04_open_benchmark/report.py` from
`results/tidy/exp04.parquet`, `exp04_effects.csv` and `exp04_summary.csv`.

## Coverage (ok / planned fit units, both repeats)

| arm                | model    |   planned |   ok |   coverage |
|:-------------------|:---------|----------:|-----:|-----------:|
| ag_default         | catboost |       490 |  490 |     1.0000 |
| ag_default         | lgbm     |       500 |  500 |     1.0000 |
| ag_tabprep         | catboost |       490 |  459 |     0.9370 |
| ag_tabprep         | lgbm     |       500 |  500 |     1.0000 |
| ag_tabprep_keepcat | catboost |       490 |  461 |     0.9410 |
| ag_tabprep_keepcat | lgbm     |       500 |  500 |     1.0000 |
| autofeat           | catboost |       490 |  308 |     0.6290 |
| autofeat           | lgbm     |       500 |  308 |     0.6160 |
| base               | catboost |       490 |  490 |     1.0000 |
| base               | lgbm     |       500 |  500 |     1.0000 |
| lama_adv           | catboost |       490 |  490 |     1.0000 |
| lama_adv           | lgbm     |       500 |  500 |     1.0000 |
| manual_baseline    | catboost |       490 |  490 |     1.0000 |
| manual_baseline    | lgbm     |       500 |  500 |     1.0000 |
| mljar_golden       | catboost |       490 |  460 |     0.9390 |
| mljar_golden       | lgbm     |       500 |  460 |     0.9200 |
| openfe             | catboost |       490 |  453 |     0.9240 |
| openfe             | lgbm     |       500 |  470 |     0.9400 |
| skrub_tv           | catboost |       490 |  490 |     1.0000 |
| skrub_tv           | lgbm     |       500 |  500 |     1.0000 |

## Median effect vs `base` over primary datasets (base_fallback)

Positive favours the arm. 95% bootstrap CI over datasets; Wilcoxon signed-rank with
Holm within (task, model, effect, rule).

| task       | effect      | model    | arm                |   n_datasets |   median_effect |   ci_lo |   ci_hi |   wilcoxon_p_holm |   share_win |   share_loss |   coverage |
|:-----------|:------------|:---------|:-------------------|-------------:|----------------:|--------:|--------:|------------------:|------------:|-------------:|-----------:|
| binary     | auc_diff    | catboost | ag_default         |           20 |         -0.0000 | -0.0009 |  0.0000 |            0.2149 |      0.0000 |       0.1500 |     1.0000 |
| binary     | auc_diff    | catboost | ag_tabprep         |           20 |          0.0000 | -0.0016 |  0.0016 |            1.0000 |      0.0500 |       0.2000 |     1.0000 |
| binary     | auc_diff    | catboost | ag_tabprep_keepcat |           20 |          0.0000 | -0.0018 |  0.0014 |            1.0000 |      0.0500 |       0.1500 |     1.0000 |
| binary     | auc_diff    | catboost | autofeat           |           20 |          0.0000 |  0.0000 |  0.0000 |            1.0000 |      0.0500 |       0.0000 |     0.5000 |
| binary     | auc_diff    | catboost | lama_adv           |           20 |         -0.0002 | -0.0015 |  0.0001 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| binary     | auc_diff    | catboost | manual_baseline    |           20 |          0.0000 | -0.0010 |  0.0020 |            1.0000 |      0.1500 |       0.1000 |     1.0000 |
| binary     | auc_diff    | catboost | mljar_golden       |           20 |          0.0002 | -0.0000 |  0.0019 |            0.3192 |      0.0000 |       0.0000 |     0.9000 |
| binary     | auc_diff    | catboost | openfe             |           20 |          0.0017 |  0.0000 |  0.0039 |            0.2149 |      0.1500 |       0.0500 |     0.9500 |
| binary     | auc_diff    | catboost | skrub_tv           |           20 |          0.0000 | -0.0004 |  0.0000 |            1.0000 |      0.0000 |       0.1500 |     1.0000 |
| binary     | auc_diff    | lgbm     | ag_default         |           20 |          0.0000 |  0.0000 |  0.0000 |            1.0000 |      0.0000 |       0.0500 |     1.0000 |
| binary     | auc_diff    | lgbm     | ag_tabprep         |           20 |          0.0009 | -0.0008 |  0.0037 |            1.0000 |      0.2000 |       0.1500 |     1.0000 |
| binary     | auc_diff    | lgbm     | ag_tabprep_keepcat |           20 |          0.0018 | -0.0001 |  0.0037 |            0.6604 |      0.2000 |       0.1000 |     1.0000 |
| binary     | auc_diff    | lgbm     | autofeat           |           20 |          0.0000 | -0.0001 |  0.0000 |            1.0000 |      0.0500 |       0.0000 |     0.5000 |
| binary     | auc_diff    | lgbm     | lama_adv           |           20 |          0.0000 | -0.0002 |  0.0002 |            1.0000 |      0.0500 |       0.0500 |     1.0000 |
| binary     | auc_diff    | lgbm     | manual_baseline    |           20 |          0.0001 | -0.0008 |  0.0029 |            1.0000 |      0.1000 |       0.0000 |     1.0000 |
| binary     | auc_diff    | lgbm     | mljar_golden       |           20 |          0.0004 | -0.0001 |  0.0013 |            0.6296 |      0.0000 |       0.0000 |     0.9000 |
| binary     | auc_diff    | lgbm     | openfe             |           20 |          0.0002 | -0.0024 |  0.0033 |            1.0000 |      0.1000 |       0.2000 |     0.9500 |
| binary     | auc_diff    | lgbm     | skrub_tv           |           20 |          0.0000 | -0.0000 |  0.0002 |            1.0000 |      0.1000 |       0.0500 |     1.0000 |
| multiclass | logloss_rel | catboost | ag_default         |            9 |          0.0000 | -0.0090 |  0.0000 |            1.0000 |      0.0000 |       0.1111 |     1.0000 |
| multiclass | logloss_rel | catboost | ag_tabprep         |            9 |         -0.0064 | -0.0693 |  0.0010 |            0.9844 |      0.0000 |       0.1111 |     0.6556 |
| multiclass | logloss_rel | catboost | ag_tabprep_keepcat |            9 |         -0.0064 | -0.0693 |  0.0022 |            0.9844 |      0.0000 |       0.1111 |     0.6778 |
| multiclass | logloss_rel | catboost | autofeat           |            9 |          0.0000 |  0.0000 |  0.0059 |            1.0000 |      0.0000 |       0.0000 |     0.2000 |
| multiclass | logloss_rel | catboost | lama_adv           |            9 |         -0.0040 | -0.0484 |  0.0071 |            1.0000 |      0.0000 |       0.1111 |     1.0000 |
| multiclass | logloss_rel | catboost | manual_baseline    |            9 |         -0.0165 | -0.0824 | -0.0014 |            0.2461 |      0.1111 |       0.2222 |     1.0000 |
| multiclass | logloss_rel | catboost | mljar_golden       |            9 |         -0.0004 | -0.0516 |  0.0122 |            1.0000 |      0.1111 |       0.0000 |     1.0000 |
| multiclass | logloss_rel | catboost | openfe             |            9 |          0.0310 | -0.0037 |  0.2265 |            0.3750 |      0.3333 |       0.1111 |     0.8111 |
| multiclass | logloss_rel | catboost | skrub_tv           |            9 |          0.0000 | -0.0011 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | ag_default         |           10 |          0.0000 |  0.0000 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | ag_tabprep         |           10 |         -0.0023 | -0.0424 |  0.1418 |            1.0000 |      0.4000 |       0.1000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | ag_tabprep_keepcat |           10 |         -0.0028 | -0.0404 |  0.1418 |            1.0000 |      0.4000 |       0.1000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | autofeat           |           10 |          0.0000 | -0.0022 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     0.1800 |
| multiclass | logloss_rel | lgbm     | lama_adv           |           10 |         -0.0089 | -0.0329 | -0.0005 |            0.1758 |      0.0000 |       0.2000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | manual_baseline    |           10 |         -0.0078 | -0.0482 |  0.0069 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | mljar_golden       |           10 |         -0.0005 | -0.0150 |  0.0195 |            1.0000 |      0.0000 |       0.0000 |     0.9000 |
| multiclass | logloss_rel | lgbm     | openfe             |           10 |          0.0733 | -0.0083 |  0.3487 |            0.5938 |      0.3000 |       0.1000 |     0.9000 |
| multiclass | logloss_rel | lgbm     | skrub_tv           |           10 |          0.0000 | -0.0001 |  0.0061 |            1.0000 |      0.2000 |       0.0000 |     1.0000 |
| regression | rmse_rel    | catboost | ag_default         |           20 |          0.0000 | -0.0020 |  0.0052 |            1.0000 |      0.1000 |       0.0000 |     1.0000 |
| regression | rmse_rel    | catboost | ag_tabprep         |           20 |         -0.0070 | -0.0579 |  0.0089 |            1.0000 |      0.1000 |       0.3500 |     1.0000 |
| regression | rmse_rel    | catboost | ag_tabprep_keepcat |           20 |         -0.0023 | -0.0338 |  0.0068 |            1.0000 |      0.1000 |       0.2500 |     1.0000 |
| regression | rmse_rel    | catboost | autofeat           |           20 |         -0.0001 | -0.0255 |  0.0103 |            1.0000 |      0.1000 |       0.1500 |     0.9500 |
| regression | rmse_rel    | catboost | lama_adv           |           20 |         -0.0152 | -0.0344 | -0.0030 |            0.1731 |      0.0000 |       0.2500 |     1.0000 |
| regression | rmse_rel    | catboost | manual_baseline    |           20 |         -0.0030 | -0.0117 |  0.0156 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| regression | rmse_rel    | catboost | mljar_golden       |           20 |         -0.0022 | -0.0078 |  0.0018 |            1.0000 |      0.0000 |       0.1000 |     0.9500 |
| regression | rmse_rel    | catboost | openfe             |           20 |         -0.0006 | -0.0166 |  0.0353 |            1.0000 |      0.1500 |       0.1000 |     0.9500 |
| regression | rmse_rel    | catboost | skrub_tv           |           20 |          0.0000 | -0.0027 |  0.0011 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| regression | rmse_rel    | lgbm     | ag_default         |           20 |          0.0000 | -0.0000 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     1.0000 |
| regression | rmse_rel    | lgbm     | ag_tabprep         |           20 |         -0.0189 | -0.0596 |  0.0260 |            1.0000 |      0.2000 |       0.2000 |     1.0000 |
| regression | rmse_rel    | lgbm     | ag_tabprep_keepcat |           20 |         -0.0046 | -0.0522 |  0.0261 |            1.0000 |      0.1500 |       0.2000 |     1.0000 |
| regression | rmse_rel    | lgbm     | autofeat           |           20 |          0.0000 | -0.0143 |  0.0201 |            1.0000 |      0.0500 |       0.1500 |     0.9500 |
| regression | rmse_rel    | lgbm     | lama_adv           |           20 |         -0.0106 | -0.0537 | -0.0007 |            0.0152 |      0.0000 |       0.2500 |     1.0000 |
| regression | rmse_rel    | lgbm     | manual_baseline    |           20 |          0.0011 | -0.0066 |  0.0309 |            1.0000 |      0.1500 |       0.1500 |     1.0000 |
| regression | rmse_rel    | lgbm     | mljar_golden       |           20 |         -0.0029 | -0.0119 |  0.0004 |            0.7728 |      0.0000 |       0.0500 |     0.9500 |
| regression | rmse_rel    | lgbm     | openfe             |           20 |          0.0047 | -0.0167 |  0.0308 |            1.0000 |      0.1500 |       0.1000 |     0.9500 |
| regression | rmse_rel    | lgbm     | skrub_tv           |           20 |         -0.0001 | -0.0028 |  0.0000 |            1.0000 |      0.0500 |       0.0500 |     1.0000 |

## Median effect vs `base` over primary datasets (complete_cases)

Positive favours the arm. 95% bootstrap CI over datasets; Wilcoxon signed-rank with
Holm within (task, model, effect, rule).

| task       | effect      | model    | arm                |   n_datasets |   median_effect |   ci_lo |   ci_hi |   wilcoxon_p_holm |   share_win |   share_loss |   coverage |
|:-----------|:------------|:---------|:-------------------|-------------:|----------------:|--------:|--------:|------------------:|------------:|-------------:|-----------:|
| binary     | auc_diff    | catboost | ag_default         |           20 |         -0.0000 | -0.0009 |  0.0000 |            0.2113 |      0.0000 |       0.1500 |     1.0000 |
| binary     | auc_diff    | catboost | ag_tabprep         |           20 |          0.0000 | -0.0016 |  0.0016 |            1.0000 |      0.0500 |       0.2000 |     1.0000 |
| binary     | auc_diff    | catboost | ag_tabprep_keepcat |           20 |          0.0000 | -0.0018 |  0.0014 |            1.0000 |      0.0500 |       0.1500 |     1.0000 |
| binary     | auc_diff    | catboost | autofeat           |           10 |          0.0001 | -0.0006 |  0.0035 |            1.0000 |      0.1000 |       0.0000 |     0.5000 |
| binary     | auc_diff    | catboost | lama_adv           |           20 |         -0.0002 | -0.0015 |  0.0001 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| binary     | auc_diff    | catboost | manual_baseline    |           20 |          0.0000 | -0.0010 |  0.0020 |            1.0000 |      0.1500 |       0.1000 |     1.0000 |
| binary     | auc_diff    | catboost | mljar_golden       |           18 |          0.0006 | -0.0002 |  0.0019 |            0.2694 |      0.0000 |       0.0000 |     0.9000 |
| binary     | auc_diff    | catboost | openfe             |           19 |          0.0020 |  0.0000 |  0.0045 |            0.2069 |      0.1579 |       0.0526 |     0.9500 |
| binary     | auc_diff    | catboost | skrub_tv           |           20 |          0.0000 | -0.0004 |  0.0000 |            1.0000 |      0.0000 |       0.1500 |     1.0000 |
| binary     | auc_diff    | lgbm     | ag_default         |           20 |          0.0000 |  0.0000 |  0.0000 |            1.0000 |      0.0000 |       0.0500 |     1.0000 |
| binary     | auc_diff    | lgbm     | ag_tabprep         |           20 |          0.0009 | -0.0008 |  0.0037 |            1.0000 |      0.2000 |       0.1500 |     1.0000 |
| binary     | auc_diff    | lgbm     | ag_tabprep_keepcat |           20 |          0.0018 | -0.0001 |  0.0037 |            0.6604 |      0.2000 |       0.1000 |     1.0000 |
| binary     | auc_diff    | lgbm     | autofeat           |           10 |         -0.0003 | -0.0034 |  0.0026 |            1.0000 |      0.1000 |       0.0000 |     0.5000 |
| binary     | auc_diff    | lgbm     | lama_adv           |           20 |          0.0000 | -0.0002 |  0.0002 |            1.0000 |      0.0500 |       0.0500 |     1.0000 |
| binary     | auc_diff    | lgbm     | manual_baseline    |           20 |          0.0001 | -0.0008 |  0.0029 |            1.0000 |      0.1000 |       0.0000 |     1.0000 |
| binary     | auc_diff    | lgbm     | mljar_golden       |           18 |          0.0008 | -0.0001 |  0.0018 |            0.5988 |      0.0000 |       0.0000 |     0.9000 |
| binary     | auc_diff    | lgbm     | openfe             |           19 |          0.0004 | -0.0033 |  0.0039 |            1.0000 |      0.1053 |       0.2105 |     0.9500 |
| binary     | auc_diff    | lgbm     | skrub_tv           |           20 |          0.0000 | -0.0000 |  0.0002 |            1.0000 |      0.1000 |       0.0500 |     1.0000 |
| multiclass | logloss_rel | catboost | ag_default         |            9 |          0.0000 | -0.0090 |  0.0000 |            1.0000 |      0.0000 |       0.1111 |     1.0000 |
| multiclass | logloss_rel | catboost | ag_tabprep         |            7 |         -0.0250 | -0.0693 |  0.0010 |            1.0000 |      0.0000 |       0.2857 |     0.6556 |
| multiclass | logloss_rel | catboost | ag_tabprep_keepcat |            7 |         -0.0250 | -0.0693 |  0.0022 |            1.0000 |      0.1429 |       0.2857 |     0.6778 |
| multiclass | logloss_rel | catboost | autofeat           |            2 |          0.0108 |  0.0074 |  0.0142 |            1.0000 |      0.0000 |       0.0000 |     0.2000 |
| multiclass | logloss_rel | catboost | lama_adv           |            9 |         -0.0040 | -0.0484 |  0.0071 |            1.0000 |      0.0000 |       0.1111 |     1.0000 |
| multiclass | logloss_rel | catboost | manual_baseline    |            9 |         -0.0165 | -0.0824 | -0.0014 |            0.2461 |      0.1111 |       0.2222 |     1.0000 |
| multiclass | logloss_rel | catboost | mljar_golden       |            9 |         -0.0004 | -0.0516 |  0.0122 |            1.0000 |      0.1111 |       0.0000 |     1.0000 |
| multiclass | logloss_rel | catboost | openfe             |            8 |          0.0312 | -0.0037 |  0.4237 |            0.3125 |      0.5000 |       0.1250 |     0.8111 |
| multiclass | logloss_rel | catboost | skrub_tv           |            9 |          0.0000 | -0.0011 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | ag_default         |           10 |          0.0000 |  0.0000 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | ag_tabprep         |           10 |         -0.0023 | -0.0424 |  0.1418 |            1.0000 |      0.4000 |       0.1000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | ag_tabprep_keepcat |           10 |         -0.0028 | -0.0404 |  0.1418 |            1.0000 |      0.4000 |       0.1000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | autofeat           |            2 |         -0.0095 | -0.0135 | -0.0056 |            1.0000 |      0.0000 |       0.0000 |     0.1800 |
| multiclass | logloss_rel | lgbm     | lama_adv           |           10 |         -0.0089 | -0.0329 | -0.0005 |            0.1758 |      0.0000 |       0.2000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | manual_baseline    |           10 |         -0.0078 | -0.0482 |  0.0069 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| multiclass | logloss_rel | lgbm     | mljar_golden       |            9 |         -0.0010 | -0.0278 |  0.0343 |            1.0000 |      0.0000 |       0.0000 |     0.9000 |
| multiclass | logloss_rel | lgbm     | openfe             |            9 |          0.0821 | -0.0166 |  0.6153 |            0.4375 |      0.3333 |       0.1111 |     0.9000 |
| multiclass | logloss_rel | lgbm     | skrub_tv           |           10 |          0.0000 | -0.0001 |  0.0061 |            1.0000 |      0.2000 |       0.0000 |     1.0000 |
| regression | rmse_rel    | catboost | ag_default         |           20 |          0.0000 | -0.0020 |  0.0052 |            1.0000 |      0.1000 |       0.0000 |     1.0000 |
| regression | rmse_rel    | catboost | ag_tabprep         |           20 |         -0.0070 | -0.0579 |  0.0089 |            1.0000 |      0.1000 |       0.3500 |     1.0000 |
| regression | rmse_rel    | catboost | ag_tabprep_keepcat |           20 |         -0.0023 | -0.0338 |  0.0068 |            1.0000 |      0.1000 |       0.2500 |     1.0000 |
| regression | rmse_rel    | catboost | autofeat           |           19 |         -0.0002 | -0.0258 |  0.0113 |            1.0000 |      0.1053 |       0.1579 |     0.9500 |
| regression | rmse_rel    | catboost | lama_adv           |           20 |         -0.0152 | -0.0344 | -0.0030 |            0.1731 |      0.0000 |       0.2500 |     1.0000 |
| regression | rmse_rel    | catboost | manual_baseline    |           20 |         -0.0030 | -0.0117 |  0.0156 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| regression | rmse_rel    | catboost | mljar_golden       |           19 |         -0.0027 | -0.0108 |  0.0027 |            1.0000 |      0.0000 |       0.1053 |     0.9500 |
| regression | rmse_rel    | catboost | openfe             |           19 |         -0.0012 | -0.0217 |  0.0499 |            1.0000 |      0.1579 |       0.1053 |     0.9500 |
| regression | rmse_rel    | catboost | skrub_tv           |           20 |          0.0000 | -0.0027 |  0.0011 |            1.0000 |      0.1000 |       0.1000 |     1.0000 |
| regression | rmse_rel    | lgbm     | ag_default         |           20 |          0.0000 | -0.0000 |  0.0000 |            1.0000 |      0.0000 |       0.0000 |     1.0000 |
| regression | rmse_rel    | lgbm     | ag_tabprep         |           20 |         -0.0189 | -0.0596 |  0.0260 |            1.0000 |      0.2000 |       0.2000 |     1.0000 |
| regression | rmse_rel    | lgbm     | ag_tabprep_keepcat |           20 |         -0.0046 | -0.0522 |  0.0261 |            1.0000 |      0.1500 |       0.2000 |     1.0000 |
| regression | rmse_rel    | lgbm     | autofeat           |           19 |          0.0000 | -0.0174 |  0.0217 |            1.0000 |      0.0526 |       0.1579 |     0.9500 |
| regression | rmse_rel    | lgbm     | lama_adv           |           20 |         -0.0106 | -0.0537 | -0.0007 |            0.0152 |      0.0000 |       0.2500 |     1.0000 |
| regression | rmse_rel    | lgbm     | manual_baseline    |           20 |          0.0011 | -0.0066 |  0.0309 |            1.0000 |      0.1500 |       0.1500 |     1.0000 |
| regression | rmse_rel    | lgbm     | mljar_golden       |           19 |         -0.0046 | -0.0169 |  0.0009 |            0.8333 |      0.0000 |       0.0526 |     0.9500 |
| regression | rmse_rel    | lgbm     | openfe             |           19 |          0.0093 | -0.0181 |  0.0336 |            1.0000 |      0.1579 |       0.1053 |     0.9500 |
| regression | rmse_rel    | lgbm     | skrub_tv           |           20 |         -0.0001 | -0.0028 |  0.0000 |            1.0000 |      0.0500 |       0.0500 |     1.0000 |

## Binary secondary effect: relative log loss improvement (base_fallback)

| model    | arm                |   n_datasets |   median_effect |   ci_lo |   ci_hi |   share_win |   share_loss |
|:---------|:-------------------|-------------:|----------------:|--------:|--------:|------------:|-------------:|
| catboost | ag_default         |           20 |          0.0000 | -0.0082 |  0.0000 |      0.0000 |       0.2000 |
| catboost | ag_tabprep         |           20 |         -0.0007 | -0.0102 |  0.0130 |      0.0500 |       0.2000 |
| catboost | ag_tabprep_keepcat |           20 |         -0.0007 | -0.0126 |  0.0143 |      0.1000 |       0.1500 |
| catboost | autofeat           |           20 |          0.0000 | -0.0003 |  0.0000 |      0.1000 |       0.0000 |
| catboost | lama_adv           |           20 |         -0.0012 | -0.0190 |  0.0000 |      0.1000 |       0.1000 |
| catboost | manual_baseline    |           20 |         -0.0002 | -0.0029 |  0.0060 |      0.0500 |       0.1500 |
| catboost | mljar_golden       |           20 |          0.0001 | -0.0004 |  0.0145 |      0.0000 |       0.0000 |
| catboost | openfe             |           20 |          0.0106 | -0.0000 |  0.0339 |      0.2500 |       0.0500 |
| catboost | skrub_tv           |           20 |          0.0000 |  0.0000 |  0.0004 |      0.0000 |       0.2000 |
| lgbm     | ag_default         |           20 |          0.0000 |  0.0000 |  0.0000 |      0.0000 |       0.0500 |
| lgbm     | ag_tabprep         |           20 |          0.0011 | -0.0131 |  0.0348 |      0.1500 |       0.1500 |
| lgbm     | ag_tabprep_keepcat |           20 |          0.0025 | -0.0062 |  0.0346 |      0.1500 |       0.1000 |
| lgbm     | autofeat           |           20 |          0.0000 |  0.0000 |  0.0000 |      0.0000 |       0.0500 |
| lgbm     | lama_adv           |           20 |          0.0000 | -0.0139 |  0.0011 |      0.1000 |       0.0500 |
| lgbm     | manual_baseline    |           20 |          0.0002 | -0.0040 |  0.0136 |      0.1500 |       0.0000 |
| lgbm     | mljar_golden       |           20 |          0.0006 | -0.0005 |  0.0097 |      0.0000 |       0.0000 |
| lgbm     | openfe             |           20 |          0.0041 | -0.0196 |  0.0336 |      0.1000 |       0.3000 |
| lgbm     | skrub_tv           |           20 |          0.0000 | -0.0001 |  0.0003 |      0.1500 |       0.1000 |

## Breakdown: datasets with a categorical of >= 10 levels (base_fallback, descriptive)

| model    | arm                |   no_hicard |   hicard |
|:---------|:-------------------|------------:|---------:|
| catboost | ag_default         |      0.0000 |  -0.0013 |
| catboost | ag_tabprep         |     -0.0002 |  -0.0028 |
| catboost | ag_tabprep_keepcat |     -0.0002 |  -0.0027 |
| catboost | autofeat           |      0.0000 |   0.0000 |
| catboost | lama_adv           |     -0.0015 |  -0.0058 |
| catboost | manual_baseline    |     -0.0011 |  -0.0008 |
| catboost | mljar_golden       |      0.0000 |   0.0000 |
| catboost | openfe             |      0.0029 |   0.0000 |
| catboost | skrub_tv           |      0.0000 |  -0.0027 |
| lgbm     | ag_default         |      0.0000 |   0.0000 |
| lgbm     | ag_tabprep         |     -0.0001 |   0.0008 |
| lgbm     | ag_tabprep_keepcat |      0.0002 |   0.0023 |
| lgbm     | autofeat           |      0.0000 |   0.0000 |
| lgbm     | lama_adv           |     -0.0024 |  -0.0002 |
| lgbm     | manual_baseline    |     -0.0009 |   0.0025 |
| lgbm     | mljar_golden       |      0.0003 |  -0.0008 |
| lgbm     | openfe             |      0.0039 |  -0.0015 |
| lgbm     | skrub_tv           |      0.0000 |  -0.0004 |
