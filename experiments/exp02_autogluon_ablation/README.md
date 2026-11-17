# exp02 AutoGluon ablation

What in AutoGluon's gain comes from new features, from model diversity, from stacking, and from the
portfolio. 12 datasets (4 per task type, rule in `config.yaml`), repeat 0, 5 outer folds, `time_limit=600`,
log loss or RMSE as the eval metric, one unit at a time on the whole machine. Arms: 7 main ones (`worker.py`
docstring), 4 single-model diagnostics on all datasets (`worker_single.py`) and 4 hyperparameter-group arms on
elevators, kr-vs-kp and socmob (`worker_hp.py`, `hp_groups` in `config.yaml`). 720 units, all ok (420 + 240 + 60).
`make exp02` runs it, `make aggregate` writes the tables and `results/figures/exp02/*`, `make article` the article
PNGs.

Contrast = relative improvement of the first arm over the second on the eval metric (positive favours the
first), mean over folds per dataset with a fold-bootstrap 95% CI. Per task type the result is `N better of M`
datasets (`n_better` of `n_datasets`, the sign of the mean effect) with how many of them are clearly better or
clearly worse (the fold CI excludes zero), and the per-dataset table beside it. The median over 4 datasets is a
secondary column, not a result. Numbers live in `results/tidy/exp02_contrasts_summary.csv` and
`results/claims.yaml` (`s2_*`; `*_median` entries are `role: secondary`).

## Findings (descriptive; 4 datasets per task type)

Counts are better by sign (clearly better by CI) on binary, multiclass and regression.

- `D` over `A`: 4 (4), 4 (3) and 4 (4) of 4; cmc is the smallest gain (+0.7%, CI includes zero). Against the
  default-hyperparameter `A_def`: 4 (3), 4 (4), 4 (4) (`s2_d_a_*`, `s2_d_a_def_*`). `D` is `best_quality` with the 2023 portfolio, bagging
  and stacking; it has no TabPrep model, and it bundles many models, so it does not say which part helps.
- Stacking over the same portfolio (`D` over `D0`): 1 (0), 2 (2) and 3 (2) of 4. On all four binary datasets `D`
  trained no level-2 models (model counts equal `D0`'s, for example 109 and 109 on credit-g; why was not
  investigated), so there `D` over `D0` compares two runs of the same configuration, and two datasets are exactly
  equal. It is a stacking comparison on multiclass and regression only. Large gains: pol +17.5%, pendigits +10.3%; cmc is -1.6% (clearly worse, CI [-2.0%, -1.2%]) (`s2_d_d0_*`).
- The 2025 portfolio with `GBM_PREP` and no stacking (`E` over `D0`): 0 (0), 2 (1) and 1 (0) of 4; clearly worse
  on all 4 binary datasets, on 1 of 4 multiclass datasets (pendigits, -24.9%) and on 2 of 4 regression datasets (`s2_e_d0_*`). The arms are not matched in effort: `E`
  trains 6 to 18 models (including ensembles) and stops early (mean 346 s), `D0` 14 to 109 and runs to the budget (mean 559 s), so the
  contrast mixes the portfolio with the amount of search.
- `B` over `A` keeps the `_r13` hyperparameters and changes the features: feature group on, native categoricals
  replaced by out-of-fold target encodings. 2 (2), 3 (3) and 4 (2) of 4 (`s2_b_a_*`). `B` over `B2` changes the
  hyperparameters as well (`B2` is the portfolio's `_r21`: `extra_trees=False`, learning rate 0.013, 20 leaves, no
  feature group): 1 (0), 4 (4) and 4 (4) of 4 (`s2_b_b2_*`). The two contrasts answer different questions; the
  single-model section below separates features from hyperparameters.
- Cost: mean wall per unit `A` 23 s, `B` 40 s, `B2` 21 s, `C` 83 s, `E` 346 s, `D0` 559 s, `D` 611 s;
  `D` and `D0` sit at the 600 s budget on most sets and overshoot it by up to about 110 s (`s2_wall_*`).

## Single-model diagnostics: why `_r13` is weak

`A` and `B` carry the `_r13` hyperparameters (a member of AutoGluon's 2025 portfolio, which it fits bagged:
`extra_trees=True`,
learning rate 0.019, 64 leaves, `lambda_l1` 0.91, `lambda_l2` 1.90), so `B - A` is the feature effect on top of that
configuration. Arms: `A_def` (GBM, AutoGluon defaults, no bagging), `B_def` (`_r13` feature group, default
hyperparameters), `A_et0` and `B_et0` (`A` and `B` with `extra_trees=False`), and four arms that return one
`_r13` group to the default while keeping `A_et0`: sampling (`feature_fraction`, `bagging_*`), capacity (learning
rate, leaves, `min_data_in_leaf`), regularization (`lambda_l1`, `lambda_l2`) and categorical (`cat_*`,
`max_cat_to_onehot`, `min_data_per_group`). Per-dataset numbers: `s2_single_*`, `s2_hp_*`, `s2_mean_loss_*`,
`s2_rounds_*`, images `exp02_single_model*` and `exp02_hp_groups_table`.

- `_r13` is not uniformly weak. `A` against `A_def` is clearly worse on 5 datasets: pol (-64.8%), sick (-68.0%),
  elevators (-68.2%), kr-vs-kp (-61.8%) and bank-marketing (-0.8%, CI upper bound -0.02%: at the edge, other bootstrap seeds put it on zero, so 4 or 5); clearly better on 3: credit-g (+4.9%),
  mfeat-fourier (+10.1%) and Moneyball (+5.8%); the CI includes zero on the other 4 (pendigits +18.0%, socmob
  -33.2%).
- `extra_trees=False` explains pol and sick. `A_et0` over `A`: pol RMSE 7.06 to 4.59 (`A_def` 4.30), sick log loss
  0.0548 to 0.0351 (`A_def` 0.0352). It does not explain elevators (0.0034 against 0.0021) or kr-vs-kp (0.0350
  against 0.0248).
- On elevators and kr-vs-kp returning the regularization group to default closes the gap. With `lambda_l1` and `lambda_l2` at default, `A_et0`
  reaches 0.00214 and 0.0242 (`A_def` 0.00212 and 0.0248); against `A_def` the paired difference is -0.9%
  (CI [-3.1%, +0.6%]) and -0.3% (CI [-9.8%, +7.1%]), so no difference is detectable with 5 folds. The other groups
  do not close the gap: sampling +1.9% and +2.5%, capacity +0.3% and +6.6% over `A_et0` (elevators, kr-vs-kp), and
  the categorical group changes nothing (identical predictions). That fits how LightGBM treats categoricals: a feature with at most
  `max_cat_to_onehot` levels (4 by default, 18 in `_r13`) gets one-vs-rest splits, and kr-vs-kp has at most 3 levels per
  column while elevators has no categorical column, so the `cat_*` values never apply there.
- A confound: the four datasets where `A` loses by more than 60% are the ones where it trains to AutoGluon's
  10,000-round cap (median rounds 9,997 on elevators, 9,967 on kr-vs-kp, 10,000 on pol, 9,966 on sick); on the
  other datasets it stops at 639 to 6,845 rounds, except pendigits (9,976 rounds, `A` better). `A_et0` on sick stops at 2,613 rounds,
  where the gap closes; `A_et0_noreg` stops at 2,397 and 902 rounds; `A_def` stops at 1,082 (elevators), 154
  (kr-vs-kp), 4,261 (pol) and 122 (sick). So the cases the groups fix are models still improving at the cap. The cap
  is not specific to `A`: `A_et0` (elevators 9,990, kr-vs-kp 9,967, pol 10,000), `B` (9,995, 9,967) and `B2` (10,000,
  9,998, pol 9,999) reach it too, while `B` stops at 4,322 rounds on pol and `B_et0` at 1,826 (`s2_rounds_*`). This
  is a pattern in the records; whether the regularization values or the cap drive the loss was not tested.
- socmob is unresolved: the categorical group moves the mean from 19.8 to 17.0 (`A_def` 18.0), but the fold CI of
  the improvement is [-1.2%, +29.9%] and the sd of the fold RMSE is 4.9. Its categorical columns have up to 17 levels,
  one-hot under `_r13` (at most 18) and not by default (4), which is consistent with that group mattering here; this was
  not isolated. pol after `extra_trees=False` still
  trails `A_def` (4.59 against 4.30).
- The feature-group effect depends on the hyperparameters under it. On pol the `_r13` features improve RMSE by
  42.1% on the full `_r13` set, by 12.0% with `extra_trees=False`, and worsen it by 9.3% with default
  hyperparameters (`s2_single_features_pol_*`). With default hyperparameters (`B_def` over `A_def`) the features
  are better on 1 (0), 2 (2) and 2 (2) of 4 datasets. `B - A` is therefore not a measure of the feature group alone: the
  same features give +42.1%, +12.0% and -9.3% on pol under three hyperparameter settings. On pol `B` over `A` compares a model that
  stops at 4,322 rounds with one at the 10,000 cap, and `B_et0` over `A_et0` 1,826 with 10,000, so those two gains
  mix the features with the cap; `B` over `B2` on pol (4,322 against 9,999) likewise. On elevators and kr-vs-kp both
  arms of `B` over `B2` sit at the cap.

Scope of the hyperparameter-group arms: they ran on elevators, kr-vs-kp and socmob, three of the datasets where
`A_et0` still trailed `A_def` (pol and okcupid-stem also trail and were not run); each group is removed on its
own, so interactions between groups are untested, and `extra_trees` belongs to no group.

Caveats: 4 datasets per task type make the medians descriptive; the CIs are over folds of one dataset and say
nothing about other datasets. `D` and `D0` turn dynamic stacking off so that both spend the full budget;
AutoGluon's default would stop earlier. Timing is one machine with all cores and no other job (the run was paused
for other jobs; each paused unit was recomputed from scratch). The bootstrap CIs come from one random
stream over `CONTRASTS`, so a CI depends on the contrast's position: add new contrasts at the end.
