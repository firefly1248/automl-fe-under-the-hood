# exp06 AutoML end to end

AutoGluon `best_quality` (`ag_best`), LightAutoML `TabularAutoML` (`lama`) and H2O AutoML without (`h2o`)
and with (`h2o_te`) `preprocessing=["target_encoding"]`, each given 600 s on the whole machine, same folds
as exp02 (repeat 0, 5 outer folds). 10 datasets: the exp02 list minus mfeat-fourier and pendigits (pure
numeric multiclass, where the target-encoding contrast is moot), so multiclass has 2 datasets. The contrast is moot
on most of the remaining ones too (see the findings). 230 units, all
ok: 200 for the four arms and 30 LAMA diagnostics on pol and socmob (below). `make exp06` runs it,
`make aggregate` writes `results/tidy/exp06*.{parquet,csv}` and `results/figures/exp06/contrasts.*`, `make
article` the article PNGs; numbers for the article are the `s6_*` entries of `results/claims.yaml`.

Contrast as in exp02: relative improvement of the first arm over the second on log loss (classification)
or RMSE (regression), mean over folds per dataset with a fold-bootstrap 95% CI. The headline per task type is
`N better of M` datasets, with the per-dataset table beside it; the median is secondary (`role: secondary` in
the claims) and is not shown without that count.

## Findings (descriptive; 4 binary, 2 multiclass, 4 regression datasets)

Counts are better by sign (clearly better by the fold CI) on binary, multiclass and regression.

- `ag_best` is the strongest arm. `lama` is better than it on 0 (0) of 4, 1 (0) of 2 and 1 (0) of 4 datasets and
  clearly worse on 4, 1 and 3 (`s6_lama_ag_*`); `h2o` is better on 0 of 4, 0 of 2 and 0 of 4, clearly worse on 3, 1
  and 3 (`s6_h2o_ag_*`).
- Target encoding in H2O encodes only columns of 25 or more levels (the probe left a 24-level column raw; claim
  `s2_h2o_te_columns_encoded`). Only okcupid-stem (7,019 levels) and Moneyball (39) have such a column, and `h2o_te` is
  clearly worse than `h2o` on both (-2.1%, -5.3%; `s6_h2o_te_h2o_okcupid_stem`, `s6_h2o_te_h2o_moneyball`). On the
  other 8 datasets nothing is encoded, yet the two arms still differ: pol -10.2% (clearly worse by the fold CI),
  credit-g -23.5% (`s6_h2o_te_h2o_pol`, `s6_h2o_te_h2o_credit_g`), cmc exactly 0. That is a difference the encoding
  cannot cause; it is probably H2O's own variation between time-limited runs, which the fold CIs do not include (the
  cause was not isolated). The counts over all 10 datasets (`s6_h2o_te_h2o_*`) mix the two cases, so read them with
  this caveat.
- Budget use differs: `ag_best` (mean 588 s), `h2o` (611 s) and `h2o_te` (609 s) run to the limit, `lama`
  stops on its own at a mean of 213 s (52 to 450 s; `s6_wall_*`). Giving `lama` the full budget was tested on pol
  and socmob only (below): +2.6% on socmob and no extra use on pol, small next to the gap to `ag_best`.

## LAMA diagnostics: pol and socmob

`lama` trailed the exp04 base GBM on pol (RMSE 4.91 against 4.49 for LightGBM and 4.48 for CatBoost) and on
socmob (20.59 against 18.29 LightGBM, 15.81 CatBoost; same folds, repeat 0). Those base models fit on 68% of the
rows, the LAMA arms on 80%. The closer reference is exp02 `A_def` (default LightGBM through AutoGluon, same
folds, the same 80% outer training rows): pol 4.30, socmob 18.00. Against it `lama` trails (4.91 against 4.30 on pol, 20.59 against 18.00 on socmob).
`A_def` does not fit on all of those rows: without bagging AutoGluon holds part of them out for early stopping, by its
default rule (read from its source; not stored in the records) 10% on pol and 20% on socmob, so about 72% and 64% of
the data, close to the 68% of the exp04 base models. Neither reference matches the LAMA arms' rows exactly.

Arms, regression only, 5 folds each (`worker_lama_diag.py`): `lama_num` (all columns forced to numeric roles,
category codes as numbers), `lama_full` (`TabularUtilizedAutoML`) and `lama_num_full` (both). Per-dataset numbers:
`s6_lama_diag_*` and the `exp06_lama_diag_table` image.

- Numeric roles are what helps. pol: 4.91 to 4.07 (+17.2%, CI [14.7%, 19.5%]), a modest gain over `A_def` (4.30)
  and over both base GBMs. socmob: 20.59 to 17.98 (+12.0%, CI [2.2%, 23.3%]), level with `A_def` (18.00). Forcing
  numeric roles does not stop LAMA from dropping columns: `lama_num` uses 26 of 48 columns on pol in every fold
  (the default arm does not record it). Which of the 22 dropped columns or the changed types explains the gain was
  not isolated.
- The full-budget variant did not use more budget on pol. `lama_full` returns the same predictions as `lama` (4.9146
  both) after 346 s of the 600 s with a single config (`conf_0`); why no further config started was not
  investigated, so the full-budget question is open on pol. On socmob it ran 560 s against 65 s and gains +2.6%
  (CI [0.4%, 4.6%]).
- Both together on socmob: 17.13 (+16.3% over `lama`, CI [6.9%, 25.8%]), below `A_def` (18.00) and the base LightGBM
  (18.29), above the base CatBoost (15.81) and `ag_best` (11.90). On pol the same arm equals `lama_num`.
- Summary: forced numeric roles bring `lama` from clearly behind the `A_def` LightGBM to about level or slightly
  better on both datasets; full budget adds a little on socmob only. `lama` as a tool stays behind `ag_best`, `h2o`
  and the base CatBoost on socmob.

The docstring of `worker_lama_diag.py` says forced numeric roles bypass the reader's column drops, and that `lama_full`
keeps training until the budget is spent; the records above show neither (26 of 48 columns used on pol, 346 of 600 s).
The file is hashed into the unit records, so the text stays as it was run.

Caveats: 2 multiclass datasets only; per-dataset CIs are over folds of one dataset and do not include the variation
between repeated time-limited runs of a tool (the `h2o_te` over `h2o` rows on datasets with nothing to encode show it
can reach 10% or more), and the bootstrap CIs come from one random stream over the contrast list, so a CI depends on the
contrast's position; default settings of
each tool, no tuning, one machine, one repeat. `lama` has no `best_quality` equivalent in this setup, and the
H2O and `ag_best` units overshoot the budget by up to about 30 s.

H2O probe: `probe_h2o.py` (h2o env, run by hand, no make target) trains a 45 s H2O AutoML on a typed frame and writes
`results/tidy/exp06_h2o_probe.csv`; the `s2_h2o_*` claims read it and state only per-algorithm facts, because the
number of models finished in 45 s depends on the machine.
