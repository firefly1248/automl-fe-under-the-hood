# exp07 CatBoost ablation: built-in categorical combinations

Hypothesis (protocol change A4, approved 2026-11-17): CatBoost builds combinations of categorical features and target
statistics over them while it grows trees, so external category crossings and target encodings should help it less than
they help LightGBM. `max_ctr_complexity` is the largest number of categorical features in one combination; 1 switches the
combinations off.

Design, downstream only: CatBoost with its default `max_ctr_complexity` (the exp04 fits) against CatBoost with
`max_ctr_complexity=1`; everything else as in exp04 (3000 iterations, learning rate 0.1, early stopping 200 on the inner
rows, one thread, same seeds). Arms `base`, `ag_tabprep_keepcat`, `lama_adv`, `openfe` (the ones that build crossings or
target encodings); the 17 primary datasets with a categorical column of 10 or more levels; all 10 outer folds (2 repeats
of 5). Generator outputs come from the exp04 cache, so nothing is generated here: 680 new CatBoost fits
(`make exp07`; `worker_ctr.py` is the exp04 fit stage with the parameter set, `run.py` plans the units from the exp04 plan).
Default in the installed CatBoost (1.2.10, CPU): the library docstring says `max_ctr_complexity` default 4. Careful: a fitted
model with the parameter unset reports `get_all_params()["max_ctr_complexity"] == 1`, which is not what training uses.
Checked on credit-g (repeat 0, fold 0, base arm, protocol parameters of exp04): the unset setting and an explicit 4 give
the same test log loss to the last digit (0.5009686182654032, best iteration 132), `max_ctr_complexity=1` gives 0.4764858
(best iteration 184). So the exp04 fits are the "default = 4" side of the comparison.

Analysis (`afe.analysis.exp07`, in `make aggregate`; tables `results/tidy/exp07_effects.csv`, `exp07_summary.csv`; claims
`s4_ctr_*`; article table `ctr_ablation_table`). Per dataset and arm, over the folds where all four fits exist (default and
`ctr=1`, arm and base): the effect of the arm over `base` with the default CatBoost, the same with `ctr=1`, and their paired
difference `ctr=1 minus default`, as relative loss improvement (log loss, RMSE for regression) with a fold-bootstrap 95% CI;
counts better by sign and clearly better or worse (CI excludes zero). A positive difference means the arm helps more once
CatBoost has no combinations of its own. `base` alone: `ctr=1` over default, negative where CatBoost loses without its own
combinations.

## Results

680 planned units: 670 ok, 10 `no_generator` (openfe on fps_benchmark, whose generator failed in exp04, so openfe has 16 datasets
and the other arms 17). Mean wall time of a `ctr=1` fit: base 15.5 s, `lama_adv` 5.0 s, `openfe` 93.6 s, `ag_tabprep_keepcat` 159.3 s
(`s4_ctr_*`, `ctr_ablation_table`).

- `base` alone, `ctr=1` over default CatBoost: better on 3 of 17 datasets by sign, clearly worse on 8 (fold CI excludes zero),
  clearly better on 0; median -0.5%. Large losses without CatBoost's own combinations: fps_benchmark (RMSE 5.56 against 0.90,
  -537%), socmob -25.8%, Amazon_employee_access -11.0%.
- Paired difference `ctr=1` minus default (the arm helps more once CatBoost has no combinations of its own), better by sign /
  clearly better / clearly worse: `ag_tabprep_keepcat` 13 / 6 / 0 of 17, `lama_adv` 13 / 8 / 0 of 17, `openfe` 12 / 6 / 2 of 16. The
  median shift is about half a point (+0.5, +0.5 and +0.3 points; secondary figures), so the sign counts are lopsided but the typical
  size is small; the large differences sit on socmob, Amazon_employee_access and fps_benchmark, where `base` also loses most
  without combinations. Brazilian_houses has huge fold-to-fold variation and no clear result.
- Reading: on these datasets the arms that add crossings or target encodings help CatBoost more when its own combinations are off,
  which fits the hypothesis, but turning the combinations off also hurts `base` itself on most datasets, so the comparison is of
  two different CatBoosts, and 17 datasets with one default setting of everything else do not say how general it is.
