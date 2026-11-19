# exp08: skrub TableVectorizer arm and probe

Protocol change A4 (approved 2026-11-17). Arm `skrub_tv` of exp04 (`src/afe/generators/skrub_tv.py`, env `envs/skrub`,
skrub 0.11.0, the latest stable on PyPI when the arm was added, pinned in `envs/skrub/uv.lock`): a third implementation of
"type guessing + encoding" next to `ag_default` and `lama`, in the scikit-learn ecosystem. Docs:
https://skrub-data.org/stable/reference/generated/skrub.TableVectorizer.html

- `TableVectorizer()` with all defaults, fitted on the fit rows only and applied to the inner-validation and test rows.
  It does not use `y`, so there is no out-of-fold step.
- `replaces_input=True` as for `ag_default`: the downstream model gets only the vectorized table. The output is all
  float32, so no categorical column reaches CatBoost; that is part of what the arm measures (CatBoost's own categorical
  handling is switched off for this arm).
- The only non-default thing is `np.random.seed(seed)` before the fit: the default `StringEncoder` has
  `random_state=None` and its SVD would otherwise differ between runs.
- Width of the output: one column per level of each string column below 40 unique values (one for a binary column), 30
  columns per string column with 40 or more unique values, and for a date column year, month, day and total seconds.

Defaults of the installed version (read by `make exp08`, `results/tidy/exp08_skrub_defaults.yaml`): `cardinality_threshold`
40 (strictly fewer unique values: low cardinality); low cardinality `OneHotEncoder(drop='if_binary', dtype='float32',
handle_unknown='ignore')`; high cardinality `StringEncoder` with 30 components, TF-IDF over character 3-4-grams
(`char_wb`); numeric `PassThrough` (cast to float32; strings that hold only numbers are converted, 'n/a'-like strings
become NaN); datetime `DatetimeEncoder` with resolution 'hour', no weekday, total seconds added; `drop_null_fraction`
1.0 (only all-null columns are dropped), `drop_if_constant` False. Unseen categories give an all-zero one-hot block.

`make exp08` runs `run.py` in the skrub env: one fit on `hard_columns()` and on the typed datasets of the exp03 probe
(adult, bank-marketing, credit-g, Amazon_employee_access, Amazon_employee_access_int, diamonds, house_sales_42092) and
writes `results/tidy/exp08_skrub_types.csv`, one row per input column (dtype, unique values, the kind skrub assigned,
processing steps, outputs, action). On `hard_columns()`: numbers stored as strings and the 'n/a' column become numeric,
the date string becomes four columns, the id column and the free text each become 30 SVD columns, no column is dropped
(constant and duplicate columns stay), all 114 probed columns come out as float32.

The arm runs in exp04 on all 50 datasets, both repeats, both models, with the limits of the other arms; its output sits
in the same plan as the other arms (`arms` in the exp04 config), so the results appear in the exp04 tables.
