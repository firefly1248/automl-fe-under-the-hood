"""exp05 check 5: max abs difference per OpenFE operator between the inductive evaluator and OpenFE's own
`Node.calculate` on fit rows, on the data of contract test 7. Writes results/tidy/exp05_openfe_ops.csv.

    AFE_ENV=openfe uv run --project envs/openfe python experiments/exp05_leakage_checks/openfe_ops.py
"""

from copy import deepcopy

import numpy as np
import pandas as pd
from openfe import get_candidate_features

from afe.data.synthetic import hard_columns
from afe.generators.openfe_gen import Inductive, prep
from afe.paths import RESULTS

COLS = ["num", "city_code", "a", "b", "cat_low", "cat_rare", "cat_binary"]  # as in test 7

X, _, _ = hard_columns(n=600, seed=0)
X = X[COLS]
X.loc[X.sample(30, random_state=0).index, ["a", "cat_low"]] = np.nan
Xp = prep(X)
cats = [c for c in Xp.columns if not pd.api.types.is_numeric_dtype(Xp[c])]
num = [c for c in Xp.columns if c not in cats and Xp[c].nunique() > 100]
ordinal = [c for c in Xp.columns if c not in cats and c not in num]
nodes = get_candidate_features(numerical_features=num, categorical_features=cats, ordinal_features=ordinal)
fit = Xp.iloc[:400]
ours = Inductive(nodes).fit_transform(fit)

rows = []
for i, n in enumerate(nodes):
    ref = deepcopy(n).calculate(fit, is_root=True).astype(float).to_numpy()
    o = ours[i]
    o = o.cat.codes.to_numpy(float) if isinstance(o.dtype, pd.CategoricalDtype) else o.to_numpy(float)
    both_nan = np.isnan(o) & np.isnan(ref)
    nan_mismatch = int((np.isnan(o) != np.isnan(ref)).sum())
    rows.append({"operator": n.name, "max_abs_diff": float(np.nanmax(np.abs(o - ref)[~both_nan], initial=0.0)),
                 "nan_mismatch": nan_mismatch})
out = pd.DataFrame(rows).groupby("operator").agg(n_features=("max_abs_diff", "size"), max_abs_diff=("max_abs_diff", "max"),
                                                 nan_mismatch=("nan_mismatch", "sum")).reset_index()
out.to_csv(RESULTS / "tidy" / "exp05_openfe_ops.csv", index=False)
print(out.to_string(index=False))
