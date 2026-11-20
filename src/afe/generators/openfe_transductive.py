"""`openfe_transductive` (exp05 check 3 only): `OpenFETransductive` of `openfe_gen` with the category levels of its output renamed to
strings.

OpenFE's own `transform` returns its `Combine` features as `category` columns with float levels. The parquet cache between generator
and fit stores such a column as plain `double`, so the model would get the column as a number while the inductive arm (string
levels) keeps it categorical; the same loss as for `ag_default`. Kept in its own module so that `openfe_gen.py`, and with it the
hash of the `openfe` arm, stays as it was.
"""

import pandas as pd

from afe.generators.openfe_gen import OpenFETransductive as _OpenFETransductive


class OpenFETransductive(_OpenFETransductive):
    def fit_transform_all(self, X_fit, y_fit, X_other, task, seed) -> tuple[pd.DataFrame, pd.DataFrame]:
        return tuple(o.apply(lambda s: s.cat.rename_categories(str) if isinstance(s.dtype, pd.CategoricalDtype) else s)
                     for o in super().fit_transform_all(X_fit, y_fit, X_other, task, seed))
