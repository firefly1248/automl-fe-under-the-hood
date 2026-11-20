"""`openfe_transductive` hands the model its category columns with string levels (the parquet cache turns float levels into numbers)."""

import pandas as pd
import pytest

pytest.importorskip("openfe")

from afe.generators import openfe_gen, openfe_transductive  # noqa: E402


def test_category_levels_become_strings(monkeypatch):
    frame = pd.DataFrame({"combine": pd.Categorical([0.0, 1.0, 1.0]), "num": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(openfe_gen.OpenFETransductive, "fit_transform_all", lambda self, *a: (frame, frame))
    fit_out, other_out = openfe_transductive.OpenFETransductive().fit_transform_all(None, None, None, "binary", 0)
    assert list(fit_out["combine"].cat.categories) == ["0.0", "1.0"] and list(other_out["combine"].cat.categories) == ["0.0", "1.0"]
    assert fit_out["num"].dtype == "float64"
