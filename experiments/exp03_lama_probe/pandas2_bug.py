"""Minimal reproduction: LightAutoML 0.4.2 under pandas >= 2 turns every string categorical into NaN.

    uv run --project envs/lama python experiments/exp03_lama_probe/pandas2_bug.py

`OrdinalEncoder.fit` (`transformers/categorical.py:1053`) reads the `count` column of
`value_counts().reset_index()` as the category values under pandas >= 2, so the learned mapping has
counts as keys and no real value matches it.
"""

import numpy as np
import pandas as pd
from lightautoml.pipelines.features.lgb_pipeline import LGBSimpleFeatures
from lightautoml.reader.base import PandasToPandasReader
from lightautoml.tasks import Task

rng = np.random.default_rng(0)
cat = rng.choice(list("abcde"), 1000)
df = pd.DataFrame({"cat": cat, "x": rng.normal(size=1000), "y": (cat == "a").astype(int)})

data = PandasToPandasReader(Task("binary"), cv=2, random_state=0).fit_read(df, roles={"target": "y"})
out = LGBSimpleFeatures().fit_transform(data).to_numpy()
enc = out.data[:, out.features.index("ord__cat")]
print(f"pandas {pd.__version__}: share of NaN in the encoded `cat` = {np.isnan(enc).mean():.2f}")
