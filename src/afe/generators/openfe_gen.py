"""`openfe` arm: OpenFE's search and selection, with an inductive transform (plan sections 1.6, 6).

OpenFE's own `transform(train, test)` computes column-wide statistics (freq, Combine, GroupByThen*) on
train and test together. Here `fit` is OpenFE's (library defaults, `task` explicit, `seed`, `n_jobs=1`),
and the selected features are evaluated by the code below: row-wise operators by OpenFE's own
`Node.calculate`; column-wide operators with statistics fitted on fit rows only and mapped to other rows
(unseen key -> NaN). OpenFE's conventions are kept: raw counts, `std` with ddof=1, pct rank, `Combine`
codes in order of first appearance. `GroupByThenRank` of a new row = share of fit-row values of its group
below the value plus half of those equal to it.

Kept features: stage-2 gain > 0 (the positive-score rule OpenFE applies at stage 1), at most 2,000.
Columns are renamed c0000, c0001, ... because OpenFE parses formulas on `()+-*/,`; names are mapped back.
OpenFE calls exit() on errors inside its process pool; that is turned into an exception.
"""

import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from openfe import OpenFE
from openfe.FeatureGenerator import FNode, num_num_operators, num_operators
from openfe.utils import tree_to_formula

from afe.generators.base import FeatureGenerator

CAP = 2000
ROW_WISE = set(num_operators + num_num_operators)
AGG = {"GroupByThenMin": "min", "GroupByThenMax": "max", "GroupByThenMean": "mean",
       "GroupByThenMedian": "median", "GroupByThenStd": "std", "GroupByThenNUnique": "nunique"}
TOKEN = re.compile(r"c\d{4}")


def prep(X: pd.DataFrame) -> pd.DataFrame:
    """bool -> int8, datetime -> int64 ns, other non-numeric -> str (NaN kept); columns renamed c0000..."""
    out = {}
    for i, c in enumerate(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(np.int8)
        elif pd.api.types.is_datetime64_any_dtype(s):
            s = s.astype("int64").where(s.notna())
        elif not pd.api.types.is_numeric_dtype(s):
            s = s.astype(object).where(s.isna(), s.astype(str))
        out[f"c{i:04}"] = s
    return pd.DataFrame(out, index=X.index)


def _combine(d1: pd.Series, d2: pd.Series) -> pd.Series:
    key = d1.astype(str) + "_" + d2.astype(str)
    return key.where(d1.notna() & d2.notna())


def _rank_new(d1: pd.Series, d2: pd.Series, groups: dict) -> pd.Series:
    out = pd.Series(np.nan, index=d1.index)
    for k, rows in d2.groupby(d2).groups.items():
        if k in groups:
            v, x = groups[k], d1.loc[rows].to_numpy(float)
            r = (np.searchsorted(v, x, "left") + np.searchsorted(v, x, "right")) / 2 / len(v)
            out.loc[rows] = np.where(np.isnan(x), np.nan, r)
    return out


class Inductive:
    """Fit-rows statistics for first-order OpenFE feature trees (children are columns)."""

    def __init__(self, nodes: list):
        self.nodes, self.stats = nodes, {}

    def fit_transform(self, X: pd.DataFrame) -> dict:
        out = {}
        for i, n in enumerate(self.nodes):
            d = [X[ch.name] for ch in n.children]
            if n.name in ROW_WISE:
                continue
            if n.name == "freq":
                self.stats[i] = d[0].value_counts()
            elif n.name == "CombineThenFreq":
                self.stats[i] = _combine(*d).value_counts()
            elif n.name == "Combine":
                self.stats[i] = pd.Index(_combine(*d).dropna().unique())  # factorize order, as OpenFE
            elif n.name == "GroupByThenRank":
                self.stats[i] = {k: np.sort(v.dropna().to_numpy(float)) for k, v in d[0].groupby(d[1])}
                out[i] = d[0].groupby(d[1]).rank(ascending=True, pct=True).astype(float)
            else:
                self.stats[i] = d[0].groupby(d[1]).agg(AGG[n.name])
        return {i: out[i] if i in out else self._value(i, X) for i in range(len(self.nodes))}

    def _value(self, i: int, X: pd.DataFrame) -> pd.Series:
        n = self.nodes[i]
        if n.name in ROW_WISE:
            return n.calculate(X)
        d = [X[ch.name] for ch in n.children]
        if n.name == "freq":
            return d[0].map(self.stats[i]).astype(float)
        if n.name == "CombineThenFreq":
            return _combine(*d).map(self.stats[i]).astype(float)
        if n.name == "Combine":
            return pd.Series(pd.Categorical(_combine(*d), categories=self.stats[i]), index=X.index)
        if n.name == "GroupByThenRank":
            return _rank_new(d[0], d[1], self.stats[i])
        return d[1].map(self.stats[i]).astype(float)

    def transform(self, X: pd.DataFrame) -> dict:
        return {i: self._value(i, X) for i in range(len(self.nodes))}


class OpenFEInductive(FeatureGenerator):
    name = "openfe"
    env = "openfe"

    def _search(self, X_fit, y_fit, task, seed) -> tuple[pd.DataFrame, list]:
        Xp = prep(X_fit)
        self.names = dict(zip(Xp.columns, X_fit.columns))
        cats = [c for c in Xp.columns if not pd.api.types.is_numeric_dtype(Xp[c])]
        self.ofe = OpenFE()
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            try:
                self.ofe.fit(data=Xp, label=y_fit.to_frame(),
                             task="regression" if task == "regression" else "classification",
                             categorical_features=cats, n_jobs=1, seed=seed, verbose=False,
                             tmp_save_path=str(Path(tmp) / "openfe.feather"))
            except SystemExit as e:  # OpenFE prints the traceback, then calls exit() in the pool
                raise RuntimeError("OpenFE called exit(); its traceback is in the log above") from e
        scored = self.ofe.new_features_scores_list
        kept = [node for node, gain in scored if gain > 0][:CAP]
        self.info = {"stage1_survivors": len(scored), "kept": len(kept), "cap_fired": bool(sum(g > 0 for _, g in scored) > CAP),
                     "gains": {self._name(n): float(g) for n, g in scored if g > 0}}
        return Xp, kept

    def fit_transform(self, X_fit, y_fit, task, seed) -> pd.DataFrame:
        Xp, kept = self._search(X_fit, y_fit, task, seed)
        self.ind = Inductive(kept)
        return self._frame(self.ind.fit_transform(Xp), X_fit.index)

    def _name(self, node) -> str:
        return TOKEN.sub(lambda m: self.names[m.group()], tree_to_formula(node))

    def _frame(self, values: dict, index) -> pd.DataFrame:
        cols = {self._name(n): values[i] for i, n in enumerate(self.ind.nodes)}
        return pd.DataFrame(cols, index=index)

    def transform(self, X) -> pd.DataFrame:
        return self._frame(self.ind.transform(prep(X)), X.index)

    def describe(self) -> dict:
        return self.info


class OpenFETransductive(OpenFEInductive):
    """`openfe_transductive` (exp05 check 3 only, plan section 6): the same search, then OpenFE's own
    `transform(train, test)`, with column-wide statistics over fit and other rows together. The worker calls
    `fit_transform_all` once with inner and test rows as `X_other`."""

    name = "openfe_transductive"
    transductive = True

    def fit_transform_all(self, X_fit, y_fit, X_other, task, seed) -> tuple[pd.DataFrame, pd.DataFrame]:
        Xp, kept = self._search(X_fit, y_fit, task, seed)
        with tempfile.TemporaryDirectory(dir=".") as tmp:
            self.ofe.tmp_save_path = str(Path(tmp) / "openfe.feather")
            fit_out, other_out = self.ofe.transform(Xp, prep(X_other), kept, n_jobs=1)
        cols = [self._name(n) for n in kept]
        return tuple(o.iloc[:, Xp.shape[1]:].set_axis(cols, axis=1) for o in (fit_out, other_out))
