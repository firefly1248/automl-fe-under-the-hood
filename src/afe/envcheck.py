"""`make env-check`: import each tool in its own env and write results/ENVIRONMENT.md.

    python -m afe.envcheck probe <env>   # inside an env: import tools, print versions as JSON
    python -m afe.envcheck report        # inside core: probe every env, write ENVIRONMENT.md
"""

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

from afe.machine import machine_info, strip_home, versions

ROOT = Path(__file__).resolve().parents[2]

COMMON = ["numpy", "pandas", "scikit-learn"]
ENVS = {
    "core": (["lightgbm", "catboost", "openml"], ["lightgbm", "catboost", "openml", "scipy", "matplotlib"]),
    "autogluon": (
        ["autogluon.tabular", "autogluon.features.generators"],
        ["autogluon.tabular", "autogluon.features", "autogluon.core", "lightgbm", "catboost", "xgboost", "torch"],
    ),
    "lama": (["lightautoml.automl.presets.tabular_presets", "statsmodels.api"], ["lightautoml", "torch", "lightgbm", "catboost", "statsmodels"]),
    "openfe": (["openfe"], ["openfe", "lightgbm"]),
    "autofeat": (["autofeat"], ["autofeat", "numba"]),
    "mljar": (["supervised.preprocessing.goldenfeatures_transformer"], ["mljar-supervised", "lightgbm", "xgboost"]),
}


def _autofeat_fit() -> None:
    """autofeat 2.1.3 breaks only when called (scikit-learn >= 1.8 dropped `force_all_finite`)."""
    from afe.data.synthetic import hard_columns
    from afe.generators import make

    X, y, _ = hard_columns(n=600, seed=0)
    X = X[["num", "a", "b", "city_code"]]
    g = make("autofeat")
    g.fit_transform(X, y, "binary", 0)
    g.transform(X)


def probe(env: str) -> None:
    modules, dists = ENVS[env]
    for m in modules:
        importlib.import_module(m)
    if env == "autofeat":  # a fit beyond imports
        _autofeat_fit()
    print(json.dumps(versions(dists + COMMON)))


def report() -> None:
    uv = subprocess.check_output(["uv", "--version"], text=True).strip()
    lines = ["# Environment", "", "Written by `make env-check`.", "", "## Machine", ""]
    lines += [f"- {k}: {v}" for k, v in machine_info().items()]
    lines += [f"- uv: {uv}", "", "## Envs", ""]
    failed = []
    for env in ENVS:
        proc = subprocess.run(
            ["uv", "run", "--frozen", "--project", f"envs/{env}", "python", "-m", "afe.envcheck", "probe", env],
            cwd=ROOT, capture_output=True, text=True,
            env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
        )
        lines.append(f"### {env}")
        lines.append("")
        if proc.returncode == 0:
            lines += [f"- {k}: {v}" for k, v in json.loads(proc.stdout.strip().splitlines()[-1]).items()]
        else:
            failed.append(env)
            lines += ["import FAILED:", "", "```", strip_home(proc.stderr[-3000:]), "```"]
        lines.append("")
    (ROOT / "results" / "ENVIRONMENT.md").write_text("\n".join(lines))
    print("failed envs:", failed or "none")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    probe(sys.argv[2]) if sys.argv[1] == "probe" else report()
