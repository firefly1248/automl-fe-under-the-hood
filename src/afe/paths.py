"""Repo paths. AFE_CACHE and AFE_RESULTS let tests use temp dirs."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(os.environ.get("AFE_CACHE", "~/.cache/afe")).expanduser()
RESULTS = Path(os.environ.get("AFE_RESULTS", ROOT / "results"))
