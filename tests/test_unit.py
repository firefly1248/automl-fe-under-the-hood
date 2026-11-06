import pytest

from afe.generators import ARMS
from afe.run import cli
from afe.run.unit import code_hash


def test_code_hash_rejects_files_of_one_name(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "m.py").write_text(d)
    with pytest.raises(AssertionError, match="share a name"):
        code_hash(tmp_path)


def test_hashed_code_has_unique_names():
    code_hash(*cli.FIT_CODE)
    for arm in ARMS:
        cli.arm_code(arm)
