import gzip
import json

from afe.run import pack, store


def _rec(h: str) -> dict:
    return {"hash": h, "stage": "gen", "status": "ok", "wall_s": 1.5, "result": {"x": [1, None, 0.1]}}


def _lines(gz) -> list[dict]:
    return [json.loads(line) for line in gzip.open(gz, "rt")]


def test_pack_twice_no_duplicates_and_tmp_untouched(tmp_path):
    d = tmp_path / "exp" / "gen"
    for h in ("b", "a"):
        store.write_json(d / f"{h}.json", _rec(h))
    tmp = d / "c.json.tmp"
    tmp.write_bytes(b"partial")
    pack.pack(tmp_path)
    first = (tmp_path / "exp" / "gen.jsonl.gz").read_bytes()
    pack.pack(tmp_path)
    assert (tmp_path / "exp" / "gen.jsonl.gz").read_bytes() == first
    assert [r["hash"] for r in _lines(tmp_path / "exp" / "gen.jsonl.gz")] == ["a", "b"]
    assert tmp.read_bytes() == b"partial"


def test_pack_merges_archive_and_unpack_restores_bytes(tmp_path):
    d = tmp_path / "exp" / "gen"
    store.write_json(d / "a.json", _rec("a"))
    original = (d / "a.json").read_bytes()
    pack.pack(tmp_path)
    (d / "a.json").unlink()  # a clone that was never unpacked
    store.write_json(d / "b.json", _rec("b"))
    pack.pack(tmp_path)
    assert [r["hash"] for r in _lines(tmp_path / "exp" / "gen.jsonl.gz")] == ["a", "b"]
    (d / "c.json.tmp").write_bytes(b"partial")
    pack.unpack(tmp_path)
    assert (d / "a.json").read_bytes() == original
    assert (d / "c.json.tmp").read_bytes() == b"partial"
