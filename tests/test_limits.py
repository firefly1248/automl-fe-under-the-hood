import sys

from afe.run import limits


def _run(code, timeout_s=30, mem_gb=2):
    return limits.run([sys.executable, "-c", code], timeout_s, mem_gb, ".", {})


def test_ok_and_failed():
    assert _run("print(1)").status == "ok"
    r = _run("raise ValueError('boom')")
    assert r.status == "failed" and "boom" in r.error


def test_timeout():
    r = _run("import time; time.sleep(30)", timeout_s=1)
    assert r.status == "timeout" and r.wall_s < 10


def test_oom():
    r = _run("import time; x = bytearray(600 * 2**20); x[::4096] = b'1' * len(x[::4096]); time.sleep(30)", mem_gb=0.3)
    assert r.status == "oom" and r.peak_rss_mb > 300
