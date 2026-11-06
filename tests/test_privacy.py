"""No tracked file, including the records inside *.jsonl.gz, carries a home path, a temp path, the user name
or the host name of this machine (plan rule 7)."""

import getpass
import gzip
import socket
import subprocess

from afe.machine import CUT_HOME, HOME, strip_home
from afe.paths import ROOT

BANNED = ["/Users/", "/private/", getpass.getuser(), socket.gethostname().split(".")[0]]
PATTERN_FILES = {"src/afe/machine.py", "tests/test_privacy.py"}  # they spell the patterns out


def test_strip_home_catches_cut_paths():
    assert strip_home(HOME + "/Projects/auto_fe/a.py") == "~/Projects/auto_fe/a.py"
    assert strip_home('File "/Users/ab\n[...]\ncd/Projects/auto_fe/a.py') == 'File "~\n[...]\n~/Projects/auto_fe/a.py'


def test_tracked_files_have_no_private_strings():
    hits = []
    for f in subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split():
        path = ROOT / f
        if f in PATTERN_FILES or not path.is_file():  # not is_file: deleted in the tree, not yet in the index
            continue
        data = path.read_bytes()
        text = (gzip.decompress(data) if f.endswith(".gz") else data).decode(errors="ignore")
        hits += [(f, b) for b in BANNED if b in text] + [(f, m) for m in set(CUT_HOME.findall(text))]
    assert not hits
