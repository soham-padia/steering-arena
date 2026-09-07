"""The published docs under web/docs/ must match the markdown in docs/.

The deploy squashes the whole tree, so a page edited without a rebuild ships the
previous HTML with no error anywhere. This test is the gate for that.

It skips rather than fails when markdown-it-py is missing: that is a build
dependency pinned in requirements-research.txt, and a venv built only from
requirements.txt (which is what the served image installs) does not have it.
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "build_docs_site.py"

pytest.importorskip("markdown_it", reason="markdown-it-py is a build-only dependency")


def test_web_docs_match_the_markdown():
    r = subprocess.run([sys.executable, str(BUILDER), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, (
        "web/docs/ is out of date with docs/. Run:\n"
        "    python scripts/build_docs_site.py\n\n"
        f"{r.stdout}{r.stderr}"
    )


def test_every_generated_page_is_committed():
    """A page the generator writes but git does not track never reaches the Space."""
    tracked = subprocess.run(["git", "ls-files", "web/docs"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.split()
    on_disk = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "web" / "docs").rglob("*.html"))
    untracked = sorted(set(on_disk) - set(tracked))
    assert not untracked, f"generated but untracked: {untracked}"
