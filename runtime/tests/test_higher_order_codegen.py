import re
import subprocess
import tempfile
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILER = REPO_ROOT / "build" / "compiler"


@pytest.mark.skipif(not COMPILER.exists(), reason="compiler binary is not built")
def test_tcsf_add_codegen_has_outer_sparse_nonempty_guards():
    """Regression for higher-order sparse row/plane completion guards.

    For 3D all-sparse add, both outer sparse levels must gate offset/count
    updates on whether the next sparse level produced at least one element.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            [str(COMPILER), "--emit", tmpdir, "--name", "tcsf_add"],
            check=True,
            capture_output=True,
            text=True,
        )
        src = (Path(tmpdir) / "tcsf_add.cu").read_text()

    # Loop level l0 writes j-level offsets and must ensure j gained work.
    assert re.search(r"count_j_row_start_l0\s*<\s*count_j", src)
    assert re.search(r"offset_j_row_start_l0\s*<\s*offset_j", src)

    # Loop level l1 writes k-level offsets and must ensure k gained work.
    assert re.search(r"count_k_row_start_l1\s*<\s*count_k", src)
    assert re.search(r"offset_k_row_start_l1\s*<\s*offset_k", src)
