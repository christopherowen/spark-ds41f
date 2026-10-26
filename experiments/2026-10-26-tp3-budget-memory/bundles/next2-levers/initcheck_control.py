#!/usr/bin/env python3
"""The positive control for test_indexer_reads_only_visible_rows's candidate case: the
indexer with only its candidate count bound removed (``_candidate_position``'s
``j < count`` select, 0507), scored under compute-sanitizer's initcheck by
deepseek_v41_indexer_reads.py.

Past a row's count, the last kept block's ids point past the row's end, at index-cache
slots the helper never writes; without the bound the score kernel loads them, so initcheck
must report errors. The check passes when it does.

usage: initcheck_control.py SANITIZER VLLM_ROOT
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BOUND = "T.Select(j < count, offset, 0)"
UNBOUNDED = "offset"


def main() -> None:
    sanitizer, root = sys.argv[1], Path(sys.argv[2])
    source = (root / "vllm/models/deepseek_v4_1/tilelang/indexer.py").read_text()
    assert source.count(BOUND) == 1, "the candidate count bound is not in this indexer"
    with tempfile.TemporaryDirectory() as tmp:
        control = Path(tmp) / "indexer.py"
        control.write_text(source.replace(BOUND, UNBOUNDED))
        run = subprocess.run(
            [sanitizer, "--tool", "initcheck", "--error-exitcode", "1", sys.executable,
             str(root / "tests/kernels/attention/deepseek_v41_indexer_reads.py"),
             "--module", str(control)],
            env={**os.environ, "PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
            capture_output=True, text=True, timeout=3600)
    report = run.stdout + run.stderr
    summary = [line for line in report.splitlines() if "ERROR SUMMARY" in line]
    caught = run.returncode != 0 and bool(summary) and " 0 errors" not in summary[-1]
    print(report[-3000:])
    print("RECORD " + json.dumps({"control exit": run.returncode,
                                  "summary": summary[-1] if summary else None,
                                  "caught": caught}))
    sys.exit(0 if caught else 1)


if __name__ == "__main__":
    main()
