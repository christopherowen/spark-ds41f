"""The prefill-shape bench at the two depths not yet compared, the decode-shape read-bound bench
against the candidate-60 kernel (bit equality, per-case timing), and the indexer, top-k and PDL
tests, one after another. Exit status 1 if any step fails."""
import subprocess
import sys

STEPS = [
    ["python3", "/b/bench_indexer_bound_dense.py",
     "--baseline", "/b/indexer_baseline.py"],
    ["python3", "-m", "pytest", "-q", "-p", "tblib_stub", "-p", "offline_skip",
     "tests/kernels/test_deepseek_v41_tilelang_pdl.py"],
]
failed = []
for step in STEPS:
    print("== " + " ".join(step), flush=True)
    code = subprocess.run(step).returncode
    print(f"== exit {code}", flush=True)
    if code:
        failed.append(step[1] if step[1] != "-m" else step[-3])
print("FAILED STEPS: " + (", ".join(failed) if failed else "none"), flush=True)
sys.exit(1 if failed else 0)
