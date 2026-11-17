"""NaN-keeping ReLU against the bf16x2 + prefetch kernel (indexer_baseline.py): finite inputs bit
for bit and timing at prefill and decode shapes, then the indexer and top-k tests with the
non-finite test no longer xfail, and the candidate-path NaN test. Exit 1 if any step fails."""
import subprocess
import sys

STEPS = [
    ["python3", "/b/bench_prefill_indexer.py", "--baseline", "/b/indexer_baseline.py",
     "--contexts", "237000,910000"],
    ["python3", "/b/bench_indexer_bound_dense.py", "--baseline", "/b/indexer_baseline.py"],
    ["python3", "-m", "pytest", "-q", "-p", "tblib_stub", "-p", "offline_skip",
     "-p", "tests.kernels.deepseek_v41_pdl_sources",
     "tests/kernels/attention/test_deepseek_v41_tilelang.py", "-k", "indexer or topk"],
    ["python3", "-m", "pytest", "-q", "-p", "tblib_stub", "-p", "offline_skip",
     "/b/test_nan_candidates.py"],
]
failed = []
for step in STEPS:
    print("== " + " ".join(step), flush=True)
    code = subprocess.run(step).returncode
    print(f"== exit {code}", flush=True)
    if code:
        failed.append(" ".join(step[1:3]))
print("FAILED STEPS: " + (", ".join(failed) if failed else "none"), flush=True)
sys.exit(1 if failed else 0)
