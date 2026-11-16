"""All three changes together against candidate-60's kernel (indexer_baseline.py): finite inputs
bit for bit and timing at four prefill depths, decode shapes and reindex candidates, then the
indexer and top-k tests (the non-finite test no longer xfails), the candidate NaN test and the
PDL tests. Exit 1 if any step fails."""
import subprocess
import sys

B = ["--baseline", "/b/indexer_baseline.py"]
STEPS = [
    ["python3", "/b/bench_prefill_indexer.py", *B],
    ["python3", "/b/bench_indexer_bound_dense.py", *B],
    ["python3", "/b/bench_candidates.py", *B],
    ["python3", "-m", "pytest", "-q", "-p", "tblib_stub", "-p", "offline_skip",
     "-p", "tests.kernels.deepseek_v41_pdl_sources",
     "tests/kernels/attention/test_deepseek_v41_tilelang.py", "-k", "indexer or topk"],
    ["python3", "-m", "pytest", "-q", "-p", "tblib_stub", "-p", "offline_skip",
     "/b/test_nan_candidates.py"],
    ["python3", "-m", "pytest", "-q", "-p", "tblib_stub", "-p", "offline_skip",
     "tests/kernels/test_deepseek_v41_tilelang_pdl.py"],
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
