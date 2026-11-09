# Prints the SHA-256 of every candidate-47 file as the container sees it and checks
# that the patched modules import from the overlaid sources.
import hashlib, inspect, sys
root = "/opt/spark-ds41f/candidate/vllm/"
files = [l.strip() for l in open("/b/changed-files.txt") if l.strip()]
for f in files:
    print(hashlib.sha256(open(root + f, "rb").read()).hexdigest()[:12], f)
from vllm.models.deepseek_v4_1.tilelang import moe, gemm, sparse_mla, vocab
print("grouped_fp8_fp4_gemm has eviction_policy:", "eviction_policy" in inspect.signature(moe.grouped_fp8_fp4_gemm).parameters)
print("vocab has _ensure_screen_weights:", hasattr(vocab, "_ensure_screen_weights"))
print("gemm TILE_CONTIGUOUS_FP8:", sorted(gemm.TILE_CONTIGUOUS_FP8))
print("module files:", moe.__file__, gemm.__file__, sparse_mla.__file__, vocab.__file__)
