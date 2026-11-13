# Prints, as the container sees them, the SHA-256 of the files the corrected candidate-51
# arm mounts and the markers of each change (round 1 of #963 and #1001 included), and
# where the modules import from. Adapted from the own-mark run's verify_overlay.py.
import hashlib
import inspect

root = "/opt/spark-ds41f/candidate/vllm/"
files = [l.strip() for l in open("/h/files-c51r1.txt") if l.strip()]
for f in files:
    print(hashlib.sha256(open(root + f, "rb").read()).hexdigest()[:12], f)

from vllm.models.glm5next.nvidia import l2_prefetch as glm
from vllm.models.glm5next.nvidia.tilelang import l2_prefetch as tl
from vllm.models.deepseek_v4_1.tilelang import gemm, vocab, wo
from vllm.models.deepseek_v4_1 import attention, sp_prefill
from vllm.models.deepseek_v4_1.tilelang import front, cache_writer

hdr = vocab._GREEDY_HEADER
print("#961 glm _CHUNK_BYTES:", getattr(glm, "_CHUNK_BYTES", None), "_GRID:", glm._GRID,
      "tilelang paced kernel:", "l2_prefetch_segments_paced" in inspect.getsource(tl))
print("#963 vllm_packed_bf16_load_row_bounds in gemm header:",
      "vllm_packed_bf16_load_row_bounds" in gemm._PACKED_BF16_HEADER,
      "_PACKED_WIDE_TILES:", getattr(gemm, "_PACKED_WIDE_TILES", None),
      "filter_exc early return (base):", "if (num_exceptions <= 0) {" in gemm._PACKED_BF16_HEADER,
      "vllm_vocab_apply_row_exceptions in vocab:", "vllm_vocab_apply_row_exceptions" in hdr,
      "decode_vec16 halfword (base) BASE_SCALAR:", "BASE_SCALAR" in hdr, "byte SWAR BASE_BYTE:", "BASE_BYTE" in hdr)
print("#987 WIDE_DECODE[(5120,2048)]:", gemm.WIDE_DECODE[(5120, 2048)],
      "DECODE_FP8_CONFIGS[(5120,2048)] keys:", sorted(gemm.DECODE_FP8_CONFIGS[(5120, 2048)]))
print("#1001 TILE_CONFIGS[128]:", vocab.TILE_CONFIGS[128])
print("#1001-r1 SCREEN_TILE_CONFIGS:", vocab.SCREEN_TILE_CONFIGS)
print("#1009 (0778) present (front.sp_attention_front):", hasattr(front, "sp_attention_front"))
for m in (glm, tl, gemm, vocab, wo, attention, sp_prefill, front, cache_writer):
    print("module file:", m.__file__)
