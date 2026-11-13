# usage: python3 mkvar.py VARDIR
# Builds the diagnosis variants of gemm.py and vocab.py from the base (candidate-50
# without #1009) and #963 (after 0782) files exported into VARDIR:
#   gemm_pdl.py   base + pdl.chain() in _packed_bf16_gemm and packed_bf16_gemm_two
#   gemm_rb.py    base + vllm_packed_bf16_load_row_bounds in those two kernels
#                 (vllm_packed_bf16_filter_exc unchanged)
#   vocab_exc.py  #963 with the base's vllm_vocab_decode_vec16 (new exceptions, old decode)
#   vocab_dec.py  base with #963's vllm_vocab_decode_vec16 (old exceptions, new decode)
import sys

d = sys.argv[1]
rd = lambda n: open(f"{d}/{n}").read()  # noqa: E731
gb, g9, vb, v9 = rd("gemm_base.py"), rd("gemm_963.py"), rd("vocab_base.py"), rd("vocab_963.py")


def one(s, old, new, count):
    assert s.count(old) == count, (old[:60], s.count(old))
    return s.replace(old, new)


# gemm: pdl.chain() only
anchor = "            T.import_source(_PACKED_BF16_HEADER)\n            if blocked:\n"
# the first two are _packed_bf16_gemm and packed_bf16_gemm_two; the third,
# packed_bf16_gemm_partials, already chains before its import_source
assert gb.count(anchor) == 3 and gb.index(anchor, gb.index("def packed_bf16_gemm_partials(")) > 0
assert gb.index(anchor, gb.index(anchor, gb.index(anchor) + 1) + 1) > gb.index("def packed_bf16_gemm_partials(")
g_pdl = gb.replace(anchor, "            T.import_source(_PACKED_BF16_HEADER)\n            pdl.chain()\n            if blocked:\n", 2)
assert g9.count("            T.import_source(_PACKED_BF16_HEADER)\n            pdl.chain()\n            if blocked:\n") == 2

# gemm: row bounds only
rb_fn_start = g9.index("__device__ __forceinline__ void vllm_packed_bf16_load_row_bounds(")
rb_fn_end = g9.index("__device__ __forceinline__ void vllm_packed_bf16_filter_exc(")
rb_fn = g9[rb_fn_start:rb_fn_end]
g_rb = one(gb, "__device__ __forceinline__ void vllm_packed_bf16_filter_exc(", rb_fn + "__device__ __forceinline__ void vllm_packed_bf16_filter_exc(", 1)
old_call = '''                        "vllm_packed_bf16_filter_exc",
                        T.address_of(exception_entries[0]),
                        E,
                        T.address_of(row_offsets[0]),
                        r_idx,
                        0,
                        K,
'''
new_call = '''                        "vllm_packed_bf16_load_row_bounds",
                        E,
                        T.address_of(row_offsets[0]),
                        r_idx,
'''
g_rb = one(g_rb, old_call, new_call, 2)
assert g9.count(new_call) == 2


def decode_block(s, end_marker):
    a = s.index("template <int BASE_MINUS_ONE>\n__device__ __forceinline__ void vllm_vocab_decode_vec16(")
    b = s.index(end_marker, a)
    return a, b


a9, b9 = decode_block(v9, "template <int BM, int BN, int THREADS, int N_ROWS, int K_COLS>\n__device__ __forceinline__ void vllm_vocab_apply_row_exceptions(")
ab, bb = decode_block(vb, "__device__ __forceinline__ int vllm_vocab_first_argmax_keep(")
dec9, decb = v9[a9:b9], vb[ab:bb]
assert "BASE_BYTE" in dec9 and "BASE_SCALAR" in decb
v_exc = v9[:a9] + decb + v9[b9:]
v_dec = vb[:ab] + dec9 + vb[bb:]
for n, s in [("gemm_pdl.py", g_pdl), ("gemm_rb.py", g_rb), ("vocab_exc.py", v_exc), ("vocab_dec.py", v_dec)]:
    open(f"{d}/{n}", "w").write(s)
    print("wrote", n, len(s))
