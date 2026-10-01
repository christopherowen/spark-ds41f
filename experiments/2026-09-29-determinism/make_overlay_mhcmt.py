#!/usr/bin/env python3
"""Build overlay mhc-mt: B12X mHC lagged partial kernels with several tokens per CTA.

usage: make_overlay_mhcmt.py BUILD_B12X_DIR   (writes ~/spark3-overlay/mhc-mt2/b12x/norm/mhc/)

BUILD_B12X_DIR must be the serving image's b12x package (r5o: /opt/spark3/candidate/b12x/b12x).
Overlay mhc-mt was built from a pre-r5o tree (no TF32 projection proxy fences) and with a config
codec the tuning contract rejects (run63 try 1); mhc-mt2 is built from the r5o image.

The native lagged partial kernel launches one CTA per (hidden tile, partial group, token); every
CTA reloads its slice of the layer's 24x4H FP32 mixing weights, about 2.8 GB of L2 reads per call
at 1334 rows. MHCPostPrePartialMultiTokenKernel loads that slice into registers once and then
computes tokens_per_cta tokens in turn, each exactly as the one-token kernel does (the same FMA
expression per thread, the same warp reduction, warps summed in the same order), with a barrier
between tokens for the shared warp sums. MhcConfig gains tokens_per_cta (default 1; every config
payload carries it, under config schema version 5, as the tuning contract requires one field set);
the one-token kernels, their compile keys and every default plan are otherwise unchanged.
"""
import os
import re
import sys

SRC = os.path.join(sys.argv[1], "norm", "mhc")
OUT = os.environ.get("MHCMT_OUT", os.path.expanduser("~/spark3-overlay/mhc-mt2/b12x/norm/mhc"))
os.makedirs(OUT, exist_ok=True)
k = open(os.path.join(SRC, "_kernels.py")).read()

# 1. The multi-token kernel class, generated from the one-token class's methods.
cls_start = k.index("class MHCPostPrePartialKernel:")
cls_end = k.index("class MHCPostPreDecodeSplitNPartialKernel:")
cls = k[cls_start:cls_end]
call_start = cls.index("    @cute.jit\n    def __call__(")
kern_start = cls.index("    @cute.kernel\n    def kernel(")
call = cls[call_start:kern_start]
kern = cls[kern_start:].rstrip("\n") + "\n"

old = "        self.kernel(\n            x, residual, prev_post, prev_comb, fn, partials, out, pre_mix, y\n        ).launch("
assert old in call
call = call.replace(old, "        self.kernel(\n            x, residual, prev_post, prev_comb, fn, partials, out, pre_mix, y,\n            num_tokens,\n        ).launch(")
old = "                num_tokens,\n            ),\n            block="
assert old in call
call = call.replace(old, "                (num_tokens + Int32(self.tokens_per_cta - 1))\n                // Int32(self.tokens_per_cta),\n            ),\n            block=")

old = "        y: cute.Tensor,\n    ):\n        # Launched early"
assert old in kern
kern = kern.replace(old, "        y: cute.Tensor,\n        num_tokens: Int32,\n    ):\n        # Launched early")
old = "        hidden_tile, partial_group, token = cute.arch.block_idx()\n        token = Int64(token)\n"
assert old in kern
kern = kern.replace(old, "        hidden_tile, partial_group, token_block = cute.arch.block_idx()\n")
anchor = "        h = Int64(hidden_tile) * Int64(self.source_tile_h) + Int64(tidx)\n"
assert anchor in kern
head, body = kern.split(anchor)
tail_marker = "        if const_expr(_MHC_PDL):\n            cute.arch.sync_threads()\n            cute.arch.griddepcontrol_launch_dependents()\n"
assert body.rstrip("\n").endswith(tail_marker.rstrip("\n")), body[-400:]
body = body[: body.rindex(tail_marker)]
# The fn reads become reads of the per-thread cache (same values, same expression).
old = "                        value = Float32(fn[mix, h]) * r0\n"
assert old in body
body = body.replace(old, "                        value = fn_cache[slot * 4] * r0\n")
old = """                    value = (
                        Float32(fn[mix, h]) * r0
                        + Float32(fn[mix, Int64(self.hidden_size) + h]) * r1
                        + Float32(fn[mix, Int64(2 * self.hidden_size) + h]) * r2
                        + Float32(fn[mix, Int64(3 * self.hidden_size) + h]) * r3
                    )
"""
assert old in body
body = body.replace(old, """                    value = (
                        fn_cache[slot * 4] * r0
                        + fn_cache[slot * 4 + 1] * r1
                        + fn_cache[slot * 4 + 2] * r2
                        + fn_cache[slot * 4 + 3] * r3
                    )
""")
assert "fn[" not in body, "an fn read remains in the token body"
indented = "".join(("        " + line) if line.strip() else line for line in body.splitlines(True))
cache = """        # This thread's slice of the mixing weights, loaded once for every token.
        fn_cache = cute.make_rmem_tensor(
            cute.make_layout((self.partials_per_cta * 4,), stride=(1,)), Float32
        )
        for slot in cutlass.range_constexpr(self.partials_per_cta * 4):
            fn_cache[slot] = Float32(0.0)
        if const_expr(not self.post_only):
            for slot in cutlass.range_constexpr(self.partials_per_cta):
                partial = partial0 + Int32(slot)
                if partial > Int32(0):
                    if partial < Int32(self.partials):
                        mix = partial - Int32(1)
                        if const_expr(self.pre_only and cute.rank(residual) == 2):
                            fn_cache[slot * 4] = Float32(fn[mix, h])
                        else:
                            fn_cache[slot * 4] = Float32(fn[mix, h])
                            fn_cache[slot * 4 + 1] = Float32(
                                fn[mix, Int64(self.hidden_size) + h]
                            )
                            fn_cache[slot * 4 + 2] = Float32(
                                fn[mix, Int64(2 * self.hidden_size) + h]
                            )
                            fn_cache[slot * 4 + 3] = Float32(
                                fn[mix, Int64(3 * self.hidden_size) + h]
                            )
        for step in cutlass.range(self.tokens_per_cta, unroll=1):
            token = Int64(token_block) * Int64(self.tokens_per_cta) + Int64(step)
            if token < Int64(num_tokens):
"""
loop_end = """            # The next token reuses the shared warp sums.
            cute.arch.sync_threads()

"""
kern = head + anchor + cache + indented + loop_end + tail_marker
new_class = '''class MHCPostPrePartialMultiTokenKernel(MHCPostPrePartialKernel):
    """MHCPostPrePartialKernel computing tokens_per_cta tokens per CTA.

    Each thread loads its slice of the mixing weights once and computes every
    token of its CTA exactly as the one-token kernel does: the same per-thread
    expression, warp reduction and in-order warp sum, so a row's partial sums
    are bit-identical. Large capacities read the weights once per CTA instead
    of once per token.
    """

    def __init__(self, *, tokens_per_cta: int, **kwargs):
        super().__init__(**kwargs)
        tokens_per_cta = int(tokens_per_cta)
        if not 1 < tokens_per_cta <= 16:
            raise ValueError(f"tokens_per_cta must be in [2, 16], got {tokens_per_cta}")
        if self.post_only:
            raise ValueError("the multi-token partial kernel computes partial sums")
        self.tokens_per_cta = tokens_per_cta

''' + call + "\n" + kern + "\n\n"
k = k[:cls_end] + new_class + k[cls_end:]

# 2. The factory picks the class.
old = """    partials_per_cta: int = _POST_PRE_PARTIALS_PER_CTA,
) -> MHCPostPrePartialKernel:
    return MHCPostPrePartialKernel(
        hidden_size=hidden_size,
        split_k=split_k,
        compute_gram=compute_gram,
        pre_only=pre_only,
        post_only=post_only,
        lagged_mix=lagged_mix,
        partials_per_cta=partials_per_cta,
    )
"""
new = """    partials_per_cta: int = _POST_PRE_PARTIALS_PER_CTA,
    tokens_per_cta: int = 1,
) -> MHCPostPrePartialKernel:
    kwargs = dict(
        hidden_size=hidden_size,
        split_k=split_k,
        compute_gram=compute_gram,
        pre_only=pre_only,
        post_only=post_only,
        lagged_mix=lagged_mix,
        partials_per_cta=partials_per_cta,
    )
    if int(tokens_per_cta) > 1:
        return MHCPostPrePartialMultiTokenKernel(tokens_per_cta=int(tokens_per_cta), **kwargs)
    return MHCPostPrePartialKernel(**kwargs)
"""
assert old in k
k = k.replace(old, new)

# 3. post_pre launch: tokens_per_cta from the native launch record.
old = "    partials_per_cta = native.partials_per_cta\n"
assert k.count(old) == 1
k = k.replace(old, "    partials_per_cta = native.partials_per_cta\n    tokens_per_cta = int(getattr(native, \"tokens_per_cta\", 1))\n")
for spec in ('"integration.residual.mhc_post_pre_partial_hidden4096_hctile128"\n            f"_all{partials_per_cta}"',
             '"integration.residual.mhc_post_pre_partial_"\n            f"{hidden_specialization}_hctile128_all{partials_per_cta}"'):
    assert spec in k, spec
    k = k.replace(spec, spec + '\n            + (f"_t{tokens_per_cta}" if tokens_per_cta > 1 else "")')
old = """        kernel = _post_pre_partial_kernel(
            hidden_size,
            split_k,
            compute_gram,
            False,
            False,
            lagged_mix,
            partials_per_cta,
        )"""
assert k.count(old) == 2, k.count(old)
k = k.replace(old, old.replace("            partials_per_cta,\n        )", "            partials_per_cta,\n            tokens_per_cta,\n        )"))
# compile keys gain tokens_per_cta only when it is larger than one.
k = re.sub(r'(\("partials_per_cta", partials_per_cta\),\n)(\s+\("threads", _THREADS\),)',
           lambda m: m.group(1) + m.group(2).replace('("threads"', '*((("tokens_per_cta", tokens_per_cta),) if tokens_per_cta > 1 else ()),\n' + " " * (len(m.group(2)) - len(m.group(2).lstrip())) + '("threads"', 1), k)

# 4. pre launch: an explicit tokens_per_cta argument.
old = "    partials_per_cta: int,\n    _prepared=None,\n) -> None:\n    if (pre_mix is None) != (y is None):"
assert k.count(old) == 1
k = k.replace(old, "    partials_per_cta: int,\n    tokens_per_cta: int = 1,\n    _prepared=None,\n) -> None:\n    tokens_per_cta = int(tokens_per_cta)\n    if (pre_mix is None) != (y is None):")
for spec in ('"integration.residual.mhc_pre_partial_hidden4096_hctile128"\n            f"_all{partials_per_cta}"',
             '"integration.residual.mhc_pre_partial_"\n            f"{hidden_specialization}_hctile128_all{partials_per_cta}"'):
    assert spec in k, spec
    k = k.replace(spec, spec + '\n            + (f"_t{tokens_per_cta}" if tokens_per_cta > 1 else "")')
old = """        _post_pre_partial_kernel(
            hidden_size=hidden_size,
            split_k=split_k,
            compute_gram=compute_gram,
            pre_only=True,
            lagged_mix=lagged_mix,
            partials_per_cta=partials_per_cta,
        ),"""
assert k.count(old) == 1
k = k.replace(old, old.replace("partials_per_cta=partials_per_cta,\n", "partials_per_cta=partials_per_cta,\n            tokens_per_cta=tokens_per_cta,\n"))
compile(k, "_kernels.py", "exec")
open(os.path.join(OUT, "_kernels.py"), "w").write(k)

# 5. Config and preparation.
t = open(os.path.join(SRC, "_tuning.py")).read()
old = "    lagged_prepare: bool = False\n    partials_per_cta: int = 4\n"
assert old in t
t = t.replace(old, old + "    tokens_per_cta: int = 1\n")
old = """        expected = frozenset(cls.__dataclass_fields__)
        if frozenset(payload) != expected:
            raise ValueError(f"mHC config fields must be {sorted(expected)}")"""
assert old in t
t = t.replace(old, """        expected = frozenset(cls.__dataclass_fields__)
        if frozenset(payload) | {"tokens_per_cta"} != expected:
            raise ValueError(f"mHC config fields must be {sorted(expected)}")""")
old = """            partials_per_cta=int(payload["partials_per_cta"]),
        )"""
assert old in t
t = t.replace(old, """            partials_per_cta=int(payload["partials_per_cta"]),
            tokens_per_cta=int(payload.get("tokens_per_cta", 1)),
        )""")
old = """            "partials_per_cta": self.partials_per_cta,
        }"""
assert old in t
t = t.replace(old, """            "partials_per_cta": self.partials_per_cta,
            "tokens_per_cta": self.tokens_per_cta,
        }""")
old = "    config_schema_version=4,\n"
assert t.count(old) == 1
t = t.replace(old, "    config_schema_version=5,\n")
old = """    if not config.lagged_prepare and config.partials_per_cta != 4:
        raise ValueError("partial grouping requires the fused lagged producer")"""
assert old in t
t = t.replace(old, old + """
    if type(config.tokens_per_cta) is not int or not 1 <= config.tokens_per_cta <= 16:
        raise ValueError("tokens_per_cta must be an integer in [1, 16]")
    if config.tokens_per_cta != 1 and not config.lagged_prepare:
        raise ValueError("several tokens per CTA require the fused lagged producer")""")
compile(t, "_tuning.py", "exec")
open(os.path.join(OUT, "_tuning.py"), "w").write(t)

p = open(os.path.join(SRC, "_preparation.py")).read()
old = "    block_m: int = 2\n    tile_n: int = 24\n"
assert old in p
p = p.replace(old, old + "    tokens_per_cta: int = 1\n")
old = """    return _NativeLaunch(
        route, source_splits, tile_n, bf16x2, partials, threads, groups,
        int(controls.get("B12X_MHC_PREFILL_BLOCK_M_SIZE", "2")),
        int(controls.get("B12X_MHC_PREFILL_TILE_N", "12" if hidden == 7168 else "24")),
    )"""
assert old in p
p = p.replace(old, """    return _NativeLaunch(
        route, source_splits, tile_n, bf16x2, partials, threads, groups,
        int(controls.get("B12X_MHC_PREFILL_BLOCK_M_SIZE", "2")),
        int(controls.get("B12X_MHC_PREFILL_TILE_N", "12" if hidden == 7168 else "24")),
        config.tokens_per_cta if config.lagged_prepare and route in ("pre", "post_pre") else 1,
    )""")
old = "                    y=y if config.lagged_prepare else None, partials_per_cta=native.partials_per_cta,\n"
assert old in p
p = p.replace(old, "                    y=y if config.lagged_prepare else None, partials_per_cta=native.partials_per_cta,\n                    tokens_per_cta=native.tokens_per_cta,\n")
old = """            entry = partial(kernels._run_mhc_pre_partial_launch,
                            partials_per_cta=native.partials_per_cta)"""
assert old in p
p = p.replace(old, """            entry = partial(kernels._run_mhc_pre_partial_launch,
                            partials_per_cta=native.partials_per_cta,
                            tokens_per_cta=native.tokens_per_cta)""")
compile(p, "_preparation.py", "exec")
open(os.path.join(OUT, "_preparation.py"), "w").write(p)
print("wrote", OUT)
