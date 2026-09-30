#!/usr/bin/env python3
"""Write the GEMV lookup arms (run from the repository root, config = r5o).

Every arm pins the r5o screen's adaptive-verification cost table, so the
verification policy is the same in all of them.

detm-r5o-lookup-trace: detm-r5o-trace's deterministic MoE (det-masked
overlay), plus the gemv-lookup overlay (the two files of
vllm-0027-gemv-smallest-capacity.patch) and the attn-trace debug overlay
(batch-trace plus attention checksums and the first-call router gate capture
on rank 0), all behind SPARK3_MOE_CHECKSUM_DIR / SPARK3_GATE_CAPTURE_ROWS. Its
KV cache is 256 MiB smaller than r5o's so the debug logs fit above dgx1's
5 GiB startup memory guard (the first boot with 128-row logs stopped 57 MB
short); the traced requests use a small fraction of it, and the cache size
moves page placement, not arithmetic.
r5o-pin, r5o-lookup-pin: production r5o without and with the lookup overlay.
detm-r5o-pin, detm-r5o-lookup-pin: the deterministic MoE without and with it.
The MoE variant fix (b12x-0006-moe-smallest-variant.patch: a live token count
binds to the smallest planned fused-MoE variant, not the prefill capacity)
adds: detm-r5o-lookup-variant-trace (the trace arm on det-variant, the
det-masked files with that fix; KV cache 768 MiB smaller, after its first boot
also stopped 21 MB short of the guard), r5o-lookup-variant-pin (r5o, the lookup and
the fix on production's _preparation.py, overlay moe-variant) and
detm-r5o-lookup-variant-pin. detm-r5o-lookup-variant-probe is the second
trace arm with the attn-probe overlay (attn-trace plus a recompute probe of the
first layers' query projection, SPARK3_DEBUG_RECOMPUTE=1). The performance arms
carry no debug overlay or logging. detm-r5o-lookup-variant-exact and
detm-r5o-lookup-mhc-variant-exact are the second trace arm with attn-exact
(attn-probe with exact per-row fingerprints in place of sums and the fused
projection's raw q and kv latents), without and with the mHC fix
(vllm-0028-mhc-smallest-capacity.patch, overlay gemv-lookup-mhc).
detm-r5o-lookup-mhc-bi-variant-exact also plans the torch-backend GEMV
capacities on SIMT (vllm-0029-gemv-batch-invariant-backend.patch, overlay
gemv-lookup-mhc-bi, VLLM_DS41_BATCH_INVARIANT=1). The exact arms' KV caches
are 1790 MiB smaller than r5o's (0.45 GiB, just above the 0.43 GiB one
262K-token request needs); boots with 0.7 GiB stopped at the startup guard
during graph capture, and 0.31 GiB is refused.
detm-r5o-lookup-mhc-bi-variant-probe is that arm with attn-probe2 (attn-exact
plus a row-by-row recompute of layers 2 and 14's attention) and
SPARK3_DEBUG_RECOMPUTE=1. vllm-0030-block-fp8-no-split-k-batch-invariant.patch
also plans split-K block-FP8 capacities with one slice in batch-invariant mode
(overlay gemv-lookup-mhc-bi-fp8: 0027-0030): detm-r5o-bi-trace (exact trace
arm with attn-exact2, which also records each row's position and input token), detm-r5o-bi-pin (performance), and r5o-lookup-mhc-variant-pin (r5o with
the fixes that are always on: 0027, 0028 and b12x-0006). detm-r5o-final-trace
adds 0009-moe-deterministic-single-row.patch (overlay det-variant2: the
det-variant files with a deterministic single row kept off the M=1
materialized launch).
"""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-determinism")
PIN = "/cache/kkref/dspark-costs/r5o-pin-20260930"
B12X = "/opt/spark3/candidate/b12x/b12x"
VLLM = "/opt/spark3/candidate/vllm/vllm"
DET = ("moe/fused_moe/_impl.py", "moe/fused_moe/_preparation.py", "moe/fused_moe/_tuning.py",
       "moe/_shared/kernels/dynamic.py", "moe/_shared/kernels/silu.py", "moe/_shared/kernels/w4a16/kernel.py")
LOOKUP = ("models/deepseek_v4_1/b12x_layers.py", "models/deepseek_v4_1/compressor.py")
TRACE = {
    "checksum_debug.py": "model_executor/layers/fused_moe/runner/checksum_debug.py",
    "moe_runner.py": "model_executor/layers/fused_moe/runner/moe_runner.py",
    "model.py": "models/deepseek_v4/nvidia/model.py",
    "model_runner.py": "v1/worker/gpu/model_runner.py",
    "attention.py": "models/deepseek_v4_1/attention.py",
}
base = json.loads(Path("config/cluster.json").read_text())
assert base["promoted_baseline"] == "2026-09-30-karmic-kraken-r5o", base["promoted_baseline"]


def mount(arm, source, target):
    arm["container"]["mounts"].append([f"{{home}}/spark3-overlay/{source}", target, "ro"])


def with_lookup(arm, overlay="gemv-lookup"):
    arm = copy.deepcopy(arm)
    for f in LOOKUP:
        mount(arm, f"{overlay}/vllm/{f}", f"{VLLM}/{f}")
    return arm


r5o_pin = copy.deepcopy(base)
r5o_pin["environment"]["SPARK3_DSPARK_COST_DIR"] = PIN
detm_pin = copy.deepcopy(r5o_pin)
for f in DET:
    mount(detm_pin, f"det-masked/b12x/{f}", f"{B12X}/{f}")
detm_pin["environment"].update(B12X_DYNAMIC_DETERMINISTIC_OUTPUT="1", B12X_DENSE_SPLITK_TURBO="0")
detm_variant_pin = copy.deepcopy(r5o_pin)
for f in DET:
    mount(detm_variant_pin, f"det-variant/b12x/{f}", f"{B12X}/{f}")
detm_variant_pin["environment"].update(detm_pin["environment"])


def traced(arm, shrink_mib, overlay="attn-trace", lookup="gemv-lookup"):
    trace = with_lookup(arm, lookup)
    for name, target in TRACE.items():
        mount(trace, f"{overlay}/{name}", f"{VLLM}/{target}")
    trace["environment"].update(
        SPARK3_MOE_CHECKSUM_DIR="/cache/kkref/moe-checksums", SPARK3_MOE_CHECKSUM_ROWS="64",
        SPARK3_MOE_CHECKSUM_RUNNER_CAPACITY="8192", SPARK3_MOE_CHECKSUM_TAGS_CAPACITY="16384",
        SPARK3_MOE_CHECKSUM_ATTN_CAPACITY="65536", SPARK3_GATE_CAPTURE_ROWS="32",
    )
    args = trace["serve_args"]
    kv = args.index("--kv-cache-memory-bytes") + 1
    args[kv] = str(int(args[kv]) - shrink_mib * 2**20)
    return trace


bi_trace = traced(detm_variant_pin, 1790, overlay="attn-exact", lookup="gemv-lookup-mhc-bi")
bi_trace["environment"]["VLLM_DS41_BATCH_INVARIANT"] = "1"
bi_probe = traced(detm_variant_pin, 1790, overlay="attn-probe2", lookup="gemv-lookup-mhc-bi")
bi_probe["environment"].update(VLLM_DS41_BATCH_INVARIANT="1", SPARK3_DEBUG_RECOMPUTE="1")
bi_full_trace = traced(detm_variant_pin, 1790, overlay="attn-exact2", lookup="gemv-lookup-mhc-bi-fp8")
bi_full_trace["environment"]["VLLM_DS41_BATCH_INVARIANT"] = "1"
bi_full_pin = with_lookup(detm_variant_pin, "gemv-lookup-mhc-bi-fp8")
bi_full_pin["environment"]["VLLM_DS41_BATCH_INVARIANT"] = "1"
detm_variant2_pin = copy.deepcopy(r5o_pin)
for f in DET:
    mount(detm_variant2_pin, f"det-variant2/b12x/{f}", f"{B12X}/{f}")
detm_variant2_pin["environment"].update(detm_pin["environment"])
final_trace = traced(detm_variant2_pin, 1790, overlay="attn-exact2", lookup="gemv-lookup-mhc-bi-fp8")
final_trace["environment"]["VLLM_DS41_BATCH_INVARIANT"] = "1"
probe = traced(detm_variant_pin, 768, overlay="attn-probe")
probe["environment"]["SPARK3_DEBUG_RECOMPUTE"] = "1"
r5o_lookup_variant = with_lookup(r5o_pin)
mount(r5o_lookup_variant, "moe-variant/b12x/moe/fused_moe/_preparation.py",
      f"{B12X}/moe/fused_moe/_preparation.py")
r5o_lookup_mhc_variant = with_lookup(r5o_pin, "gemv-lookup-mhc")
mount(r5o_lookup_mhc_variant, "moe-variant/b12x/moe/fused_moe/_preparation.py",
      f"{B12X}/moe/fused_moe/_preparation.py")
for name, arm in (("detm-r5o-lookup-trace", traced(detm_pin, 256)),
                  ("detm-r5o-lookup-variant-trace", traced(detm_variant_pin, 768)),
                  ("detm-r5o-lookup-variant-probe", probe),
                  ("detm-r5o-lookup-variant-exact", traced(detm_variant_pin, 1790, overlay="attn-exact")),
                  ("detm-r5o-lookup-mhc-variant-exact",
                   traced(detm_variant_pin, 1790, overlay="attn-exact", lookup="gemv-lookup-mhc")),
                  ("detm-r5o-lookup-mhc-bi-variant-exact", bi_trace),
                  ("detm-r5o-lookup-mhc-bi-variant-probe", bi_probe),
                  ("detm-r5o-bi-trace", bi_full_trace), ("detm-r5o-bi-pin", bi_full_pin),
                  ("r5o-lookup-mhc-variant-pin", r5o_lookup_mhc_variant),
                  ("detm-r5o-final-trace", final_trace),
                  ("r5o-pin", r5o_pin), ("r5o-lookup-pin", with_lookup(r5o_pin)),
                  ("r5o-lookup-variant-pin", r5o_lookup_variant), ("detm-r5o-pin", detm_pin),
                  ("detm-r5o-lookup-pin", with_lookup(detm_pin)),
                  ("detm-r5o-lookup-variant-pin", with_lookup(detm_variant_pin))):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(arm, indent=2) + "\n")
    print("wrote", path)
