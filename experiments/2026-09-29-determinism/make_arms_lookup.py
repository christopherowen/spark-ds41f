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
materialized launch). detm-r5o-final-wide-trace is that arm with attn-exact3
(debug-attn-exact3.diff on attn-exact2: calls of 65-1088 rows, a long prefill,
go to separate wide logs that decode steps never overwrite; the schedule log
holds 1088 rows per step and 1024 steps). vllm-0031-reduce-scatter-rank-order-batch-invariant.patch
(overlay gemv-lookup-mhc-bi-fp8-rs: 0027-0031) adds each row's partials of the
TP reduce-scatter in rank order in batch-invariant mode; prefill sequence
parallelism reduce-scatters twice per layer: detm-r5o-rs-trace (the wide trace
arm with it) and detm-r5o-bi-rs-pin (performance). Its second version adds the
chunks as the RoCE one-shot all-reduce does (float32, rank order, one
rounding; the first rounded after each BF16 add), so a row matches between
reduce-scattered and all-reduced steps: overlay gemv-lookup-mhc-bi-fp8-rs2,
detm-r5o-rs2-trace and detm-r5o-bi-rs2-pin. detm-r5o-rs2-exact4-trace is that
trace arm with attn-exact4 (debug-attn-exact4.diff on attn-exact3: an explicit
row offset in every record, sequence-parallel WO output and index weights at
their rows, raw step labels, a one-time plan inventory) and wide logs of up to
4160 rows, a full 4096-token chunk with its decode rows.

The reference configuration (overlay ref1: vllm-0027 to 0037 on r5o's tree)
gives every operation one arithmetic for every row count in batch-invariant
mode, and aligns prefill chunks to the long-prefill threshold, which must
leave room for the largest decode batch (4000 < 4096 - 48): detm-r5o-ref-pin
(performance, the deterministic MoE of det-variant2) and detm-r5o-ref-trace
(attn-exact5: attn-exact4 with vllm-0035 in its attention copy);
detm-r5o-ref-trace6 also records the LM head's input rows and logits
(attn-exact6). Cost attribution: detm-r5o-ref-no{moe,attn,head,mhc,gemv}-pin
each leave one reference change out (vllm-0034, 0035, 0037, 0033, 0032).

The frozen reference (overlay ref2: ref1 plus vllm-0038, the target LM head's
vocabulary projection on one F.linear kernel for every logit-row count):
detm-r5o-ref2-pin (performance) and detm-r5o-ref2-trace6 (attn-exact6).
r5o-pin-prof and detm-r5o-ref2-pin-prof add the torch profiler (kernel timings
of captured workloads: profile_decode.py, profile_c8.py, profile_prefill.py).

Cost recovery, cumulative over ref2 (each overlay a copy of ref2 with the
named files from the vLLM clone's commits): ref3a adds vllm-0039 (mHC on the
lagged native route), ref3b also vllm-0040 (TMA prefill GEMV kernel), ref3c
also vllm-0041 (fused rank-order sum): detm-r5o-ref3{a,b,c}-pin and
detm-r5o-ref3c-trace6.

ref4a is ref3c with vllm-0042 in place of 0035 (decode rows keep the decode
attention kernel; the step kind from request state; sparse_mla.py and ced.py
join the mounted files): detm-r5o-ref4a-pin and detm-r5o-ref4a-trace7
(attn-exact7: attn-exact6's debug attention with 0042 instead of 0035).

ref4b adds vllm-0043 and B12X patch 0007 (overlay gemv-geom: the small-row
TMA prefill GEMV geometry for decode-sized plans): detm-r5o-ref4b-pin and
detm-r5o-ref4b-trace7.

ref4c is ref4b with vllm-0042 corrected (run62: CED layers ran a mixed step's
decode rows single-pass but a decode-only step's on the decode kernel; they now
split decode rows like every other layer, and CED decoder metadata takes the
full-row step kind): detm-r5o-ref4c-pin and detm-r5o-ref4c-trace8 (attn-exact8:
attn-exact7 with the same correction). The ref4c arms mount gemv-geom2, built
from the r5o image: gemv-geom (ref4b) came from a pre-r5o tree and lacks r5o's
proxy fence in the TMA prefill GEMV.

detm-r5o-ref4c-b4144-pin aligns chunks to 4096 tokens with a 4144-token budget
(4096 plus the largest decode batch, 8 x 6), so a prompt splits where r5o splits
it when alone (the 4000-token alignment adds a short chunk to most prompts);
detm-r5o-ref4c-b4144-trace8 traces it (wide logs of 4160 rows hold 4144).

ref4d is ref4c with vllm-0044 (16 rows per CTA for the mHC capacity plan) and
B12X patch 0008 (overlay mhc-mt2, built on the r5o image: the multi-token
lagged partial kernel): detm-r5o-ref4d-pin, detm-r5o-ref4d-b4144-pin and their
traces detm-r5o-ref4d-trace8, detm-r5o-ref4d-b4144-trace8.

ref4e (experiment) is ref4d whose mHC override takes the TF32 TMA projection at
every capacity with VLLM_DS41_MHC_TF32_SPLITS K slices for the operations B12X
serves with it (post_pre, pre on the expanded residual; run65 try 1 failed on the
others, which keep the native route) (row-invariant in run58b; prefill near
production's, more per decode call at one stream, less at eight): detm-r5o-ref4e-s{16,40}{,-b4144}-pin and -trace8.

Cost recovery candidates stay separate from the frozen ref2: detm-r5o-ref2-mhccap
is ref2 with the mHC input capture (overlay mhc-capture); detm-r5o-ref3m-pin is
ref2 with vllm-0039 in place of 0033 (overlay ref3m: every mHC pre/post_pre
capacity on the lagged native route).
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
RS = "distributed/device_communicators/cuda_communicator.py"


def with_rs(arm, overlay="gemv-lookup-mhc-bi-fp8-rs"):
    arm = copy.deepcopy(arm)
    mount(arm, f"{overlay}/vllm/{RS}", f"{VLLM}/{RS}")
    return arm


final_wide = traced(detm_variant2_pin, 1790, overlay="attn-exact3", lookup="gemv-lookup-mhc-bi-fp8")
final_wide["environment"].update(
    VLLM_DS41_BATCH_INVARIANT="1", SPARK3_MOE_CHECKSUM_WIDE_ROWS="1088",
    SPARK3_MOE_CHECKSUM_WIDE_CAPACITY="4096", SPARK3_MOE_CHECKSUM_SCHEDULE_CAPACITY="1024",
)
probe = traced(detm_variant_pin, 768, overlay="attn-probe")
probe["environment"]["SPARK3_DEBUG_RECOMPUTE"] = "1"
r5o_lookup_variant = with_lookup(r5o_pin)
mount(r5o_lookup_variant, "moe-variant/b12x/moe/fused_moe/_preparation.py",
      f"{B12X}/moe/fused_moe/_preparation.py")
r5o_lookup_mhc_variant = with_lookup(r5o_pin, "gemv-lookup-mhc")
mount(r5o_lookup_mhc_variant, "moe-variant/b12x/moe/fused_moe/_preparation.py",
      f"{B12X}/moe/fused_moe/_preparation.py")
final_wide4 = with_rs(traced(detm_variant2_pin, 1790, overlay="attn-exact4", lookup="gemv-lookup-mhc-bi-fp8"),
                      "gemv-lookup-mhc-bi-fp8-rs2")
final_wide4["environment"].update(
    VLLM_DS41_BATCH_INVARIANT="1", SPARK3_MOE_CHECKSUM_WIDE_ROWS="4160",
    SPARK3_MOE_CHECKSUM_WIDE_CAPACITY="3072", SPARK3_MOE_CHECKSUM_SCHEDULE_CAPACITY="512",
)
# attn-exact4 also hooks DS4.1's model forward (CED decoder rows) and captures index selections.
mount(final_wide4, "attn-exact4/model41.py", f"{VLLM}/models/deepseek_v4_1/nvidia/model.py")
final_wide4["environment"]["SPARK3_DEBUG_INDEX_CAPTURE"] = "2,8,14:1530:1545"
REF_FILES = ("models/deepseek_v4_1/b12x_layers.py", "models/deepseek_v4_1/compressor.py",
             "distributed/device_communicators/cuda_communicator.py",
             "model_executor/layers/fused_moe/b12x.py", "model_executor/kernels/linear/b12x_blockscaled.py",
             "v1/core/sched/scheduler.py")


REF2_FILES = REF_FILES + ("model_executor/layers/logits_processor.py",)
REF4_FILES = REF2_FILES + ("models/deepseek_v4_1/sparse_mla.py", "models/deepseek_v4_1/ced.py")


def with_reference(arm, *, attention, ref="ref1"):
    arm = copy.deepcopy(arm)
    files = REF_FILES if ref == "ref1" else REF4_FILES if ref.startswith("ref4") else REF2_FILES
    for f in files + (("models/deepseek_v4_1/attention.py",) if attention else ()):
        target = f"{VLLM}/{f}"
        arm["container"]["mounts"] = [m for m in arm["container"]["mounts"] if m[1] != target]
        mount(arm, f"{ref}/vllm/{f}", target)
    arm["environment"]["VLLM_DS41_BATCH_INVARIANT"] = "1"
    args = arm["serve_args"]
    args[args.index("--long-prefill-token-threshold") + 1] = "4000"
    return arm


ref_pin = with_reference(detm_variant2_pin, attention=True)


def reference_without(drop=None, layers=None):
    arm = with_reference(detm_variant2_pin, attention=drop != "models/deepseek_v4_1/attention.py")
    if drop:
        arm["container"]["mounts"] = [m for m in arm["container"]["mounts"] if m[1] != f"{VLLM}/{drop}"]
    if layers:
        target = f"{VLLM}/models/deepseek_v4_1/b12x_layers.py"
        arm["container"]["mounts"] = [m for m in arm["container"]["mounts"] if m[1] != target]
        mount(arm, f"ref1-var/{layers}/vllm/models/deepseek_v4_1/b12x_layers.py", target)
    return arm


attribution = {
    "nomoe": reference_without(drop="model_executor/layers/fused_moe/b12x.py"),
    "noattn": reference_without(drop="models/deepseek_v4_1/attention.py"),
    "nohead": reference_without(drop="model_executor/kernels/linear/b12x_blockscaled.py"),
    "nomhc": reference_without(layers="nomhc"),
    "nogemv": reference_without(layers="nogemv"),
}
def profiled(arm, name):
    arm = copy.deepcopy(arm)
    arm["serve_args"] += ["--profiler-config", json.dumps({
        "profiler": "torch", "torch_profiler_dir": f"/cache/kkref/profiles/det-{name}",
        "torch_profiler_with_stack": False, "ignore_frontend": True, "torch_profiler_use_gzip": True,
    })]
    return arm


def with_overlay_file(arm, overlay, f):
    """arm with one vLLM file replaced by overlay/vllm/f."""
    arm = copy.deepcopy(arm)
    target = f"{VLLM}/{f}"
    arm["container"]["mounts"] = [m for m in arm["container"]["mounts"] if m[1] != target]
    mount(arm, f"{overlay}/vllm/{f}", target)
    return arm


ref2_pin = with_reference(detm_variant2_pin, attention=True, ref="ref2")
mhc_capture = copy.deepcopy(ref2_pin)
mhc_capture["container"]["mounts"] = [m for m in mhc_capture["container"]["mounts"]
                                      if m[1] != f"{VLLM}/models/deepseek_v4_1/b12x_layers.py"]
mount(mhc_capture, "mhc-capture/b12x_layers.py", f"{VLLM}/models/deepseek_v4_1/b12x_layers.py")
mhc_capture["environment"]["SPARK3_DEBUG_MHC_CAPTURE"] = "/cache/kkref/mhc-capture:512:2048"


GEMV_GEOMETRY_FILES = ("gemm/bf16_gemv/_prefill.py", "gemm/bf16_gemv/_tuning.py", "gemm/bf16_gemv/_preparation.py")


def with_env(arm, **values):
    arm = copy.deepcopy(arm)
    arm["environment"].update(values)
    return arm


def with_budget(arm, threshold=4096, budget=4144):
    """arm with chunks aligned to threshold and a batch budget leaving room for every decode row."""
    arm = copy.deepcopy(arm)
    args = arm["serve_args"]
    args[args.index("--long-prefill-token-threshold") + 1] = str(threshold)
    args[args.index("--max-num-batched-tokens") + 1] = str(budget)
    return arm


MHC_MULTI_TOKEN_FILES = tuple(f"norm/mhc/{f}" for f in ("_kernels.py", "_preparation.py", "_tuning.py"))


def with_mhc_multi_token(arm, overlay="mhc-mt2"):
    """arm with B12X's multi-token mHC lagged partial kernels (B12X patch 0008)."""
    arm = copy.deepcopy(arm)
    for f in MHC_MULTI_TOKEN_FILES:
        mount(arm, f"{overlay}/b12x/{f}", f"{B12X}/{f}")
    return arm


def with_gemv_geometry(arm, overlay="gemv-geom"):
    """arm with B12X's TMA prefill GEMV launch geometries (B12X patch 0007; gemv-geom2 on r5o's tree)."""
    arm = copy.deepcopy(arm)
    for f in GEMV_GEOMETRY_FILES:
        mount(arm, f"{overlay}/b12x/{f}", f"{B12X}/{f}")
    return arm


def reference_trace(overlay, ref="ref1"):
    arm = with_reference(traced(detm_variant2_pin, 1790, overlay=overlay, lookup=ref), attention=False, ref=ref)
    mount(arm, f"{overlay}/model41.py", f"{VLLM}/models/deepseek_v4_1/nvidia/model.py")
    arm["environment"].update(
        SPARK3_MOE_CHECKSUM_WIDE_ROWS="4160", SPARK3_MOE_CHECKSUM_WIDE_CAPACITY="3072",
        SPARK3_MOE_CHECKSUM_SCHEDULE_CAPACITY="512", SPARK3_DEBUG_INDEX_CAPTURE="2,8,14:1530:1545",
    )
    return arm


ref_trace = reference_trace("attn-exact5")
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
                  ("detm-r5o-final-wide-trace", final_wide),
                  ("detm-r5o-rs-trace", with_rs(final_wide)), ("detm-r5o-bi-rs-pin", with_rs(bi_full_pin)),
                  ("detm-r5o-rs2-trace", with_rs(final_wide, "gemv-lookup-mhc-bi-fp8-rs2")),
                  ("detm-r5o-bi-rs2-pin", with_rs(bi_full_pin, "gemv-lookup-mhc-bi-fp8-rs2")),
                  ("detm-r5o-rs2-exact4-trace", final_wide4),
                  ("detm-r5o-ref-pin", ref_pin), ("detm-r5o-ref-trace", ref_trace),
                  ("detm-r5o-ref-trace6", reference_trace("attn-exact6")),
                  ("detm-r5o-ref2-pin", with_reference(detm_variant2_pin, attention=True, ref="ref2")),
                  ("detm-r5o-ref2-trace6", reference_trace("attn-exact6", ref="ref2")),
                  ("r5o-pin-prof", profiled(r5o_pin, "r5o-pin-prof")),
                  ("detm-r5o-ref2-mhccap", mhc_capture),
                  *((f"detm-r5o-{r}-pin", with_reference(detm_variant2_pin, attention=True, ref=r))
                    for r in ("ref3a", "ref3b", "ref3c")),
                  ("detm-r5o-ref3c-trace6", reference_trace("attn-exact6", ref="ref3c")),
                  ("detm-r5o-ref4a-pin", with_reference(detm_variant2_pin, attention=True, ref="ref4a")),
                  ("detm-r5o-ref4a-trace7", reference_trace("attn-exact7", ref="ref4a")),
                  ("detm-r5o-ref4b-pin", with_gemv_geometry(
                      with_reference(detm_variant2_pin, attention=True, ref="ref4b"))),
                  ("detm-r5o-ref4b-trace7", with_gemv_geometry(reference_trace("attn-exact7", ref="ref4b"))),
                  ("detm-r5o-ref4c-pin", with_gemv_geometry(
                      with_reference(detm_variant2_pin, attention=True, ref="ref4c"), "gemv-geom2")),
                  ("detm-r5o-ref4c-trace8", with_gemv_geometry(reference_trace("attn-exact8", ref="ref4c"),
                                                               "gemv-geom2")),
                  ("detm-r5o-ref4c-b4144-pin", with_budget(with_gemv_geometry(
                      with_reference(detm_variant2_pin, attention=True, ref="ref4c"), "gemv-geom2"))),
                  ("detm-r5o-ref4c-b4144-trace8", with_budget(with_gemv_geometry(
                      reference_trace("attn-exact8", ref="ref4c"), "gemv-geom2"))),
                  *((f"detm-r5o-ref4d{b}-{kind}", budget(with_mhc_multi_token(with_gemv_geometry(arm, "gemv-geom2"))))
                    for b, budget in (("", lambda a: a), ("-b4144", with_budget))
                    for kind, arm in (("pin", with_reference(detm_variant2_pin, attention=True, ref="ref4d")),
                                      ("trace8", reference_trace("attn-exact8", ref="ref4d")))),
                  *((f"detm-r5o-ref4e-s{n}{b}-pin", with_env(budget(with_mhc_multi_token(with_gemv_geometry(
                      with_reference(detm_variant2_pin, attention=True, ref="ref4e"), "gemv-geom2"))),
                      VLLM_DS41_MHC_TF32_SPLITS=str(n)))
                    for n in (16, 40) for b, budget in (("", lambda a: a), ("-b4144", with_budget))),
                  *((f"detm-r5o-ref4e-s{n}{b}-trace8", with_env(budget(with_mhc_multi_token(with_gemv_geometry(
                      reference_trace("attn-exact8", ref="ref4e"), "gemv-geom2"))),
                      VLLM_DS41_MHC_TF32_SPLITS=str(n)))
                    for n in (16, 40) for b, budget in (("", lambda a: a), ("-b4144", with_budget))),
                  ("detm-r5o-ref3m-pin", with_overlay_file(ref2_pin, "ref3m", "models/deepseek_v4_1/b12x_layers.py")),
                  ("detm-r5o-ref2-pin-prof", profiled(with_reference(detm_variant2_pin, attention=True, ref="ref2"),
                                                      "detm-r5o-ref2-pin-prof")),
                  *((f"detm-r5o-ref-{k}-pin", v) for k, v in attribution.items()),
                  ("r5o-pin", r5o_pin), ("r5o-lookup-pin", with_lookup(r5o_pin)),
                  ("r5o-lookup-variant-pin", r5o_lookup_variant), ("detm-r5o-pin", detm_pin),
                  ("detm-r5o-lookup-pin", with_lookup(detm_pin)),
                  ("detm-r5o-lookup-variant-pin", with_lookup(detm_variant_pin))):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(arm, indent=2) + "\n")
    print("wrote", path)
