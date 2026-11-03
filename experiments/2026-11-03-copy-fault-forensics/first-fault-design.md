# Proposed first-fault recorder

This design has not been implemented or validated on a GPU. Locations refer to
candidate-37 vLLM commit `5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575`.
It is a diagnostic experiment, not a proposed inference fix.

## Producer and consumer

The gate projection is launched at
`vllm/models/deepseek_v4_1/tilelang/router.py:1243`. A shared-expert launch hook
runs at 1244–1245, before the route consumer at 1282. The hook originates in
`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:773–790,814`.
This defines an interval to inspect; it is not evidence of an overwrite.
Shared-expert workspace is intended to be disjoint.

Instrument the actual projection producer in `tilelang/gemm.py:1023–1050`:

- Check the A/B tiles actually read by the computation for nonfinite values.
- Mirror its existing `C_local` result when it writes partials, before another
  operation can change them. Use dedicated preallocated storage outside the
  reusable workspace, including the input values needed for a failing-row
  replay. An additional copy scheduled after the hook cannot recover values
  that were already overwritten.

Instrument the existing consumer in `tilelang/router.py:962–974`:

- Compare the consumed partials bit-for-bit with their producer mirror.
- Observe the existing ordered accumulation and retain the first addition
  that changes a finite accumulator into a nonfinite one. A replacement
  `torch.sum` would not preserve that reduction order.

| Observed boundary | Interpretation |
| --- | --- |
| Nonfinite A/B read by producer | A nonfinite value existed before this projection. |
| Finite A/B, nonfinite produced partial | The first observed nonfinite value appears within the projection; preserve operands for a reference calculation. |
| Consumed partial differs from produced partial | The value changed between production and consumption. |
| Matching finite partials, nonfinite ordered sum | The recorded reduction step overflows. |

Locating the first nonfinite result is not by itself a complete root cause:
finite but incorrect or excessively large inputs may originate earlier.

## Attribution and recovery

Freeze the first record per rank with immutable model role/layer identity,
invocation sequence, row domain, capture-time request/slot mapping, masks,
dimensions and original offending values. Bind role/layer at router creation.
Do not label accumulated bits with the phase or batch active when the host
eventually reads them. Cross-rank records still require matching step identity;
independent per-rank clocks do not establish global execution order.

Device assertions can make subsequent CUDA copies fail, preventing retrieval
of an otherwise intact device snapshot. Before implementing this recorder,
validate a recovery path: CUDA postmortem capture, or a diagnostic-only trap
replacement that freezes the evidence, skips the fault row, asynchronously
collects the snapshot and aborts the run. Any output after that diagnostic
intervention is invalid inference evidence. Zeroing gate logits and continuing
normally is unsuitable.

Extra stores and checks alter timing and register use. Failure disappearance
under this instrumentation would therefore be inconclusive.

## Required controls

Inject separate bad-input, bad-produced-partial, post-producer overwrite and
finite-partial-overflow cases. Check exact attribution and preserved operands;
verify masked rows keep their normal early exit. For finite clean inputs,
compare routing output bit-for-bit with candidate-37 and inspect captured
programmatic dependency edges. None of these recorder controls has been run.

Nsight Systems and CUDA-GDB are present on sp8. A stream timeline can help
inspect buffer ownership; it is not itself a detector of every global-memory
race. Full serving instrumentation remains subject to the coordinated cluster
window and explicit serving-operation authorization.

## Later evidence and recovery control

The subsequent layout records and real transport/reduction fixtures establish
an upstream unequal-layout defect. They do not establish the first failing
GPU forward: the host layout record precedes graph/eager launch. Any future
recorder must distinguish host preparation, successful enqueue, GPU entry and
GPU completion, and retain a common invocation identity across ranks. The
first target embedding collective is an earlier observation boundary than
the MoE gate and warrants input/output and payload-extent capture.

CUDA-GDB has now preserved a deliberate control's original operands, NaN bits
and consumer location, then recovered those values from a fresh GPU-core
reload. See `sp8-tool-cuda-gdb-controls-20261103T082442Z.json`. This validates
the small fixture's manual debugger capture, not automatic full-model capture
or recovery of all pinned-host receive buffers. The original serving crash
still has no captured first nonfinite tensor. None of the producer/consumer
recorder controls above has run.

For a future automatic dump arm, CUDA 13.0 documents
`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`, `CUDA_COREDUMP_FILE` with `%h`, `%p`, `%t`,
and `CUDA_COREDUMP_GENERATION_FLAGS`. Do not set `skip_global_memory` when the
purpose is operand recovery. Confirm free space for the model's allocations
on every rank before launch, and preserve the JIT binary/source identities.
Core-dump enablement itself can affect timing. These are preparation notes,
not a reviewed or authorized serving launch plan.

Reference: [CUDA-GDB 13.0 GPU core dump support](https://docs.nvidia.com/cuda/archive/13.0.1/cuda-gdb/index.html#gpu-core-dump-support).
