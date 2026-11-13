# PR #1005 rework, round 1: no defensible correction; dropped from this release

**Sub-agent verdict:** `no-defensible-correction`. No patch.

**Mark:** the FFN sublayer by graph replay, on the serving code path. Every arm is bit-exact against candidate-47 without #1005. All cells are in `result.json`.

**The hook half** (shared_experts launched after the gate) does nothing on the GPU.
- On the serving path the shared expert's `workspace_size` is 0, so 1107 already launches it on the aux stream before the router gate, and the post-gate hook is never reached.
- #1005 reaches the hook, but records the input-ready event before the gate. The captured graph keeps the same dependencies; only the host-side capture order changes.
- ABBA: 2 of 56 cells faster, none slower. Paired run: 0 of 44 cells move.

**The kernel half** (`moe_swiglu_cast`) saves about 1–1.5 µs at M≥24.
- That is against a sublayer of 2,400–5,200 µs a layer, while the two baseline boots differ by 10–20 µs. At M≤12 it saves nothing measurable.
- ABBA: 1 cell faster, 1 slower. Paired run: 1 faster, 2 slower, both at TP3, where 0765 also changes `rows_per_cta` from 2 to 4.

**The one alternative that moves the sublayer** records the event after the gate GEMM.
- Warm L2 at TP3 M=1..4: 1–3 µs faster.
- Cold L2 plus the FFN prefetch: 2–5 µs slower in 13 of 44 cells, which breaks "slower in no cell".

**Whole PR against candidate-47 without #1005** (session 1): 2 cells faster, 4 slower, 50 within the interval, out of 56.

## Coordinator's verdict: dropped from this release

Under the owner's rule, a second rework is for a PR whose first rework shows promise, and this one shows none:
- the hook change is a no-op on the device;
- the kernel saving is below the noise of a single boot;
- the only schedule change that moves the sublayer is slower in cold cells.

#1005 is dropped from this release; its record stays here. The triage paired it with #960, which uses the same hook. #960 must be judged on its own measurement, not on this one.

The analysis is the rework sub-agent's; the coordinator saved it and wrote the verdict.
