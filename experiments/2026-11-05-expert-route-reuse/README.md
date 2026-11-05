# Exact grouped-expert reuse accounting

Status: offline investigation and an unqualified scheduling fix. No GPU execution,
model boot, performance gain, or promotion is claimed. No real route-ID capture
for the requested current code, prose, and distinct agent workloads was found.
The synthetic unit tests are contract checks, not evidence of those workloads.

Source review: [vLLM PR #5](https://github.com/christopherowen/vllm/pull/5),
a draft based on the exact candidate-38 source. The exported experimental
patch is in `vllm/`; no active profile selects it.

Validation completed: 10 accounting tests, both six-case CPU code-generation
runs, static doctor, fresh 572-patch source reproduction, and exported-patch
tree reproduction passed. Existing source lint failures also reproduce on the
untouched base (three E501 lines, formatting, 122 typing errors, and three
TileLang import-policy violations). GPU and model gates remain pending.

## Identities and scope

- Deployment base: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
- Promoted configuration remains `config/cluster.json`; no promoted files change.
- Inspected candidate: `experiments/2026-10-26-spark-ds41f-candidate/candidate-38`.
- vLLM upstream base: the `19f2c20ed4d6` revision in that candidate's lock.
- Prepared candidate head: `1f33102547e4da5c64712390604695d1abd577c4`.
- Prepared candidate tree: `c061e5f6931306fa4655ba531428f639c365bf10`.
- CPU compiler environment: existing image
  `sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
- One source variable: honor the existing positive `swizzle_panel` argument in
  `grouped_fp8_fp4_gemm`. The patch is experimental and is not in any active series.

## What is already implemented

The current dispatcher groups all routed rows in a layer invocation by expert,
including rows from different requests. It preserves each route's row/slot order.
Each expert's rows are padded to an M tile. Candidate 38 chooses M=16 below
`rows * topk = 16 * experts`, M=32 below `32 * experts`, and M=64 thereafter.
For the 384-expert, top-6 target these boundaries are 1024 and 2048 allocated rows;
for the 128-expert, top-3 drafter they are 683 and 1366. Graph padding counts toward
tier selection; masked IDs of -1 do not create expert weight loads.

Within a tile, each expert's weight payload is loaded once across all N blocks.
An expert receiving at most M live routes already needs only one weight copy at
this level. Recomputing its union cannot remove another copy. An expert receiving
more than M routes needs `ceil(count / M)` copies; actual cache reuse between
those loads requires hardware measurements. Empty allocated tiles return before
barrier initialization and do not load their expert weights.

Native FP4 weights use two values per byte. The N/K tiles are 128 by 128; TP4's
576-wide down projection has an exact 64-column tail, not 64 padded FP4 columns.
Scale encoding retains exact UE8M0 row bases and 2- or 4-bit deltas, selected per
expert and projection. The last TP4 down scale-code word contains unused bits;
those bits must not be counted as padded FP4 weights. TP3 uses intermediate 768,
TP4 uses 576; the native intermediate is 2304 and hidden dimension is 5120.

The combine sums slots in fixed FP32 order, adds the shared expert, and rounds at
the existing output boundary. This work does not change IDs, routing weights,
that order, packing, tile sizes, or arithmetic. It does not alter draft budgeting.

## Accounting contract

Run on a remote host with a saved JSONL trace:

```sh
python3 experiments/2026-11-05-expert-route-reuse/analyze_routes.py \
  /node-local/routes.jsonl --output /node-local/routes-accounting.json
python3 -m unittest discover -s experiments/2026-11-05-expert-route-reuse \
  -p 'test_*.py' -v
```

One input object is one actual layer invocation. Never merge calls, steps,
requests from separate launches, or separate ranks into an artificial batch.
Required fields: `workload`, `capture`, `step`, `layer`, `source_tree`,
`model_revision`, `experts`, `topk`, and `rows`. Each row has the consumed `ids`
(in their original slot order), an anonymized `request` identity when live, and
`kind` (`prefill`, `anchor`, `draft`, `padding`, or `dead`). Include all allocated
rows; padding/dead IDs must be the -1 values consumed by dispatch. Raw prompt
text, request content, and credentials are not needed. Optional `hidden` defaults
to 5120. Mark synthetic inputs with `evidence_kind: "synthetic"`.

Optional `scale_bits["3" or "4"]` contains `gate_up` and `down` arrays, with one
2 or 4 per expert. Without observed flags the analyzer reports all-2-bit and
all-4-bit bounds, not a falsely exact scale size. The TP projections use the same
IDs with each shard geometry; they do not establish cross-TP route equivalence.
The `1030` policy is only a tile-height counterfactual with current payload
packing. It is not a simulated or remeasured release.

Outputs include distinct experts, per-expert counts, per-request unions and
pairwise overlap, experts beyond anchors, active/padded rows, used/empty/repeated
tiles, active/empty CTAs, logical FP4/scale payloads, unused code bits, and
alternative 16/32/64 tile counts. Repeated payload is a ceiling on removable
logical loads, not a prediction of saved DRAM bytes or time. Shared experts,
activation/dispatch traffic, narrow-flag loads, and cache-line overfetch are
excluded. An input SHA-256 binds each result to its trace.

For one expert M tile, FP4 payload is `3 * 5120 * intermediate / 2`: 5,898,240
bytes at TP3 and 4,423,680 at TP4. For a projection N by K, scale-code payload is
`N * ceil(K / 128) * bits / 2`, plus N row-base bytes. At TP4, down-projection
2-bit code loads contain 2,560 bytes' worth of unused bits; the FP4 tail is exact.
These are logical payload formulas, not memory-controller counters.

## Swizzle defect and candidate

The candidate plan already passes `prefill_swizzle_panel=3`, but its grouped
GEMM body never calls `T.use_swizzle`. The option therefore does not affect
emitted code. The experimental patch invokes that operation only for positive
panels. Panel zero retains the existing scheduling. Logical payload accounting
is unchanged; only block ordering can change cache locality.

The source regression checks that panel 3 emits `rasterization2DRow<3>` for gather,
TP4 tail, and TP3 aligned K cases, and that panel 0 does not. Bit-equality tests
alone could pass with an ignored option, so generated-code coverage is necessary.
`check_codegen.py` also lowers those cases on CPU without device compilation.
The existing GPU `test_prefill_swizzle_panel_matches_unswizzled_bits` remains
required before treating the fix as qualified. No timing is inferred from
code generation.

## Execution record and remaining gate

See `runs/` and `commands.md` for successful runs and failed setup attempts.
All substantial work after the Mac recovery ran on `sp8`, with a two-CPU,
8-GiB limit and shared CPU locks. The CPU containers had no network and no GPU.
There was no serving operation and no GPU hold was acquired.

Fresh application of all 572 candidate patches reproduced the exact head and
tree above. The initial local preparation had failed with a full disk; its log
is preserved. Remote CPU accounting and generated-source checks are separate
from native GPU compilation, bit equality, batch invariance, and model evaluation.
The task checkout does not have complete build inputs; a complete image build
check is still required when those inputs are prepared.

Before any performance claim, obtain the real route captures described in
`lab-plan.md`, analyze code, prose, and distinct agent requests separately, and
qualify the source patch on idle hardware. Candidate measurements require a boot
and repeat boot with confidence intervals, compared with immutable recorded
references. Do not rerun a released baseline. If the remaining loads or measured
cost do not improve, reject the scheduling change. No promotion is proposed.
