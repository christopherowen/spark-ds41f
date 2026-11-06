# Exact grouped-expert reuse accounting

Status: offline investigation and an unqualified scheduling fix. No GPU execution,
model boot, preserved hardware performance, performance gain, or promotion is claimed. No real route-ID capture
for the requested current code, prose, and distinct agent workloads was found.
The synthetic unit tests are contract checks, not evidence of those workloads.

Source review: [vLLM PR #5](https://github.com/christopherowen/vllm/pull/5),
a draft based on the exact candidate-38 source. The exported experimental
patch is in `vllm/`; no active profile selects it.

Validation completed: 14 accounting/ordering tests, the initial six-case CPU
code-generation runs, a 72-case native-shape CPU audit, 48 workspace checks,
949,524 comparisons with the installed compiler rasterizer, static doctor, and
fresh 572-patch source reproduction passed. Four final default-plan CUDA sources
are byte-identical to the untouched candidate-38 default. Existing source lint failures also reproduce on the
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
- Source candidate: `f3bfdf8b8c08870754cc3c756b75faa9156595e0`, tree
  `8d20314f29ed7cc26b8c4e15f85046b36faaa6df`.
- One source variable: honor explicitly positive `swizzle_panel` in
  `grouped_fp8_fp4_gemm`; set the plan default to zero to preserve the existing
  default emitted code. The two-patch stack is not in any active series.

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

The candidate-38 plan default is `prefill_swizzle_panel=3`, but its grouped
GEMM never calls `T.use_swizzle`: panels zero and three emit identical code.
The initial patch activates positive panels. The audit then makes zero the
explicit default, preserving the previous default schedule until hardware
supports a choice. A positive caller option now changes block ordering only.
Logical payload, route IDs/weights, packing and arithmetic are unchanged.

Regression coverage goes through the plan factory for both TP shard widths and
both projections. The final default and explicit zero sources match the old
default byte-for-byte. Positive panels add just one logical-block remapping line
in each of 72 CPU-lowered cases: M16/32/64, panels 0/1/2/3/4/8, native hidden
5120, intermediate 576/768, 384 experts and top-6. These are source-generation
checks, not native compilation or execution. GPU bit-equality cases additionally
cover both native shard widths with eight experts to bound temporary memory.

## Alternative and residual bounds

`analyze_ordering.py` considers M16/32/64, both shard widths, six panel sizes,
all-2/all-4 scale bounds and five explicitly synthetic route populations.
`check_audit.py` compares its ordering with the installed compiler's actual C++
rasterizer for 949,524 coordinates, including partial and oversized panels.
The compiler header SHA-256 is
`37b8b17dda33204c494586eeabebc6d8d08cec808c67bac46e8d847fd1f9517a`.

| Alternative | Bound or observed offline result | Limitation |
| --- | --- | --- |
| More expert-union bookkeeping | Zero removable weight copies when each expert already fits one M tile | Actual workload occupancy is missing |
| M16 to M32/M64 | Can reduce repeated logical weight loads; also changes padding, scratch and occupancy | Existing M64 screening is recorded in the 2026-10-26 TP3 memory experiment; do not remeasure that release |
| Panel 2 versus 3 | For synthetic balanced 8192-row TP4 down, two tiles/expert, panel 2 keeps repeated weight keys one grid index apart; panel 3 has p95 199 | Linearized grid order is not GPU execution order, physical traffic or elapsed time |
| 4-bit to 2-bit scale deltas everywhere | At most 1.51% (TP3) / 1.57% (TP4) of all-4-bit logical weight-plus-scale payload | Existing exact packing already selects narrow experts; the remaining real fraction is unknown |
| Remove every scale load | At most 3.14% / 3.27% of the same logical payload | A hypothetical ceiling, not a viable exact implementation or speedup |
| Remove unused TP4 scale-tail bits | At most 0.112% of that payload | FP4 tail is already exact; transactions may be unchanged |

In the disjoint 48-row fixture, M16/32/64 all load the same expert weights while
larger tiles increase padding and intermediate scratch. In the 17-row hot-expert
fixture, M32 halves repeated logical payload against M16 while M64 gives no
further payload reduction and doubles intermediate scratch. There is no
universal best panel or tile height established by these fixtures. Fixed
geometry, hardware scheduling, activation locality and true route distribution
remain relevant. No global optimality claim is supported.

The first audit accidentally passed topk=7 while exercising the compiler; its
successful 72-case output is retained as `audit-cpu-matrix-topk7.*` and is not
native-contract evidence. `audit-cpu-matrix.*` reruns the full matrix at topk=6.
The initial compiler-file inspection TypeError is also retained.

## Execution record and remaining gate

See `runs/` and `commands.md` for successful runs and failed setup attempts.
All substantial work after the Mac recovery ran on `sp8`, with a two-CPU,
8-GiB limit and shared CPU locks. The CPU containers had no network and no GPU.
At this audit checkpoint there was no serving operation and no GPU hold was
acquired. The owner has subsequently authorized hardware tests and the necessary
coordinated serving operations; foreign holds and queued requests still apply.

Fresh application of all 572 candidate patches reproduced the exact head and
tree above. The initial local preparation had failed with a full disk; its log
is preserved. Remote CPU accounting and generated-source checks are separate
from native GPU compilation, bit equality, batch invariance, and model evaluation.
The task checkout does not have complete build inputs; a complete image build
check is still required when those inputs are prepared.

Before any faster-expert-reuse claim, obtain the real route captures described in
`lab-plan.md`, analyze code, prose, and distinct agent requests separately, and
qualify the source patch on idle hardware. Candidate measurements require a boot
and repeat boot with confidence intervals, compared with immutable recorded
references. Do not rerun a released baseline. If the remaining loads or measured
cost do not improve, reject the scheduling change. No promotion is proposed.


## Correctness acceptance and separate optional tuning

Before hardware execution, the owner clarified that a correctness improvement
with no performance loss is acceptable. This does not turn the ordering audit
into a speed claim. The required behavior repair is that an explicitly positive
swizzle panel actually takes effect, while the default remains unchanged.

The `contract-before` bundle mode runs the final four generated-code assertions
against the untouched candidate-38 file. It accepts only four AssertionErrors
containing the swizzle marker, with no import, compiler, device or skipped-test
failure. The `gate` mode runs those assertions and the 20 native output-bit
cases against the fix. Expected old failures and all new outputs remain raw
receipts. The final default comparison loads both actual source modules in one
process, shares prepared weights/inputs/output/workspace, verifies native
compiled-source equality, and compares full output bits before timing.

Run `default` twice in fresh processes on a qualified node. There are 30
synthetic geometry/routing cases across both shard widths; each has eight input
blocks, six timing replays and three whole-MoE calls per graph. Across the 48
replays each of two arms appears first 24 times; the six-panel screen likewise
places each arm at each position eight times. A CPU test protects this balance.
The initial rotate-plus-reverse formula would have cancelled ordering changes
for two arms; this was corrected before any hardware run.

Predeclared component preservation screen: in each run, every case's paired
95% upper bound on added latency must be below 1%, with unchanged workspace
size and exact output bits. Report repeat variation and per-case intervals; a
wide interval containing a meaningful regression is inconclusive, not neutral.
This is a conservative experiment threshold, not a replacement for the
repository's serving comparison/noise bounds. Two process runs do not claim two
serving boots. Default native-code identity supports a proportionate component
qualification; optional panel speed, model boot/throughput and actual-route
claims remain separate and unproved until their own evidence exists.

Preparation now passes 31 CPU tests, including an actual pytest-report format
probe and strict rejection of unrelated failures. Native gates retain JUnit
receipts and require all 24 selected tests without skips. The dual-module CPU
rehearsal covers all 16 grouped-GEMM tiers/widths and passes.
The unchanged SwiGLU helper asks the real GPU for vector width, so the first
CPU-only attempt stopped with a missing-driver error. Its failed log is kept;
CPU checks skip that unchanged helper, while native default mode compares it.

The capture prototype now has exact consumer hooks and ten CPU row-join tests.
It uses each layer's actual attention metadata key and explicit CED tag, the
GPU query starts, and request/position coordinates. Native integration and a
published serving capture profile remain pending. `capture/README.md` records
its separate source identity and limits; it is not part of source PR #5.
