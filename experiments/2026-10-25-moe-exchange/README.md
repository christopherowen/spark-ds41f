# SP prefill MoE exchange: the combine writes the slots, the exchange runs under it

For the release after spark-ds41f-1025. The kernel audit
([draft #585](https://github.com/christopherowen/spark-ds41f/pull/585), its finding 3)
found that every sequence-parallel layer copies its combined MoE output into the
exchange buffers. The fusion survey (draft, `experiments/2026-10-25-fusion-survey`,
rows 17 and 18) ranks running the exchange under the combine at the same boundary. This
experiment does both in one change. Owner rules apply: bit-identical, no slower, no
switch, no fallback.

**Nothing here has run on a GPU yet.** The release window holds the cluster, and there
is no local torch. The tests were checked by reading; they and the benchmark run in the
lab bundle below. Times are CET.

## Base and change

- **vLLM:** the local branch `moe-exchange-direct` (worktree
  `~/projects/vllm-moe-exchange`), commit f18ccfda0 over `tilelang-ports` 26382f5cf, the
  package files of candidate 6's image (`spark-ds41f:19f2c20ed4d6-next-6`). Candidate 6's
  two test patches (0489, 0490) touch other files. The branch is not pushed: pushing it
  would publish the frozen `tilelang-ports` history, as with `kernel-audit` and
  `moe-overlap`.
- **Files:** `fused_moe/runner/moe_runner.py`, `fused_moe/tilelang.py`,
  `deepseek_v4_1/sp_prefill.py` and `deepseek_v4_1/tilelang/moe.py`, plus three test
  files and `benchmarks/kernels/benchmark_deepseek_v41_moe_exchange.py`.

## What the traces show

Request inventory 5, TP4 ring, rank 0, the 15,395-token prompt (one forward of 2,140 ms;
[exchange_span.py](exchange_span.py) on `request-inventory-5/ports`). Each of the 20
sequence-parallel MoE layers spends 13.94 ms (median) from its first combine launch to
the next sublayer entry, which reads the exchanged parts:

| Part (a layer, median) | ms | Over 20 layers |
|---|---:|---:|
| `moe_combine` (one launch, the shared expert folded in) | 5.42 | 109.7 ms |
| four `Memcpy DtoD` of 39.4 MB, combine output into `send` / `parts` | 1.31 | 27.6 ms (1.29%) |
| the NCCL exchange, with nothing beside it | 7.01 | 165.3 ms (7.7%) |
| gaps | 0.03 | |
| **stage** | **13.94** | **306.2 ms (14.3%)** |

The copies come from `MoERunner._reduce_scatter_final_output`'s `write`
(`out.copy_(fused_output[start:stop])`). The model's hook
(`sp_prefill.project_reduce_scatter_current`) calls it once a destination block.

## The change

1. **The combine is deferred to the exchange.** When the model's SP hook is installed and
   the shared expert folds into the combine (the production path,
   `MOE_FP32_COMBINE`), the runner sets `defer_combine` on the TileLang experts around
   the MoE op. `TileLangExperts.apply` runs the plan without an output:
   `MoeExpertsV41Plan.run(out=None)` launches the dispatch, the cast, both grouped GEMMs
   and the SwiGLU, and returns `combine(start, stop, out)`. The hook's writer *is* that
   combine. `MoeDispatchV41Plan.combine` takes `start` / `stop`: the kernel is one CTA
   a token, and its row-indexed inputs are sliced views, so the kernel does not change.
2. **Each block goes straight into its slot.** `_project_exchange` writes, block by
   block, the other ranks' blocks into `send` and this rank's into `parts[rank]`. The pad
   rows are still TileLang's fill (`columns.zero_`). The copy writer is gone: the hook
   refuses a call without the deferred combine.
3. **The exchange runs under the combine.** In each slice the other ranks' blocks are
   written first, then that slice's exchange starts on the side stream, then this rank's
   own block is written under it. The MoE output is exchanged in
   `moe_slices(T) = max(1, T // 4096)` slices: 3 at the 15K prompt and at the TP4
   recipe's 16,288-token long-prefill chunk, 4 at a full 16,384-row step, 2 at the
   determinism check's 10.5K prompt. A slice's message to a peer is at least 1,024 rows
   at TP4 (10 MB), so an exchange's fixed cost stays small beside it.
4. **WO's two slices** (`wo_slices`, unchanged) go through the same loop and take the
   same order: each slice's exchange now starts before this rank's own WO block is
   projected rather than after it.

**Why the bits do not change.** Every row is the same combine of the same routes; a
token's sum reads only its own routes, so splitting the launch changes no row. Each row
lands in the slot the copy put it in, the exchange only moves bytes, and the consumer
(mHC `post_pre`'s rank-order sum) reads the same parts.

**What stays.** The non-hook reduce-scatter path (no SP hook, NCCL's summing
reduce-scatter) still writes the whole output and copies it into its padded buffer:
DS4.1's TileLang family on RoCE always installs the hook. Decode and every non-SP
forward are untouched.

## Estimate

At 3 slices a layer's stage becomes exchange-bound. The first slice's other blocks are
combined (0.75 × 5.42 / 3 = 1.36 ms), then the exchange runs (7.01 ms), and the rest of
the combine (4.06 ms) runs beside it. That is about 8.4 ms a layer against 13.94, plus a
fixed cost for each of the extra exchanges:

- **Estimate:** about 5 ms a layer, **5% of the 15K prefill** (about 100-110 ms of
  2,140).
- **Floor:** the copies alone, **1.29%**.
- **Ceiling:** the fusion survey's row 17 (the exchange hidden under the combine and the
  copies), **6.32%**.

The estimate assumes the exchange runs at its measured rate while the combine streams
beside it. The combine reads about 1 GB a layer from DRAM; the exchange's own DRAM
traffic is a few tens of GB/s of GB10's 273. If they contend, the combine slows first, and it has about
3 ms of slack under the exchange. WO's reordering is worth at most a quarter of one
slice's WO projection a layer, and is not counted.

## Tests (vLLM branch)

- `tests/kernels/moe/test_deepseek_v41_tilelang_moe.py::test_moe_experts_deferred_combine_has_the_combine_bits`:
  the deferred combine, piece by piece into separate buffers, equals the one combine
  bit for bit. Covers the BF16, FP32 and folded kernels at 1, 12 and 300 rows. A callable
  shared addend is called once, by `run`.
- `tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`:
  - `test_moe_combine_writes_the_exchange_slots` uses the real TileLang plan through the
    hook. Each other rank's sent block and this rank's part carry the bits of the one
    combine the copies moved. The pad rows come from TileLang's fill (torch's `zero_` is
    refused). Covers 1 to 12 slices, ranks 0 and 3. Each slice is exchanged after the
    other ranks' blocks of it and before this rank's own.
  - `test_unreduced_parts_hold_every_block` now also checks that order, and that
    `then()` runs after the last projection. Its stand-in exchange no longer copies this
    rank's own block, which is now written after the exchange starts.
  - `test_moe_slices` covers the slice counts.
- `tests/models/test_deepseek_v4_1_sp_prefill.py`:
  - the hook receives the deferred combine as its writer and refuses a call without one;
  - the runner's forward defers only a folded combine bound for the hook, hands it over
    and resets the experts' state;
  - the experts defer only a folded combine and leave `output` unwritten.

## Benchmark

`benchmarks/kernels/benchmark_deepseek_v41_moe_exchange.py --gate`. It times the real
`_project_exchange` with the deferred combine against 26382f5cf's path (one combine,
four copies, pad fill), with the exchange stubbed: the exchange needs every rank.

- Shapes, with DS4.1 Flash routing (384 experts, 6 a token, hidden 5120):
  - TP4: 16,384 tokens, 15,395 at ranks 0 and 3, and 8,192;
  - TP3: 15,395 at ranks 0 and 2.
- Method: eight fresh inputs a shape, bits checked first, CUDA-event device time.
- Gate: no shape slower (the 95% interval's lower bound above zero).
- One slice (blocks without slicing) is reported beside it, for the cost of the extra
  combine launches.

## Lab plan (after the release window; the specs sync node checkouts to origin/main, so this PR merges first)

`python3 experiments/2026-10-25-moe-exchange/make_bundles.py ~/projects/vllm-moe-exchange`
writes `bundles/exchange`, `bundles/serving/overlay`, `exchange-tp4.json`,
`prof-candidate6.json` and `prof-exchange.json` from the branch (BASE 26382f5cf, HEAD
`moe-exchange-direct`). Rerun it after any change to the branch. The dry runs of all
three specs plan cleanly.

1. **[exchange-kernels.json](exchange-kernels.json)**, run `moe-exchange-kernels`: the
   bundle on dgx1, on candidate 6's image with the package files mounted. It runs the
   MoE kernel tests, the SP prefill tests, then the gated benchmark. Pass: every test
   passes and the gate holds.
2. **[exchange-serving.json](exchange-serving.json)**, run `moe-exchange-serving`: TP4
   ring, arms exchange / candidate 6 / exchange (candidate first).
   - Temperature-0 determinism with and without log probabilities (the 10.5K long
     prompt, 2 slices), and a 16.5K long prompt (`--long-lines 1100`: a 16,288-token
     chunk, 3 slices). The long prompts run sequence-parallel.
   - The cold prefill suite [prefill-suite.json](prefill-suite.json): 15K, 32K and 131K,
     five repeats.
   - Pass: each arm identical within itself, every exchange arm's digests equal
     candidate 6's ([tables_arms.py](tables_arms.py), written by the run), no undeclared
     kernel, and prefill rates no lower than candidate 6's
     (`python3 scripts/servebench_suite_table.py moe-exchange-serving exchange candidate6 exchange-2`).
3. **[exchange-profile.json](exchange-profile.json)**, run `moe-exchange-profile`: the
   15,395-token prompt under the torch profiler, candidate 6 then the exchange, without
   Python stacks.
   - `python3 experiments/2026-10-25-moe-exchange/exchange_span.py moe-exchange-profile candidate6 exchange`
     gives each arm's step, the SP MoE stages' span, the combine launches, copies (80 to
     0) and NCCL time; `--node dgx2` and so on for the other ranks.
   - The lab's cost step compares every kernel against candidate 6.
     `profile_prefill.py`, `summarize_kernels.py` and `analyze_costs.py` are request
     inventory 5's, copied here because the profile job runs them from the experiment.

Run them in that order; stop at the first failure.

## Coordination

`git merge-tree` against the other sessions' local vLLM branches:

- **`moe-overlap`:** one conflict, in `TileLangExperts.apply`'s single `run` call. That
  branch passes `_provided_mxfp8(hidden_states)` there; the merged call takes it and
  `None if self.defer_combine else output`. After the merge, this branch's new tests
  call `plan.run(x, ...)` with BF16 tokens and need `cast_mxfp8(x)`, as that branch's
  own tests do.
- **`pdl`, `retire-conditionals`, `kernel-audit`:** merge cleanly. `pdl` adds
  `pdl.chain()` to the combine kernel, whose body this change leaves alone.

Whoever lands second reruns `exchange-kernels.json`.

## Runs

### 2026-10-25 19:09 CET: the kernel bundle on dgx4: pass

The kernel bundle ran on dgx4, which is uncabled and opened for single-node testing while
dgx1-3 serve the TP3 triangle. The cluster specs (serving, profile) stay queued.

- Command, on dgx4 in its own checkout:
  `scripts/lab.py kernel-local experiments/2026-10-25-moe-exchange/bundles/exchange --out ~/spark-lab/agents/moe-exchange/out/exchange`.
  The checkout is `~/spark-lab/agents/moe-exchange/spark-ds41f`, branch `moe-exchange` at
  7afddf79.
- Bundle: vLLM f18ccfda0, eight files over 26382f5cf, on image `spark-ds41f:19f2c20ed4d6-next-6`
  (`sha256:82b071b57485`, candidate 6).
- Output: `~/spark-lab/agents/moe-exchange/out/exchange/{output.txt,verdict.json}` on dgx4.
- Verdict: **pass**, exit 0, 72 s.
  - MoE kernel tests: 66 passed.
  - SP prefill tests: 175 passed, none skipped (the real-combine slot test ran on the GPU).
  - Benchmark gate: holds at every shape.

The benchmark measures the copies removed; the exchange is stubbed, so the overlap is not
measured here.

| Shape | Slices | 26382f5cf: combine + copies (µs) | The combine into the slots (µs) | Difference (95% interval) | One slice (µs) |
|---|---:|---:|---:|---:|---:|
| TP4 16,384, rank 0 | 4 | 6,999.1 | 5,680.5 | −1,332.0 (−1,380.5, −1,283.5), −19.0% | 5,589.0 |
| TP4 15,395, rank 0 | 3 | 6,569.8 | 5,282.2 | −1,274.4 (−1,312.4, −1,236.3), −19.4% | 5,262.4 |
| TP4 15,395, rank 3 | 3 | 6,564.5 | 5,309.9 | −1,267.5 (−1,292.1, −1,242.9), −19.3% | 5,247.7 |
| TP4 8,192, rank 0 | 2 | 3,519.8 | 2,829.7 | −693.8 (−717.6, −670.0), −19.7% | 2,822.5 |
| TP3 15,395, rank 0 | 3 | 6,562.5 | 5,274.4 | −1,290.4 (−1,311.2, −1,269.6), −19.7% | 5,262.3 |
| TP3 15,395, rank 2 | 3 | 6,557.4 | 5,266.6 | −1,287.9 (−1,301.9, −1,274.0), −19.6% | 5,248.7 |

- The 15,395-token baseline (6.57 ms) matches the trace's combine plus copies
  (5.42 + 1.31 ms).
- Removing the copies saves 1.27 ms a layer: 25.5 ms over the 20 SP layers, 1.19% of the
  2,140 ms prefill on one rank, before any overlap.
- Splitting the combine into slices costs 20 µs at three slices and 91 µs at four, against
  one slice; the exchange each slice starts early is what that buys, on the cluster.
- Next: exchange-serving.json and exchange-profile.json need the full cluster.
