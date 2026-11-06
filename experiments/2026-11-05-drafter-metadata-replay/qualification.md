# Remaining qualification plan

This plan is not an instruction to run while another holder owns the cluster.
Before any GPU job or node operation, read dgx1 `~/spark-hold.json`; yield to its
holder. Preserve an existing `~/spark-request.json`. At preparation time both an
active frontend qualification hold and another task's request existed.

1. Completed on two fresh Spaceport boots (`runs/micro-20261106/` and
   `runs/gates-20261106/`). Using the clean `candidate_source` checkout
   (`a2b0ea500dc5022e23c4d6fc667254a806c984dc`) overlaid onto the immutable
   candidate-38 image (`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`)
   under explicit 2-CPU/8-GiB limits and a scoped `sp8` hold:
   - `trace_and_gates.py` measured cold-cache startup compilation across all five
     exercised `metadata_builder` specializations (`94,391.504 ms` on boot 1,
     `93,430.649 ms` on boot 2; warm lookups `47.088 us` and `47.136 us`; zero
     incremental specializations added by the candidate; zero late compilation
     misses after `prepare.freeze()`).
   - Before/after `DFlashCudaGraphManager` capture across `(5,1), (10,2), (20,4), (40,8), (80,16)`
     recorded `0` incremental bytes in the CUDA graph pool and `0` bytes difference
     in allocated tensor memory (`5,120` bytes in both arms).
   - `cudaGraphDebugDotPrint` dumps (`graph-before-req8.dot` vs `graph-after-req8.dot`)
     and `torch.profiler` replay traces confirmed `metadata_builder_kernel<<<2,128,0>>>`
     is captured inside the FULL query graph, `eager_build_calls` dropped from
     `27` to `0`, host `cuLaunchKernelEx` launches disappeared during replay, and
     all `192/192` exact bitwise comparisons passed on each boot.
   - Focused `draft_metadata_graph` (3 passed) and broader GPU suites (388 passed,
     0 failed per boot across CED metadata, DFlash inputs, DFlash causality,
     DFlash DCP, CUDA graph manager, DSpark context, and DSpark/TileLang Markov
     sampling) passed on both fresh boots. All three earlier runner/collection
     attempts are retained in `runs/gates-20261106/attempt-1..3/`.

2. Full pinned build inputs were prepared on Spaceport and the candidate's
   `bin/spark build check` passed (`runs/build-20261106/`). The image build and
   its gate remain pending. Build once and record
   its digest, source labels and whole release-gate result. Never overwrite an
   image tag or infer source identity from it.

3. The owner has authorized the normal coordinated operations needed for testing.
   This does not bypass holds, queued requests or publication/restore rules.
   Before a serving run, produce the site-resolved TP3 plan
   for all configured nodes and the exact restore profile observed by the window
   owner. The source-only `candidate.json` deliberately has launch disabled.
   Publish one clean deployment commit before node synchronization; use only
   `bin/spark cluster sync/start/stop` with `--cluster-config` and inspect `live:`
   and `shape:` before any authorized `--apply`. Do not use `--change-shape`.

4. Within the authorized window, run candidate boot and repeat boot back to back;
   restore production once at the end. Require equal source/image identity on
   every rank and identical metadata/draft layout for fixed request inputs,
   including padding, cancellations and slot reuse. Exercise greedy and seeded
   probabilistic target verification, variable rejection counts, mixed prefill,
   streaming and disconnect/cancellation. Compare target tokens and sampled
   probability/logprob contracts with retained compatible fixtures. Check every
   rank's logs for stalled collectives, startup JIT, fallbacks and non-finite data.
   Untouched sampler/transport code is not a substitute for these checks.

5. Use the recorded reference's exact workload identity. A lean exploratory
   command, only where compatible saved reference runs exist, is:

   ```sh
   bin/spark --cluster-config experiments/2026-11-05-drafter-metadata-replay/candidate.json bench --suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink --concurrency 1,8 --min-samples 3 --max-samples 3
   ```

   Keep all samples and both boots, memory/swap/thermals, startup, step time,
   acceptance, aggregate/per-user decode, TTFT, TPOT and ITL where recorded.
   Include cold/mixed prefill and first-chunk waits in the subsequent full
   matrix. Compare against saved 1030, C36 and the previous compatible candidate
   with signed deltas and separate repeat margins. Do not rerun released builds,
   mix lean/full protocols, infer missing repeats or treat the old host-gap bound
   as a contemporary baseline. No comparable saved reference means inconclusive.

The owner also accepts correctness improvements with preserved performance.
For that route, first demonstrate a concrete affected-path failure in the old
implementation and a pass in the repair, including relevant boundaries and
failure/cancellation behavior. Preserve native arithmetic, batch invariance and
supported features. Keep optional tuning separate from an independently useful
repair.

Prove performance preservation using the material affected latency, throughput
and memory metrics, exact identities, repeated candidate runs and compatible
saved references. State the existing comparison/noise bound before interpreting
results and require the uncertainty bound to exclude a meaningful regression;
a confidence interval merely spanning zero does not establish no loss. Preserve
unfavorable samples. A materially underpowered comparison is inconclusive.

The current metadata change has no demonstrated pre-existing correctness defect:
its tests show equivalence, and its speedup is scoped to the component fixture.
It cannot qualify through the correctness route without such evidence. Its full
serving behavior is still unmeasured. Owner acceptance, configuration/docs and a
new immutable baseline are separate promotion work and are not authorized here.
