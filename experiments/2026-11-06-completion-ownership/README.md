# Asynchronous completion ownership

Decision: retain the existing runtime. The inspected paths already distinguish
request identity, reusable row ownership, and buffer lifetime. Adversarial tests
found no separate stale-completion defect that justifies a runtime patch. This is
a no-change correctness investigation, not a measured performance improvement.

## Identity and scope

- Deployment base: `eb8a49f6d92231acaaf3f9fd63a4b7d3a4183854`.
- Promoted release: `2026-10-30-spark-ds41f-1030`; its configuration and immutable
  baseline remain authoritative and unchanged.
- Inspected/tested source: corrected default-copy candidate C40,
  `5f926df6d3126266a9c88c82c06544060008c33c`, tree
  `ca1496c6481dc5bc4b4234b743e4ca023a6c7597`.
- C40 lock and ordered series:
  [candidate-40](../2026-10-26-spark-ds41f-candidate/candidate-40/upstreams.lock.json).
  A fresh replay of all 574 patches from
  `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` reproduced that exact commit and tree.
- CPU test container: immutable C38 image
  `sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`,
  with the eight C38-to-C40 changed files overlaid only inside a disposable
  container. [base.json](base.json) records original and tested byte hashes.
  This is an overlaid CPU test, not an unmodified C40-image qualification.
- [source-comparison.json](source-comparison.json) records identical AST hashes
  for ten named ownership methods in 1030, C36, C38 and C40. That narrow equality
  does not assert equality of the complete runtimes.

The intended variable is whether delayed work can be accepted by a later owner
of a reused row or reply channel. Normal runs leave source behavior unchanged.
Each negative control removes exactly one existing protection; these deliberate
breakages are test sensitivity checks, not newly discovered baseline failures.

## Ownership trace

Paths below are relative to the `vllm/` package in the exact source above;
`gpu/` abbreviates `v1/worker/gpu/` after the first full occurrence.

| Boundary | Existing protection and observed consequence |
| --- | --- |
| External request ID to engine request | `v1/engine/input_processor.py:assign_request_id` randomizes the internal ID. A later request with the same external alias does not inherit an earlier request's scheduler outputs. Disabling randomization explicitly gives up this guarantee and is outside the configured recipe. |
| Queued engine steps and RPC replies | `v1/engine/core.py:step_with_batch_queue` consumes the oldest queued step with its scheduler output. `v1/executor/multiproc_executor.py:FutureWrapper` drains preceding replies even if their futures were canceled. Ordered worker execution and the output thread preserve association; per-rank replies belong to the original RPC. |
| Cancellation and preemption | `v1/core/sched/scheduler.py:update_from_output` skips missing/finished internal IDs. `_preempt_request` accounts for outstanding stale tokens. Drop-mode outputs leave before token, copy-allocation and NaN updates; ordinary lossless preemption may intentionally deliver valid work to the same request. `AsyncScheduler` avoids decrementing reset placeholders or caching preempted blocks. |
| GPU row reuse | `v1/worker/gpu/states.py:RequestState` resets row state on admission. `gpu/model_runner.py` frees, adds, stages writes and launches subsequent work in ordered worker calls and the model stream. Old main-stream writes precede the new owner's reset. Finished IDs are sorted consistently across TP ranks. |
| Pipeline receives | `gpu/pp_utils.py:PPHandler` already has a row generation counter. Receives snapshot it; freeing increments it; consumption masks mismatched rows. All-stale receives produce no usable result. Valid receives retain stream/event ordering. |
| Adaptive confidence copies | `gpu/spec_decode/adaptive_verification.py` resets the readable row and records a pending reset. When the older DMA finishes, its row is reset again before that buffer becomes readable. This prevents an old confidence value from undoing admission's reset. Current/planned confidence buffers follow the same rule. |
| Verification/output copies | `gpu/async_utils.py:AsyncOutput` retains source tensors, snapshots reusable verification counts, waits for its copy event, then materializes output under the original request IDs. A generation check alone would not repair shared bytes overwritten before copying. |
| Copy-draft side-stream preparation | `gpu/spec_decode/copy_drafts.py` joins the outstanding preparation before reuse and consumption. Existing event dependencies order allocation results and subsequent row resets. |
| KV block reuse | Scheduler deferred frees use request last-scheduled and processed-step fences for asynchronous KV connector work, including copy-on-write allocations. Ordinary TP work relies on stream ordering. |
| Queued Engram row reads | `models/deepseek_v4_1/engram_rows/rows.py` and `_reader.c` use step sequences and a consumed-step handshake. Early reads use private staging while the previous gather is outstanding, then wait before copying into shared row storage. Cache prefetch does not modify GPU step buffers. This path was inspected, not newly fault-injected. |

Late request-local RPC errors remain with the old future. Fatal worker, GPU,
reader and collective failures remain engine failures: suppressing them because
a request was canceled could leave ranks inconsistent. Global executed-work
metrics may legitimately include canceled work; they are not new-owner counters.

## Tests and retained attempts

The final workload runs 27 new adversarial cases plus existing async-scheduler,
deferred-KV-free, adaptive-verification, pipeline and multiprocess-response tests.
Transport, DMA and GPU kernels in the new cases are controlled CPU doubles; the
ownership/scheduler methods are the actual source implementation. Existing
response-gather tests also exercise threaded message queues and rank ordering.

New cases cover confidence DMA landing after one or three row reuses, current
and planned confidence arrays, delayed verification snapshots, no/partial/all
stale pipeline rows, canceled or failed RPC futures, and two queued scheduler
steps spanning cancellation with either short or chunked prompts and zero/three
speculative tokens. The replacement uses the same external alias and a distinct
internal ID. Assertions cover tokens, confidence, computed/placeholders/copy
budgets, the unchanged live row, tensor weak-reference release, drained queues,
and return of all 31 usable KV blocks after cleanup. NaN metadata is supplied
on the discarded output; exclusion of its update is verified in the source trace.

| Run | Passed | Failed | Skipped | Meaning |
| --- | ---: | ---: | ---: | --- |
| [Initial](runs/attempt-1/tests.xml) | 95 | 0 | 1 | Initial scheduler/confidence/pipeline cases |
| [Snapshot added](runs/complete/tests.xml) | 96 | 0 | 1 | Adds delayed verification-buffer reuse |
| [Confidence reset removed](runs/negative-confidence/tests.xml) | 0 | 12 | 0 | All delayed-reset cases detect the injected defect |
| [Generation increment removed](runs/negative-generation/tests.xml) | 2 | 4 | 0 | Reused-row cases fail; no-reuse controls pass |
| [Verification snapshot removed](runs/negative-snapshot/tests.xml) | 0 | 1 | 0 | Detects newer counts replacing the older output |
| [Final](runs/final/tests.xml) | 114 | 0 | 1 | Adds RPC cancellation/error and existing transport suite |

The skip requires two GPUs. Final pytest duration was 9.852 seconds; complete
runner duration was 15.053 seconds. Peak process RSS was 1,293,496 KiB (about
1.234 GiB). The container had 2 CPUs, 4 GiB memory, no network and no exposed
GPU; the harness verified CUDA remained uninitialized. Tensor/queue release
checks are bounded unit observations, not a long-running allocator leak test.
All attempted runs and failures are preserved under [runs](runs).

Existing native evidence is separate:
[C40 image gate](../2026-11-04-topk-live-qualification/runs/2026-11-04-dgx/attempt-1/image-gate/receipt.json)
records 8,074 passed and 360 skipped, with saved native copy-allocation,
side-stream and growth/shrinkage/slot-reuse checks. The slot-reuse test uses real
copy kernels with a toy verifier, not a full-model cancellation run. Its
[two-boot serving decision](../2026-11-04-topk-live-qualification/decision.md)
and all adverse metrics remain as recorded. No released or already measured
image was rebenchmarked for this investigation.

## Choice and limits

| Option | Decision |
| --- | --- |
| Existing request IDs, targeted row generations, snapshots and event/sequence dependencies | Retain. The tests exercise the protections already present. |
| Add generations to every TP row consumer | No demonstrated uncovered ownership transition. Adds state/checks without replacing the buffer lifetime obligations. |
| Add a global synchronization before reuse | No demonstrated missing dependency. Would constrain overlap without a demonstrated correctness benefit. |
| Add more per-token snapshots/allocations | Existing verification snapshots protect the mutable buffer found in this trace. No additional buffer needing a snapshot was demonstrated. |

No runtime candidate, new baseline, promotion, performance delta or confidence
interval is produced. Serving source and configuration are unchanged; CPU
durations do not estimate serving throughput. A future change that moves row
mutation to an independent stream or permits unordered per-request worker
execution must revisit these ownership dependencies.

No new GPU/multirank delayed-completion fault injection was performed. A foreign
hold on dgx1 and another request were present when inspected; neither was
overwritten. No cluster sync/start/stop/restart or GPU operation was issued.
The hold's C36 restoration profile was observed operational evidence, not a
replacement for promoted configuration authority. Pipeline generations were
tested on CPU; this does not qualify a new pipeline topology or prove all
possible concurrency schedules safe.

## Reproduction and validation

Prepare C40 from its pinned upstream base and ordered series. The independent
replay used `git am --committer-date-is-author-date` with the repository owner's
committer identity and reproduced the expected head and tree; see
[replay receipt](runs/source-replay.json) and [log](runs/source-replay.log).
Existing patch whitespace warnings were retained; they are not edits from this
investigation.

From this deployment checkout, assemble the input bundle without editing the
prepared source:

```sh
python3 experiments/2026-11-06-completion-ownership/prepare_bundle.py \
  /path/to/prepared/vllm /tmp/completion-ownership/input
mkdir -p /tmp/completion-ownership/output
docker run --rm --runtime=runc --network=none \
  --memory=4g --memory-swap=4g --cpus=2 --pids-limit=256 \
  -e CUDA_VISIBLE_DEVICES=-1 -e VLLM_TARGET_DEVICE=cpu -e VLLM_PLUGINS= \
  -v /tmp/completion-ownership/input:/input:ro \
  -v /tmp/completion-ownership/output:/output --entrypoint timeout \
  sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b \
  240 python3 /input/run_cpu.py
```

Run only on a compatible CPU host with that ARM image already available. The
recorded runs used sp8. Exact executed remote commands are in `runs/*-command.json`.
For a negative control, add `-e COMPLETION_MUTATION=confidence`, `generation`,
or `snapshot` to a fresh disposable container and use a distinct output directory.
Those three runs intentionally exit nonzero. `cpu_declarations.py` exposes the
installed compiler decorators for imports; it does not execute GPU kernels.

`bin/spark doctor` initially reported the isolated checkout's absent ignored
node map. After copying the existing node-local maps, it passed with
`configuration OK`. `bin/spark build check` remains incomplete because the
full build input tree is not prepared locally; its warning and exit are kept.
This audit did not rebuild an unchanged image. Python syntax, bundle source
hashes, fresh patch replay and `git diff --check` passed. No runtime pull request
is warranted by the present evidence.
