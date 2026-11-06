# Prefix group validity

This experiment checks that a reused prefix has every state group needed to
resume at its reported boundary. Global KV and index keys need a complete
prefix; encoder sliding-window state needs the preceding window. Decoder,
drafter and compressor state is private and must be reconstructed.

The unreleased control is candidate 38, vLLM head
`1f33102547e4da5c64712390604695d1abd577c4`, tree
`c061e5f6931306fa4655ba531428f639c365bf10`. The deployment base is
`eb8a49f6d92231acaaf3f9fd63a4b7d3a4183854`. The promoted release remains
spark-ds41f-1030 and its recorded measurements are not rerun.

## Candidate

The candidate variable is the recovery boundary after preemption has discarded
private decoder state. The control treats accepted output tokens as a new
prompt and replays only its final 128 rows. Recovery should instead reconstruct
the decoder history from the original prompt's replay boundary, including the
accepted output suffix. This hypothesis requires a native-kernel reproducer
and full-model validation before acceptance.

The separate encoder-endpoint retention change in deployment PR #923 is outside
this experiment. No retention policy is changed here.

## Gates

- Independently evict or shorten global, compressed/indexer and encoder-window
  groups; compare reported hits with an independent availability oracle.
- Cover 128-token replay and 256-token global-page boundaries, forks, request
  cancellation/freeing, preemption, shared pages and group/batch reordering.
- Show an old-fails/new-passes native three-layer decoder reproducer, then
  compare real generated tokens and scores against uninterrupted execution.
- Compare ordinary uncached and reused serving latency/throughput, memory and
  failures on boot and repeat boot; report recovery work separately.
- Preserve all unsuccessful runs and exact identities. Hardware requires our
  own exclusive window, all-rank identity, guards, cooling and restoration.

## Status

Draft investigation. Native SM121 GPU kernel and metadata verification completed on
`dgx4`; full-model TP=3 serving recovery remains a multi-node hardware gate.
Source proposal: https://github.com/christopherowen/vllm/pull/6 (draft).
Deployment/evidence proposal: https://github.com/christopherowen/spark-ds41f/pull/931 (draft).
The source branch is `feat/prefix-group-validity`; `base.json`, `source.json` and
`overlay.json` record its exact identities. The recovery policy is explicitly limited to
fixed input requests. Actual streaming-input sessions retain their existing
cache-hit and worker replay behavior; their recovery correctness remains
unresolved. Stateless multi-turn prompts and output-only streaming remain
eligible for the candidate.

The initial independent group suite passed 168 cases; the locked control failed the six
preemption-boundary cases and passed the other 162. A broader CPU cache suite
passed 395 cases with four HiSparse admission cases deselected. Attempts to check
those four separately first failed at a missing `tblib` dependency and then were
interrupted; they are not validated here. Isolated policy functions passed 45
cases. The separately labeled NumPy dependency model passed 27 cases. Native
kernel cases remain unrun. The current scoped head passed 183 group/identity
cases on DGX4. Seventeen harness tests and six subtests passed, followed by three
all-rank source-identity rejection tests. The remote GPU-less native-module
attempts could not import the CUDA-specific code. Both arms' four HiSparse
cases failed identically at device inference; these are unavailable-platform
results, not functional passes or evidence of a candidate regression.

The serving probe now also cancels a producer by its exact request ID, requires
the server's abort counter to increase and its queue to become idle, and then
compares a same-prefix follower with uninterrupted execution. Cached tokens are
required to qualify that reuse case. Its 24 harness tests passed remotely at
deployment commit `ac6e1f2f9500d30964b915e5fa3e080b45f43f44`; the receipt is
`runs/cancellation-harness.json`. Subsequent restoration-control hardening added
a sticky management/telemetry-loss latch (including guard-loss propagation from
`guarded_probe.py`), pre-mutation activity-marker checks before `stop`/`sync`/`start`,
and exact window/holder/token revalidation under the hold-mutation lock immediately
before `hold_path.unlink()`. All 28 harness tests and four `doctor` checks passed
remotely at deployment commit `be762869f7e4c7087581c9637b222cc1a70ef958`
(`runs/restoration-controls-harness.json`). This verifies probe and watchdog
controls, not serving behavior. The native runbook separates the unchanged-API
old-source reproducer from candidate metadata tests and records the required
test/compiled-package mount layout. Actual DSpark context storage and
full-checkpoint recovery remain hardware gates.

The first fresh 573-patch preparation applied the source but found a committer
identity mismatch. The source commit was normalized to the repository's fixed
build committer, yielding the reproduced head `3fe3dfe937c67ee26e3a73d514f4ade73f62ec9e`.
After adding the finite-cone tests, the 574-patch retry failed during fetch with
`No space left on device`. Its receipt is preserved. Final-head preparation and
unfinished tests moved remotely; neither failure is a successful
build check. The remote 574-patch fresh application subsequently reproduced head
`2b7790c8fdbf9031560950ce3e7280d0cd2b609d` and tree
`24f540f78abb4f88019427d72d65c7ad94a3aba8`. All three experiment profiles passed
`doctor`. Full build inputs are not prepared; no complete image-build check is
claimed or required for this Python-only screen. The subsequent scope patch
continued that same fresh series to 575 patches, reproducing current head
`c229ab4d2c4d18b3dcd157ff8a3ec7255108164e` and tree
`11d1cf3e6ac525173683dbfcd3b08de1c945934d`.

On owner instruction, the disposable local build tree, Python environment and
control checkout were removed. `runs/mac-cleanup.json` records the exact paths
and 1,217,118,208 allocated bytes removed. Source changes and all available
evidence were preserved. Further builds, dependencies and tests run remotely. After independently
recovering the published source on DGX4, all 7,812 tracked local paths matched
the published snapshot and no extra files, registered worktrees or known
alternates dependents remained. The retained local source was then removed too;
`runs/mac-source-recovery.json` records another 155,832,320 allocated bytes,
bringing the total removed allocation to 1,372,950,528 bytes.

`screen-tp3.json` follows the repository's Python-only screening procedure:
four exact source files from the published commit are mounted read-only over
the unchanged candidate-38 image. `source_overlay.py` verifies the image digest,
every changed file hash and the mounts on every rank before a serving probe.
`tp3.json` separately describes a future complete candidate image, which has not
been built. The two source identities must never be conflated.

Local setup initially failed with missing `aiohttp`, then `openai_harmony` in
the borrowed read-only Python dependency set. A separate environment was created
under `.work/cpu`; neither serving images nor the neighboring task's environment
were modified. The candidate-38 existing CED cache tests then passed: 6 passed,
200 deselected, in 9.06 s. These are allocator tests, not serving evidence.

## Streaming-input scope

The scalar numerical model in `streaming_replay.py` passes six phase checks with
append lengths 1, 2, 127, 128, 129 and 256. A one-token append can retain decoder
inputs produced under an older floor; a longer append can deliberately truncate
that history. Neither a permanently first anchor nor a mutable latest anchor
reproduces all these cases. This is dependency evidence, not a native kernel or
checkpoint result. Exact input-session recovery would need phase history and
historical model-affecting settings, including steering and prompt-logprob mode.
The candidate transmits the durable input-session identity to the worker and
leaves that path unchanged, including after the final input sentinel clears
`resumable`. No new immutable first-prompt field was added.

The remote source push initially failed because DGX4 had no GitHub publishing
credentials. The exact same commit was published through the existing
authenticated GitHub API connection after matching every blob, the tree and
commit hash. No credentials were copied to the node; `runs/source-publication.json`
records the resulting identity.

## Native SM121 GPU Verification and 2-Boot Benchmark (`dgx4`)

Under an exclusive node-scoped lease on `dgx4` (`pool: dgx4-single`,
`window: dgx4-prefix-group-validity-20261106`), the native SM121 TileLang 3-layer
CED + SparseMLA reproducer, GPU metadata suites, HiSparse admission checks, and
a 2-boot CED replay latency/accuracy benchmark (`bench_ced_replay.py`) ran inside
the locked candidate-38 container (`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`)
on `NVIDIA GB10` (`compute capability 12.1`) at deployment commit
`2c3c406de461458184fcf29e655968a455eb3674` and source commit
`c229ab4d2c4d18b3dcd157ff8a3ec7255108164e` (tree
`11d1cf3e6ac525173683dbfcd3b08de1c945934d`). Complete receipts are recorded in
`runs/step-b6-summary.json` and `runs/gpu-*.{json,txt}`.

### GPU Test Suites

| Arm | Step | Suite / Filter | Result | Wall Time | Receipt |
| --- | --- | --- | --- | --- | --- |
| Control | `gpu-control-ced-reproducer` | `test_deepseek_v41_ced_replay.py -k test_preempted_decode_rebuilds_original_decoder_history` | **4 failed** (expected old-fails finite mismatch) | 11.814 s | `runs/gpu-control-ced-reproducer.json` |
| Control | `gpu-control-hisparse` | `test_prefix_caching.py -k hisparse_async_admission_requires_only_import_destinations` | **4 passed** | 12.295 s | `runs/gpu-control-hisparse.json` |
| Candidate Boot 1 (`candidate-a`) | `gpu-candidate-a-ced-reproducer` | `test_deepseek_v41_ced_replay.py -k test_preempted_decode_rebuilds_original_decoder_history` | **4 passed** | 11.753 s | `runs/gpu-candidate-a-ced-reproducer.json` |
| Candidate Boot 1 (`candidate-a`) | `gpu-candidate-a-ced-replay-full` | `test_deepseek_v41_ced_replay.py` | **32 passed** | 16.680 s | `runs/gpu-candidate-a-ced-replay-full.json` |
| Candidate Boot 1 (`candidate-a`) | `gpu-candidate-a-ced-metadata` | `test_b12x_v41_ced_metadata.py` | **82 passed** | 11.338 s | `runs/gpu-candidate-a-ced-metadata.json` |
| Candidate Boot 1 (`candidate-a`) | `gpu-candidate-a-glp` | `test_deepseek_v4_1_glp.py` | **48 passed, 1 skipped** (`GLP_TEST_FILE` unset) | 7.910 s | `runs/gpu-candidate-a-glp.json` |
| Candidate Boot 1 (`candidate-a`) | `gpu-candidate-a-prefix-group-validity` | `test_prefix_group_validity.py` | **183 passed** | 7.000 s | `runs/gpu-candidate-a-prefix-group-validity.json` |
| Candidate Boot 1 (`candidate-a`) | `gpu-candidate-a-hisparse` | `test_prefix_caching.py -k hisparse_async_admission_requires_only_import_destinations` | **4 passed** | 11.974 s | `runs/gpu-candidate-a-hisparse.json` |
| Candidate Boot 2 (`candidate-b`) | `gpu-candidate-b-ced-reproducer` | `test_deepseek_v41_ced_replay.py -k test_preempted_decode_rebuilds_original_decoder_history` | **4 passed** | 23.562 s | `runs/gpu-candidate-b-ced-reproducer.json` |
| Candidate Boot 2 (`candidate-b`) | `gpu-candidate-b-ced-replay-full` | `test_deepseek_v41_ced_replay.py` | **32 passed** | 16.485 s | `runs/gpu-candidate-b-ced-replay-full.json` |
| Candidate Boot 2 (`candidate-b`) | `gpu-candidate-b-ced-metadata` | `test_b12x_v41_ced_metadata.py` | **82 passed** | 28.061 s | `runs/gpu-candidate-b-ced-metadata.json` |
| Candidate Boot 2 (`candidate-b`) | `gpu-candidate-b-glp` | `test_deepseek_v4_1_glp.py` | **48 passed, 1 skipped** (`GLP_TEST_FILE` unset) | 7.912 s | `runs/gpu-candidate-b-glp.json` |
| Candidate Boot 2 (`candidate-b`) | `gpu-candidate-b-prefix-group-validity` | `test_prefix_group_validity.py` | **183 passed** | 7.066 s | `runs/gpu-candidate-b-prefix-group-validity.json` |
| Candidate Boot 2 (`candidate-b`) | `gpu-candidate-b-hisparse` | `test_prefix_caching.py -k hisparse_async_admission_requires_only_import_destinations` | **4 passed** | 11.844 s | `runs/gpu-candidate-b-hisparse.json` |

### 2-Boot Native 3-Layer CED + SparseMLA Benchmark (`bench_ced_replay.py`)

Each benchmark point runs 10 warmup iterations and 25 CUDA-event timed iterations
on `NVIDIA GB10`.

| Workload | Shape | Control Median (ms) | Control `exact_match` (`max_abs_diff`) | Candidate Boot 1 Median (ms) | Candidate Boot 2 Median (ms) | Candidate `exact_match` (`max_abs_diff`) |
| --- | --- | --- | --- | --- | --- | --- |
| Ordinary prompt/prefix chunking (`prompt_end=1152`) | `chunk=512, prefix=0` | 3.3498 | `true` (`0.0`) | 3.3414 | 3.3409 | `true` (`0.0`) |
| Ordinary prompt/prefix chunking (`prompt_end=1152`) | `chunk=384, prefix=0` | 3.3921 | `true` (`0.0`) | 3.3847 | 3.3898 | `true` (`0.0`) |
| Ordinary prompt/prefix chunking (`prompt_end=1152`) | `chunk=512, prefix=448` | 3.3357 | `true` (`0.0`) | 3.3307 | 3.3363 | `true` (`0.0`) |
| Preempted decode recovery (`original_end=1024, hit=768`) | `gen=129, chunk=64` | 7.5623 | **`false` (`0.0029297`)** | 7.5266 | 7.5482 | **`true` (`0.0`)** |
| Preempted decode recovery (`original_end=1024, hit=768`) | `gen=129, chunk=512` | 2.3666 | **`false` (`0.0029297`)** | 2.3856 | 2.3971 | **`true` (`0.0`)** |
| Preempted decode recovery (`original_end=1024, hit=768`) | `gen=257, chunk=64` | 9.7148 | **`false` (`0.0024414`)** | 9.6954 | 9.7068 | **`true` (`0.0`)** |
| Preempted decode recovery (`original_end=1024, hit=768`) | `gen=257, chunk=512` | 2.3835 | **`false` (`0.0024414`)** | 2.8455 | 2.8451 | **`true` (`0.0`)** |

Ordinary prompt/prefix chunking is bit-identical and latency-neutral (`-0.07%` to
`-0.26%` median delta). Under preempted decode recovery across the 256-token
global-page and 128-token CED replay boundaries, the control produces finite
corrupted decoder output (`max_abs_diff = 0.0029297` at `generated_rows=129` and
`0.0024414` at `generated_rows=257`), whereas both candidate boots reconstruct
bit-exact uninterrupted decoder output (`max_abs_diff = 0.0`) across all tested
chunk sizes and generated suffixes.
