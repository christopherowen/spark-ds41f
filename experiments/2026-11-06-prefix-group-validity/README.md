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

Draft investigation. No serving change, GPU measurement, or performance claim.
Source proposal: https://github.com/christopherowen/vllm/pull/6 (draft).
The source branch is `feat/prefix-group-validity`; `base.json`, `source.json` and
`overlay.json` record its exact identities. The initial recovery policy covers a
fixed original prompt; streaming input can advance that prompt boundary and is
an additional correctness gate still under investigation.

The independent group suite passed 168 cases; the locked control failed the six
preemption-boundary cases and passed the other 162. A broader CPU cache suite
passed 395 cases with four HiSparse admission cases deselected. Attempts to check
those four separately first failed at a missing `tblib` dependency and then were
interrupted; they are not validated here. Isolated policy functions passed 45
cases. The separately labeled NumPy dependency model passed 27 cases. Native
kernel cases remain unrun. Seventeen harness tests and six subtests passed.

The first fresh 573-patch preparation applied the source but found a committer
identity mismatch. The source commit was normalized to the repository's fixed
build committer, yielding the reproduced head `3fe3dfe937c67ee26e3a73d514f4ade73f62ec9e`.
After adding the finite-cone tests, the 574-patch retry failed during fetch with
`No space left on device`. Its receipt is preserved. Final-head preparation and
preparation and unfinished tests moved remotely; neither failure is a successful
build check. The remote 574-patch fresh application subsequently reproduced head
`2b7790c8fdbf9031560950ce3e7280d0cd2b609d` and tree
`24f540f78abb4f88019427d72d65c7ad94a3aba8`. All three experiment profiles passed
`doctor`. Full build inputs are not prepared; no complete image-build check is
claimed or required for this Python-only screen.

On owner instruction, the disposable local build tree, Python environment and
control checkout were removed. `runs/mac-cleanup.json` records the exact paths
and 1,217,118,208 allocated bytes removed. Source changes and all available
evidence were preserved. Further builds, dependencies and tests run remotely.

`screen-tp3.json` follows the repository's Python-only screening procedure:
three exact source files from the published commit are mounted read-only over
the unchanged candidate-38 image. `source_overlay.py` verifies the image digest,
every changed file hash and the mounts on every rank before a serving probe.
`tp3.json` separately describes a future complete candidate image, which has not
been built. The two source identities must never be conflated.

Local setup initially failed with missing `aiohttp`, then `openai_harmony` in
the borrowed read-only Python dependency set. A separate environment was created
under `.work/cpu`; neither serving images nor the neighboring task's environment
were modified. The candidate-38 existing CED cache tests then passed: 6 passed,
200 deselected, in 9.06 s. These are allocator tests, not serving evidence.
