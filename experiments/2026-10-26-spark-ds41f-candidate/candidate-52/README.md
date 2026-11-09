# candidate-52

Unpromoted release candidate: candidate-50 without #1009 plus 3 PRs, the routed-MoE rounding fix, the Engram
reader and the `max_tokens` verify-row cap. It is tested at `tp4-switched` (sp1..sp4, head sp1).

## Source

- vLLM: candidate-50's series without `0778-perf-dsv41-sp-prefill-local-attention-front.patch` (#1009):
  1107's `0001..0761` (641 active) plus candidate-50's `0762..0777`, `0779` and `0780`, 659 patches listed
  by relative path to their files. Then the patches below, renumbered from `0781` in the selection's
  order ([vllm/series](vllm/series)). Each is copied from its PR's head. The provenance and SHA-256 of each
  are in [receipts/selection.json](receipts/selection.json), and the stack log is
  [receipts/stack.log](receipts/stack.log).
- `patch_count` 662, `patch_head` `51a16b66`, `expected_tree` `1116740c`,
  `patchset_sha256` `5ef94129`, recorded by `bin/spark build prepare --lock`
  ([source.json](source.json), [upstreams.lock.json](upstreams.lock.json)).
- Every other source is 1107's.

| Patch | PR | Title | PR head | Source |
| --- | --- | --- | --- | --- |
| `0781-fix-dsv41-align-moe_swiglu_cast-with-DeepSeek-Expert.patch` | #1030 | fix(dsv41): align moe_swiglu_cast with DeepSeek Expert.forward BF16 rounding | `a7edd33143aa` | the PR's patch |
| `0782-perf-spec-decode-remaining-token-draft-cap.patch` | #968 | perf(spec_decode): clamp scheduled decode verify rows to remaining max_tokens | `1445e800b172` | the PR's patch |
| `0783-perf-dsv41-direct-insert-packed-engram-cache-and-ali.patch` | #969 | perf(dsv41): direct-insert packed Engram cache rows and align split cache arrays | `aad967cbe897` | the PR's patch |

**The base leaves out #1009.** It causes candidate-50's `tp4-ring` boot failure
([candidate-50/receipts/tp4-boot-failure](../candidate-50/receipts/tp4-boot-failure)) and is in rework;
when its rework passes it joins a later base. Without `0778` the base's tree is `14c5a2dc`, the same as
candidate-50's prepared tree with `0778` reverted, which the selection used ([stack.log](receipts/stack.log),
section 1).

The three apply cleanly with `git am` on the base, in this order ([stack.log](receipts/stack.log), section 2),
and `git diff --check` is clean. Each touches files no other patch here touches (`tilelang/moe.py`;
`v1/core/sched/scheduler.py`; `engram_rows/_reader.c` and `_row_plan.h`, compiled at runtime by `_native.py`).
Candidates 51, 52 and 53 touch no common file, test files included. Every patch the round left out, with its
reason, is in [receipts/selection.json](receipts/selection.json) (`left_out`).

## Cluster configurations

A cluster configuration for every shape the recipe serves, so the candidate can be run on any setup: `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-52/<file> …` with your site's node map. Each one extends `releases/spark-ds41f-1107/cluster.json` at its shape and sets only the image `spark-ds41f:candidate-52`, its tree labels and this lock; no PR changes the recipe. The `-dev` file adds the dev-mode collective RPC that the screen's engine grid needs. This candidate is measured at `tp4-switched` only; the other files are configurations, not part of its runs.

| Shape | Configuration | With dev RPC | |
| --- | --- | --- | --- |
| `tp4-ring` | [tp4.json](tp4.json) | [tp4-dev.json](tp4-dev.json) |  |
| `tp3-triangle` | [tp3.json](tp3.json) | [tp3-dev.json](tp3-dev.json) |  |
| `tp3-triangle-4k` | [tp3-triangle-4k.json](tp3-triangle-4k.json) | [tp3-triangle-4k-dev.json](tp3-triangle-4k-dev.json) |  |
| `tp3-switched` | [tp3-switched.json](tp3-switched.json) | [tp3-switched-dev.json](tp3-switched-dev.json) |  |
| `tp4-switched` | [tp4-switched.json](tp4-switched.json) | [tp4-switched-dev.json](tp4-switched-dev.json) | tested here |
| `tp5-switched` | [tp5-switched.json](tp5-switched.json) | [tp5-switched-dev.json](tp5-switched-dev.json) |  |
| `tp6-switched` | [tp6-switched.json](tp6-switched.json) | [tp6-switched-dev.json](tp6-switched-dev.json) |  |
| `tp7-switched` | [tp7-switched.json](tp7-switched.json) | [tp7-switched-dev.json](tp7-switched-dev.json) |  |
| `tp8-switched` | [tp8-switched.json](tp8-switched.json) | [tp8-switched-dev.json](tp8-switched-dev.json) |  |

- [image-dgx4.json](image-dgx4.json): build-only profile.
- [tp4-switched-no1030.json](tp4-switched-no1030.json), [tp4-switched-no1030-dev.json](tp4-switched-no1030-dev.json):
  candidate-52 without #1030. They use the same image, with `vllm/models/deepseek_v4_1/tilelang/moe.py` mounted
  from [no1030/overlay](no1030/overlay): candidate-52's file with `0781` reverted, which is byte for byte the
  base's. #1030 is pure Python (a TileLang kernel compiled at runtime), and no later patch here touches that
  file, so the arm is candidate-52 minus exactly #1030. Its patch applies cleanly to the overlay file
  (`git apply --check`). SHA-256s in [receipts/selection.json](receipts/selection.json) (`no1030_overlay`) and
  the commands in [stack.log](receipts/stack.log), section 3.

## Release tests

[release-tests.json](release-tests.json) is candidate-50's classification. These PRs change only test files it
already gates (`tests/kernels/moe/test_deepseek_v41_tilelang_moe.py`, `tests/v1/core/test_copy_drafts_scheduler.py`,
`tests/kernels/test_deepseek_v41_engram_rows.py`), so no file needed a new classification.
It also carries main's release-test change that landed after candidate-50 (#1038): the bundle resolves
upstream tests' stand-in hub models to the served checkpoint, and two `tests/v1/core/test_scheduler.py` tests
that then run are deselected with their reason, as in `config/release-tests.json`.
`scripts/release_tests.py check` reports no problems on dgx4's prepared tree: 215 selected files, 153 gate and 62 excluded
([receipts/release-tests-check.txt](receipts/release-tests-check.txt)). The gate manifest is
[release-tests-gate-manifest.json](release-tests-gate-manifest.json) (154 files), and the bundle is
[bundles/release-tests](bundles/release-tests).

## Runs

Each testbed opens its window with `scripts/lab.py`, after the hold and idle guards.

1. [build.json](build.json) (`candidate-52-image`), in the round's one dgx window
   ([release-benchmark](../../../docs/release-benchmark.md#4-build-and-release-tests-dgx-window), step 4):
   build the image on dgx4 and copy it along the ring. It is queued after candidate-51's and before
   candidate-53's; candidate-53's ends with the round's one `kernel` job, which runs this candidate's
   release tests ([bundles/release-tests](bundles/release-tests)) on dgx3. The image is then streamed to sp1
   (`ssh dgx1 docker save spark-ds41f:candidate-52 | ssh sp1 docker load`) and distributed from there with
   `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-52/tp4-switched.json build distribute --apply`.
   Every rank shows the same image ID.
2. [candidate-tp4-switched.json](candidate-tp4-switched.json) (`candidate-52-tp4-switched`): the determinism
   gate at `tp4-switched`, first ([step 5](../../../docs/release-benchmark.md#5-gate-first)). It boots two arms:
   - `candidate-52` ([tp4-switched.json](tp4-switched.json)): the startup check, and the record of #1030's
     declared output change. Its hashes differ from the 1017 checkpoint's by construction.
   - `candidate-52-no1030` ([tp4-switched-no1030.json](tp4-switched-no1030.json)): **the identity check of
     #968 and #969.** Its hashes must equal the 1017 checkpoint's.
3. [screen-tp4-switched.json](screen-tp4-switched.json) (`candidate-52-screen-tp4-switched`): the `screen` suite
   on `candidate-52` and `candidate-52-again`, with startup memory ([step 6](../../../docs/release-benchmark.md#6-screen)).
   - **Reference:** candidate-50's screen at `tp4-switched`, the accumulated base, recorded at
     `experiments/2026-10-26-spark-ds41f-candidate/candidate-50/receipts/references/candidate-50-no1009-tp4-switched-screen`.
   - **The reference is pending.** candidate-50's screens are running, and the coordinator commits the record.
     Until it is there, the lab's dry run reports the reference as missing.
   - #1030 changes the text, so the screen's greedy identity against the reference fails by construction
     (declared below). Its speed is judged on distinct prompts, as the release rules ask
     ([lab](../../../docs/lab.md)): the closed loop's paired decode, and the engine grid, which does not
     depend on the text.
4. **Own marks** on sp8 (below), two boots each, ABBA against the base.

## Marks

Declared before building or measuring ([methodology](../../../docs/methodology.md#round-purpose-report-and-drop-rule)).
A PR earns its keep in one of two ways:

- **Correctness fix:** its functional evidence passes on the candidate image, and no core metric is worse.
- **Performance change:** it improves its own metric, and no unrelated core metric is worse.

"Worse" and "better" mean beyond the screen table's intervals: the candidate's two boots, and the
reference's repeat (candidate-50's two boots without #1009). The paired decode row is judged against the
recorded A/A floor.

**Core metrics** (the screen, the gate, the release tests):
- greedy identity against the reference (the screen's model step), except #1030's declared change;
- the engine decode grid;
- paired decode in the 16-stream closed loop;
- prefill at 32K and 256K;
- the determinism gate (temperature-0 output hashes equal to the 1017 checkpoint's, unless a PR declares a change below; here, on the no1030 control);
- startup (lowest `MemAvailable`, time to ready, no JIT after readiness) and steady memory (KV capacity, swap, OOMs);
- zero failed requests, and no silent fallback in any rank's boot log;
- the release tests (`check` reports no problems, and the bundle exits 0).

A kernel PR's own mark is measured on the candidate image, the integrated stack, by the PR's own
benchmark: two boots on sp8. When a core metric is worse, the PRs whose path can move it are
removed one at a time until the cause is found. Each PR here touches files no other patch touches, so each
can have an overlay control (`no1030` is prepared; `no968` and `no969` would mount `scheduler.py`, and
`_reader.c` with `_row_plan.h`).

| PR | Kind | Own mark | Declared output change |
| --- | --- | --- | --- |
| #1030 | fix | Its `bench_reference_alignment.py` on the image, 2 boots: routed `moe_swiglu_cast` FP8 bytes and UE8M0 scales identical to DeepSeek's `Expert.forward` reference (0 diffs at TP4 inter=576 and TP3 inter=768), and the end-to-end MoE closer to the reference than 1107's; kernel time not worse beyond the interval at the decode, mid and prefill tiers. A precision change: the gate checks #968 and #969 on the no1030 control, speed is judged on distinct prompts, and its correctness evidence is the reference alignment plus release qualification's quality evals | temperature-0 text changes on every shape: routed MoE activations are rounded to BF16 before MXFP8 quantization, as in DeepSeek's reference |
| #968 | perf | Its `screen_remaining_token_cap.py` on the image, 2 boots: verify rows past a request's remaining `max_tokens` are not scheduled (functional), outputs bit-exact; paired decode not worse; the tail steps of bounded requests at 16 streams lighter | none |
| #969 | perf | Its Engram row-cache bench (`run_cache_and_packed_benchmark`) on the image, 2 boots, ABBA against the base image in the same session: cold-miss and Zipfian lookups faster beyond the interval; the engram-row tests pass; rows identical | none |

### Own-mark commands

On sp8, from a checkout of this commit, each command under `flock /tmp/spark-lab-gpu.lock`, receipts in
`receipts/own-marks/pr-<N>/` with a `result.json`
([step 7](../../../docs/release-benchmark.md#7-own-marks)):

- **Images.** The candidate is `spark-ds41f:candidate-52`. The base is `spark-ds41f:candidate-50` with
  candidate-50's four `no1009/overlay` files mounted over it, as [candidate-50's tp4-no1009.json](../candidate-50/tp4-no1009.json)
  mounts them. Stream both to sp8 first (`ssh dgx1 docker save spark-ds41f:candidate-<N> | ssh sp8 docker load`,
  for candidate-52 and candidate-50).
- **ABBA in one session:** base, candidate, candidate, base. Two boots of each arm. The base arm is not a
  release record, so measuring it here is allowed.
- The PRs' commands patched or mounted their change into `spark-ds41f:1107`. On the candidate image the change
  is already in, so run the bench scripts from the PR heads without those mounts.

Per PR:

- **#1030.** Script: `experiments/2026-11-09-deepseek-reference-alignment/bench_reference_alignment.py` at
  `a7edd331`, `python3 bench_reference_alignment.py <out.json>` with `-w /opt/spark-ds41f/candidate/vllm`. It
  compares an embedded 1107 `moe_swiglu_cast` against the image's and both against DeepSeek's reference
  arithmetic. On the candidate image that is the PR's own comparison; on the base image both arms are 1107's
  kernel, which gives the A/A spread of its kernel timings. Tests in the candidate image, as the PR ran them:
  ```bash
  pytest --noconftest -v -p tests.kernels.deepseek_v41_pdl_sources \
    tests/kernels/moe/test_deepseek_v41_tilelang_moe.py tests/kernels/test_deepseek_v41_tilelang_activation.py
  ```
- **#968.** Script: `experiments/2026-11-08-remaining-token-draft-cap/screen_remaining_token_cap.py` at
  `1445e800`, `python3 screen_remaining_token_cap.py` (it prints JSON lines). Run it on both images, ABBA: the
  functional rows must show the cap on the candidate only. Tests in the candidate image:
  `tests/v1/core/test_copy_drafts_scheduler.py`, `tests/v1/core/test_sink_draft_zero.py`,
  `tests/v1/spec_decode/test_adaptive_verification.py`, `tests/v1/spec_decode/test_acceptance_estimator.py`
  (`--noconftest -o addopts=""`).
- **#969.** The PR's bench module (`bench_and_verify.run_cache_and_packed_benchmark`) is not committed in the PR;
  only its receipts are (`receipts/bench-cache-boot1.json`, `bench-cache-boot2.json`). Recover it from the PR's
  author, or write a bench of cold-miss and Zipfian lookups whose rows match the receipts, and commit it with
  the own-mark receipts. It runs with `--cap-add=IPC_LOCK --ulimit=memlock=-1:-1 --security-opt=seccomp=unconfined`
  and a bench directory on NVMe (`-v /mnt/nvme:/mnt/nvme`). The PR compared with a recorded `0750` baseline; this
  mark needs the base image in the same session, ABBA. Tests: `tests/kernels/test_deepseek_v41_engram_rows.py`.

**Rework (owner, 2026-11-09).** A PR that misses its mark goes to a sub-agent for one focused
rework, with the miss, the receipts and the mark. A second rework is allowed when the first
shows promise. A PR that still misses is dropped from this release, and its record stays here.
A reworked PR joins the next candidate with its own source identity and fresh qualification.

## Determinism gate: declared output changes

Temperature-0 output hashes must equal the 1017 checkpoint's, except where a PR declares a change:

| PR | Declared output change |
| --- | --- |
| #1030 | temperature-0 text on every shape: routed MoE activations are rounded to BF16 before MXFP8 quantization, as in DeepSeek's reference |

#968 and #969 declare none, so the gate checks them on `candidate-52-no1030`, the same image without #1030.

## Selection

One of candidates 51, 52 and 53, tested at the same time, one per testbed. The PRs that pass are carried into
all later candidates. The selection receipts are in [candidate-51/receipts](../candidate-51/receipts).

## Status

Prepared and locked. Not built, not measured.
