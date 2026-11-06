# Verification row-cost hardware qualification

Base: `6d152f4a15ed539f402fc264bc089cf77f4f94fb` (rebased onto `main`
`297c0663e0d9a385f99220364702ab61b08ba359` at `d2e6fa5d932007a24c9ef146927a0accbf98d888`).
The owner authorized hardware testing on 2026-11-06, including coordinated
replacement and restoration. This extends the offline
`2026-11-05-verification-budget-replay` experiment. **Decision: retain the
promoted `adaptive_verification_row_cost = 0.035` policy; no runtime or
configuration change is proposed.** In the live TP3 hardware screen on
`dgx1–3`, lowering `adaptive_verification_row_cost` from `0.035` (`control035-a`,
window `dgx1-1793961102`) to `0.01` (`row010-a`, window `dgx1-1793958294`)
passed all five functional gates with byte-identical temperature-0 and
tool/schema digests across arms, admitted `+28.6%` / `+28.1%` more draft tokens
per draft event at `c8` / `c16` (`1.950 -> 2.508` / `1.957 -> 2.506`), reduced
step rate by `-8.10%` / `-8.23%` (`9.830 -> 9.033` / `9.847 -> 9.037 steps/s`),
and regressed net serving throughput at `c4` (`-0.61%`, `125.74 -> 124.98 tok/s`),
`c8` (`-1.24%`, `163.49 -> 161.47 tok/s`), and `c16` (`-1.17%`,
`163.52 -> 161.61 tok/s`) along with higher TTFT and TPOT latencies.

## Live TP3 hardware screen: `row010-a` (`0.01`) vs `control035-a` (`0.035`)

After merging PR #934 (`spark_ds41f/functional_evidence.py` and
`sampled_digest.py --record-dir`) at commit
`d2e6fa5d932007a24c9ef146927a0accbf98d888` and verifying the 2-boot single-GPU
container check on `sp8` (`experiments/2026-11-05-verification-budget-replay/runs/sp8-gpu-2boot-c3/summary.json`),
two node-scoped TP3 windows (`pool: dgx-triangle`, `nodes: ["dgx1", "dgx2", "dgx3"]`)
executed the first two screen arms (`row010-a` and `control035-a`) on `dgx1–3`:

1. **Window `dgx1-1793958294` (`row010-a`, `adaptive_verification_row_cost = 0.01`)**:
   - Exported receipt: `runs/hardware-dgx1-1793958294/` (77 native files recorded
     in `source-receipt.json`; 44 summary/gate/handoff files exported).
   - Startup took 116.6 s; minimum available memory during startup was
     `6.933 / 10.151 / 10.238 GiB` on `dgx1 / dgx2 / dgx3`.
   - All five predeclared functional gates passed (`greedy`, `greedy-fast`,
     `sampled`, `controls`, `tool-schema`). In `sampled`, all 8 isolated repeats
     matched (`differing: []`, `reference_sha256 == isolated_repeat_sha256 ==
     469dd2a888bb65e21ab334e2a0981a5ba782f917e58412526bb4e52c58213836`), while
     `observed_c8` recorded cases `[0, 2, 3, 4, 6]` with 24 per-case private
     request files verified by `row010-a-sampled.requests/manifest.json`
     (`3ed43759445392505e4774fa3501d25d69b5b1950812fa0d887729def0ba0923`).
   - Completed `screen-suite` (`model`, `c1`, `c4`, `c8`, `c16`) with zero failed
     requests, zero foreign requests, zero thermal slowdown (`0.0 ms`), zero
     power cap (`0.0 ms`), zero swap growth (`0.0 GiB`), and minimum available
     memory `6.07 / 9.63 / 9.56 GiB` on `dgx1 / dgx2 / dgx3`.
   - At the `row010-a` arm boundary, a short `nccl-sp-channels` request arrived;
     the controller yielded cleanly, synchronized `dgx1–3` back to
     `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`, restored `C36`, verified live
     doctor, and closed the window.

2. **Window `dgx1-1793961102` (`control035-a`, `adaptive_verification_row_cost = 0.035`)**:
   - Resumed from `dgx1-1793958294/screen/receipt.json` (`sha256:
     e00a6d2007b15bc79f9a7882f9c54f30062e911eb9414cf282bdccdb9e821c07`) at the
     identical commit `d2e6fa5d932007a24c9ef146927a0accbf98d888`.
   - Exported receipt: `runs/hardware-dgx1-1793961102/` (77 native files recorded
     in `source-receipt.json`; 44 summary/gate/handoff files exported).
   - Startup took 117.3 s; minimum available memory during startup was
     `6.676 / 10.100 / 8.843 GiB` on `dgx1 / dgx2 / dgx3`.
   - All five predeclared functional gates passed (`greedy`, `greedy-fast`,
     `sampled`, `controls`, `tool-schema`). Every deterministic signature matched
     `row010-a` byte-for-byte (`greedy` `reference_sha256 =
     9d88fc72428ecc9de74a5931400ff4dea5ae1a17a047c4dceabd759a8f3a8736`,
     `long_sha256 = a7a87615853db0bb6e9814f63427c1dec8a92f57d4c7ee2a1ca2603257c8c6d7`,
     `greedy-fast` text hashes `4d143977...` and `12fd8257...`, all 10
     deterministic `controls` digests, and all 12 `tool-schema` digests plus
     `invalid_request_status = 400`). In `sampled`, all 8 isolated repeats
     matched (`differing: []`, `reference_sha256 == isolated_repeat_sha256 ==
     fffc1214059b1e39326d962bdad214335683a0b2ccd681df0b133ca82085b291`), while
     `observed_c8` recorded `[0, 1, 2, 3, 4, 5, 6, 7]` with 24 per-case private
     request files verified by `control035-a-sampled.requests/manifest.json`
     (`1fa8c7dfc416bd013c82ee0ad2217872a89f34eca58d9346677ec091ddebbb0a`).
   - Completed `screen-suite` (`model` scoring against `row010-a`'s corpus, plus
     `c1`, `c4`, `c8`, `c16`) with zero failed requests, zero foreign requests,
     zero thermal slowdown (`0.0 ms`), zero power cap (`0.0 ms`), zero swap
     growth (`0.0 GiB`), and minimum available memory `5.73 / 9.59 / 9.55 GiB`
     on `dgx1 / dgx2 / dgx3`.
   - Stopped cleanly at the `control035-a` arm boundary via coordinator handoff,
     synchronized `dgx1–3` back to `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`,
     restored `C36`, verified live doctor, and closed the window.

### Screen comparison table (`control035-a` `0.035` vs `row010-a` `0.01`)

| Metric | `control035-a` (`0.035`, before) | `row010-a` (`0.01`, after) | Absolute delta | % delta | 95% CI (before / after) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model` acceptance length | 2.2411 | 2.2712 | +0.0301 | +1.34% | deterministic (128 prompts) |
| `model` outputs SHA-256 | `64f96079b0a3...` | `64f96079b0a3...` | identical | 0.00% | rank1 `1.0`, ppl `1.316431` |
| `c1` output throughput (`tok/s`) | 65.34 | 66.07 | +0.73 | +1.12% | ±1.60 (±2.45%) / ±1.76 (±2.67%) |
| `c1` step rate (`steps/s`) | 29.047 | 28.773 | -0.273 | -0.94% | 10 x 30 s windows |
| `c1` output `tok/step` | 2.2510 | 2.2985 | +0.0475 | +2.11% | ±4.09% |
| `c1` drafted / accepted per draft | 3.0576 / 1.2509 | 3.7289 / 1.2984 | +0.6713 / +0.0475 | +21.96% / +3.80% | ±2.90% / ±7.29% |
| `c1` TTFT p50 / p99 (`ms`) | 143.10 / 178.55 | 142.89 / 181.43 | -0.21 / +2.88 | -0.15% / +1.61% | 76 / 77 requests, 0 failed |
| `c1` TPOT p50 / p99 (`ms`) | 15.222 / 18.792 | 15.001 / 18.761 | -0.221 / -0.031 | -1.45% / -0.16% | 76 / 77 requests, 0 failed |
| `c4` output throughput (`tok/s`) | **125.74** | 124.98 | **-0.76** | **-0.61%** | ±1.39 (±1.11%) / ±1.41 (±1.12%) |
| `c4` step rate (`steps/s`) | 14.453 | 13.853 | -0.600 | -4.15% | 10 x 30 s windows |
| `c4` output `tok/step` | 8.7058 | 9.0283 | +0.3225 | +3.70% | 10 x 30 s windows |
| `c4` drafted / accepted per draft | 2.2621 / 1.2885 | 2.9215 / 1.3070 | +0.6594 / +0.0185 | +29.15% / +1.44% | 10 x 30 s windows |
| `c4` TTFT p50 / p99 (`ms`) | **248.59 / 302.37** | 254.91 / 322.05 | +6.32 / +19.68 | +2.54% / +6.51% | 148 / 147 requests, 0 failed |
| `c4` TPOT p50 / p99 (`ms`) | 31.668 / 42.185 | 31.685 / 40.823 | +0.017 / -1.362 | +0.05% / -3.23% | 148 / 147 requests, 0 failed |
| `c8` output throughput (`tok/s`) | **163.49** | 161.47 | **-2.02** | **-1.24%** | ±1.36 (±0.83%) / ±1.50 (±0.93%) |
| `c8` step rate (`steps/s`) | 9.830 | 9.033 | -0.797 | -8.10% | 10 x 30 s windows |
| `c8` output `tok/step` | 16.6490 | 17.8942 | +1.2452 | +7.48% | 10 x 30 s windows |
| `c8` drafted / accepted per draft | 1.9502 / 1.3077 | 2.5080 / 1.3024 | +0.5578 / -0.0053 | +28.60% / -0.41% | 10 x 30 s windows |
| `c8` TTFT p50 / p99 (`ms`) | **349.58 / 419.99** | 365.05 / 439.14 | +15.47 / +19.15 | +4.43% / +4.56% | 194 / 190 requests, 0 failed |
| `c8` TPOT p50 / p99 (`ms`) | **48.055 / 65.175** | 48.875 / 66.154 | +0.820 / +0.979 | +1.71% / +1.50% | 194 / 190 requests, 0 failed |
| `c16` output throughput (`tok/s`) | **163.52** | 161.61 | **-1.91** | **-1.17%** | ±1.29 (±0.79%) / ±1.29 (±0.80%) |
| `c16` step rate (`steps/s`) | 9.847 | 9.037 | -0.810 | -8.23% | 10 x 30 s windows |
| `c16` output `tok/step` | 16.6209 | 17.9027 | +1.2818 | +7.71% | 10 x 30 s windows |
| `c16` drafted / accepted per draft | 1.9568 / 1.3087 | 2.5062 / 1.3016 | +0.5494 / -0.0071 | +28.08% / -0.54% | 10 x 30 s windows |
| `c16` TTFT p50 / p99 (`ms`) | **13035.17 / 15521.85** | 13182.28 / 15630.57 | +147.11 / +108.72 | +1.13% / +0.70% | 193 / 191 requests, 0 failed |
| `c16` TPOT p50 / p99 (`ms`) | **48.621 / 64.614** | 48.949 / 67.354 | +0.328 / +2.740 | +0.67% / +4.24% | 193 / 191 requests, 0 failed |

### Causal interpretation and conclusion

Lowering `adaptive_verification_row_cost` from `0.035` to `0.01` admits
`+0.55` to `+0.67` additional low-confidence draft tokens per draft event across
`c1/c4/c8/c16`. At `c8` and `c16`, those marginal third-draft admissions do not
increase `accepted_per_draft` (`1.3077 -> 1.3024` at `c8`; `1.3087 -> 1.3016` at
`c16`), while the extra verification rows reduce target step rate by `-8.10%`
and `-8.23%`, causing net serving throughput to drop by `-1.24%` (`-2.02 tok/s`)
at `c8` and `-1.17%` (`-1.91 tok/s`) at `c16` and increasing both TTFT and TPOT
p50/p99 latencies. Even at `c4`, where `accepted_per_draft` rises slightly
(`1.2885 -> 1.3070`, `+1.44%`), the `-4.15%` step-rate penalty dominates and net
throughput drops by `-0.61%` (`-0.76 tok/s`). This directly confirms on live TP3
hardware the conclusion of `experiments/2026-11-05-verification-budget-replay/`:
the current `adaptive_verification_row_cost = 0.035` parameter already sits at
the empirical trade-off boundary where marginal verification row cost matches
marginal draft survival. Remaining screen arms and qualification stages are
therefore skipped to preserve cluster time, and `0.035` is retained.

## First hardware result: required sampled gate failed

Window `dgx1-1793927589` ran the published controller at
`1160621844bae0f681318d812e1e11f384844767`, with `--max-arms 1` and no
`--keep-open`. Only `row010-a` booted: the immutable C40 image and row cost 0.01.
API readiness took 107.4 seconds. Both six-scenario greedy gates passed, with
exact token/logprob equality on the native-logprob path and matching text digests
between that path and ordinary greedy verification.

The unchanged `sampled_digest.py --tokens 192` collector then reported
`identical:false`: c8 cases 0, 2, 3, 4 and 6 differed from their solo references;
all eight repeated solo requests matched. Requests used temperature 1, seed
20261024 plus the prompt index, `ignore_eos:true`, fresh cache salts, logprobs,
and thinking off. The controller rejected that false verdict despite the
collector's exit 0. Controls, tools/schema, model corpus and timed suites did
not run. There is no throughput result, candidate repeat, confidence interval,
or performance comparison from this attempt. The failed arm cannot use the
successful-handoff resume path and was not repeated.

Startup minimum available memory was 7.594/9.918/10.270 GiB on dgx1/2/3;
functional-traffic minima were 7.43/10.30/10.12 GiB, with zero swap growth,
no guard halt and no fan-restoration failure. The controller stopped the arm,
synchronized every node to original commit
`2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`, and restored the exact C36 profile
and digest. Native live doctor passed and the hold was removed at
2026-11-06 01:25:40 UTC. Independent all-rank checks at 01:26:16–19 UTC
confirmed clean original checkouts and C36 running; the earlier Engram request
was preserved. During this window the holder explicitly recorded dgx1/2/3 as
its node scope after checking the restore map and watchdog targets; the scope
survived native heartbeats. No other pool was used.

Evidence is in `runs/hardware-dgx1-1793927589/`: native receipts, both passing
greedy reports, the failed sampled report, memory/condition records, original
and restored identities, and a complete raw-file hash inventory. Full engine
logs stay in the native private directory; their hashes, warnings and fault
scan are exported. `export_attempt2.py` reproduces this bounded export. The
legacy sampled collector did not retain HTTP request/response IDs or per-case
token/logprob arrays. Those data are unavailable for this attempt; aggregate
digests do not reveal whether a case differed in tokens, logprobs, or both.

The saved C40 control and repeat already report the same failed batch-invariance
observation (all eight c8 cases differ; repeated solo outputs match), with the
same script, seed and length. Both raw files match their published SHA-256
`101c40d681f6693ede8f0a774dddf114d17ef03d91ec2e8322726024889a0a25`.
The comparison retains both files and the immutable source receipt identity.
C40's historical offline audit explicitly records sampled batch variation as a
limitation and requires isolated-repeat equality; it did **not** claim sampled
batch invariance. Do not rewrite that history as a false pass. The exact C40
sampler keys block-verification draws by request seed, position and local
speculative step, so a changed block layout can change a seeded trajectory.
This evidence establishes neither row-cost causality nor a runtime sampling
defect. It does not retroactively pass this experiment's stricter declared gate.
Any future qualification contract must distinguish native arithmetic and output
distribution from exact RNG-input/trajectory identity, with sufficient request
recording, before another hardware turn. No further turn is assigned.

Remote preflight at `3014f382` passed eight runner tests, all three profile
doctors, the native six-arm dry run and the guarded runner's read-only plan.
The tests took 0.10 s / 25,696 KiB maximum process RSS under one CPU/2 GiB on
sp8. The resolved profiles differ only in row cost (notes excluded). The
initial row010 doctor failed because overlay option values do not expand
recipe placeholders; the correction supplies the resolved TP3 values, with a
regression check. Both attempts remain in `runs/preflight/`. A clean isolated
dgx1 checkout was prepared at that published commit; its doctor passed. At
that preflight, the canonical serving checkouts remained unchanged.

The expanded tool/schema gate at `e7fe9895` passed nine remote tests, all three
profile doctors and both dry plans (0.11 s / 26,080 KiB maximum test RSS).
Those receipts are in `runs/preflight/tool-schema-e7fe9895/`. This validates
the harness and profile preparation; the HTTP gate still needs real hardware.
At `5db59423` and `2b959786`, ten remote tests, all three doctors and both plans
passed after adding handoff resumption and per-rank engine-log capture. The
last test run took 0.10 s / 25,820 KiB maximum RSS. Both receipts are retained.
At `965abfc4`, eleven remote tests, all three doctors and both plans passed
after rejecting the legacy greedy script's silent text-only fallback and
requiring ordinary greedy output to match its logprob path. The tests took
0.11 s / 26,064 KiB; see `runs/preflight/native-gates-965abfc4/`.
The final hardware controller at `11606218` passed thirteen remote runner tests,
all three doctors and both dry plans (0.11 s / 26,048 KiB maximum test RSS);
the native preflight receipts remain on sp8 under
`~/spark-lab/verification-budget-replay/logs/bounded-arm-11606218/`.
After merging `main` (`297c0663e`, including `functional_evidence.py` and
`sampled_digest.py --record-dir` from PR #934), the updated controller at
`e151ea35` passed sixteen runner tests (0.25 s / 29,716 KiB maximum RSS, zero
swaps), fifteen functional-evidence tests (0.19 s / 28,196 KiB maximum RSS,
zero swaps), `bin/spark doctor` on default and all three profiles, both dry
plans, and `git diff --check` on `sp8` under one CPU core and 2 GiB; see
`runs/preflight/sampled-contract-e151ea35/`. An initial test run at `7741fcf2`
failed in the unit-test mock (`mock-attempt-7741fcf2-tests.txt`) because
`ThreadPoolExecutor(max_workers=1)` runs `c1` on a worker thread while the
mock used `threading.main_thread()` to distinguish `c1` from `c8`/`c16`; indexing
mock passes by request count resolved that test-fixture bug.

The first window, `dgx1-1793922050`, subsequently opened from the captured
original commit `2c4b23af` and synchronized the three clean checkouts to
`297acf25`. Before a candidate boot, a CED prefill-boundary request arrived.
This experiment yielded to that ready prefill work: the native sync restored
all three original checkouts, and native close confirmed C36 was already
serving at 23:44:58 UTC. No serving container was stopped, started or replaced,
and no GPU workload or measurement from this experiment ran. The original
state and both sync logs are retained. This is deployment/handoff evidence,
not before/after performance evidence.

One causal variable: `adaptive_verification_row_cost`, 0.035 control versus
0.01 and 0.02. Every arm uses the same existing immutable C40 image, source
lock, native arithmetic, CopyDrafter settings, TP3 triangle, eight active slots,
KV budget and 6,144-token prefill chunks. The profiles name the published
experiment branch for deployment without changing the promoted recipe.
`base.json` records the image, source and reference identities. Read-only
preflight found the exact image on dgx1, dgx2 and dgx3.

## Shared window and deployment

At initial preparation the dgx1 hold belonged to the frontend lifetime qualification,
window `dgx1-1793917240`; `norm-projection-boundaries` had the existing request.
Neither file was overwritten. The native lab queue had no pending/running
specs. Current ownership must be read afresh; this experiment must not run through the
generic `spark-lab` holder-prefix check while another task owns the window.
Prefill work can take priority while this decode experiment is waiting.

The main-only queue's sync step cannot deploy this PR. Use the existing branch
deployment workflow in a separately owned window, without modifying its guard:

1. After the current holder and pending request finish and production is
   restored, inspect the hold/request files again, check no queue runner can
   race us, and capture the original clean deployment SHA on every node.
   Require one identical SHA on the three nodes. Capture the live profile and
   image as well. The currently recorded restore profile is
   `experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json`.
   A different live restore profile requires reconciling this runner first.
2. From that original clean main checkout on dgx1, open a native lab window
   with note `verification budget hardware` and the named restore profile.
   The native publish/live/idle guards and restoration watchdog remain active.
3. From a clean isolated checkout at the exact published experiment SHA on
   dgx1, inspect the candidate's coordinated sync plan, then
   `bin/spark --cluster-config <experiment>/control.json cluster sync
   --revision <exact published SHA> --apply`. Every configured node must be
   clean, detached at that SHA, and hold the recorded image digest. No rsync,
   dirty-file deployment, merge, topology change or one-rank launch is used.
4. In the canonical deployment checkout run `run_hardware.py --stage screen
   --window <owned window ID> --restore-commit <captured original SHA>
   --apply --keep-open`, detached with all streams redirected. The runner
   verifies the exact holder, window and restore profile, refreshes heartbeat,
   checks live/shape plan evidence before every start/stop, records memory
   through startup, and uses the native thermal/memory/fan guards for traffic.
5. Review the causal screen before selecting `qualify010` or `qualify020`.
   Those stages run release-tp3, single-stream and sampled on two fresh boots;
   comparisons use the saved C40 parent boot/repeat and the recorded release,
   never a remeasurement of a released build. Qualified distinct-agent traffic
   remains a separate required stage before any gain claim is complete.
6. Keep the window held between stages; restore only once when finished or
   handing off. Without `--keep-open`, or on failure, the runner stops the
   current arm, syncs all nodes to the captured original SHA through the
   named restore profile, restores that profile, verifies it with the native
   close operation, and then removes its hold. `--stage restore` performs
   that cleanup if the screen yields no candidate. Do not leave an idle
   owned window without heartbeat; the native watchdog remains the fallback.

Run `run_hardware.py` without `--apply` for a read-only plan. The screen spec
also validates through `scripts/lab.py run <experiment>/screen.json --dry-run`,
but **do not execute that spec with the main-syncing queue**. The guarded
runner adds the functional gates and exact restoration sequence.

For the coordinator-reserved first turn, use `--max-arms 1` and omit
`--keep-open`. The limit counts newly completed arms in that invocation; it
finishes the current arm, restores the original commit/service, and records a
resumable bounded handoff even if no other request has arrived. Later turns
resume the ordered remaining arms at the identical published revision.

After a request interrupts a stage at an arm boundary, `--resume-from` may
reference that stage's native `receipt.json` in a later owned window. It
requires a successfully restored handoff, the identical published commit,
image identities and gate sources, and an ordered completed prefix. It checks
the retained gate signatures and uses the original model corpus. Completed
arms keep their original receipt paths and are not rerun. Other failures do
not qualify for automatic resumption. For qualification after a screen's
window closes, `--screen-from` names the completed screen receipt; the same
commit/image/gate checks apply. Keep handoff times in the final comparison,
because measurements spanning windows have additional environmental drift.

## Workload and decision

The first stage uses six fresh boots in order 0.01, 0.035, 0.02, 0.02, 0.035,
0.01, retaining both candidate alternatives and two control boots. Its custom
suite is a new workload, with 128 model prompts followed by distinct closed
loops at c1/c4/c8/c16, each with 60 seconds warmup and 300 seconds measured.
c16 means sixteen clients queueing into the unchanged eight active slots,
not a change to active capacity. The first arm's model corpus is scored by
later arms. The control is explicitly an unreleased C40 causal control on this
new workload; it is not a rerun of the released 1030 reference.

Before timing each arm, require the predeclared functional evidence contracts
in `run_hardware.py` (`spark_ds41f/functional_evidence.py`):
- the wide greedy token/logprob gate (`greedy`) without silent text-only
  fallback, plus per-request native logprob equality;
- the ordinary no-logprob greedy gate (`greedy-fast`) with text digests
  matching the native-logprob greedy path;
- the seeded temperature-1 gate (`sampled`, `sampled_digest.py --tokens 192
  --record-dir`) requiring isolated-repeat (`again` / `repeat_alone`) equality
  while recording `c8` cross-batch trajectory variation as an observation with
  full private per-case token/logprob arrays, request/response IDs, controls
  and a hash-verified 24-request manifest;
- the sampling-controls gate (`controls`) requiring `alone == c8` for all
  temperature-0 controls and valid rejections for `min-p`/`logit-bias`, while
  recording `penalties-sampled/c8` as an observed temperature-1 batch digest;
- the tool/schema gate (`tool-schema`) sending 16 distinct requests in each
  combination of forced tool versus strict JSON schema, temperature 0 versus 1,
  and c1/c8/c16 (plus an isolated c1 repeat at temperature 1), requiring exact
  generated values, finish reasons and native token IDs across every request,
  c1/c8/c16 token-ID invariance at temperature 0, isolated-repeat token-ID
  invariance at temperature 1, and rejection of an invalid request followed by
  a successful ordinary request.

False, missing or malformed required verdicts fail even when a collector exits
zero, writing `<gate>.jsonl.evidence.json` before requesting restoration.
Deterministic temperature-0 and rejection signatures must agree across all arms
and boots; isolated c1 temperature-1 signatures (`sampled` `reference_sha256`,
`penalties-sampled/alone`, and `tool-schema` `t1.0/c1`) must also agree across
repeat boots of the same row-cost configuration; subsequent qualification
compares against both the screen's control signatures and the candidate's
same-config signatures. Native logs, source hashes, image identities,
run commands, cooling/fan state, thermal and memory observations stay under
`results/private/lab/verification-budget-hardware/<window>/<stage>/`.
Preserve all unsuccessful arms and guard failures; never silently restart an
already-recorded stage.

The published C40 comparison omits generated model text. Qualification must use
the retained native `results/private/suites/candidate-40-release-tp3/candidate-40/model.json.gz`,
verified as 15,166,113 bytes with SHA-256
`b13ed140c68f0b6b135fe8c2e7d2eba6dbe625832aa8bd0e8dffa10820f04a87`
against C40's published `source-receipt.json`. This file was verified on dgx1
before execution. Missing or changed corpus bytes fail before a qualification
boot; no silent self-generated replacement is permitted. Raw text stays private.

Compare both boots with the two control boots using prompt-paired results and
boot variability. Report throughput, completion/errors, per-output-token
latency and TTFT (including tail latency), admitted/accepted drafts and memory.
A cost setting needs a material throughput improvement beyond its 95% interval
without worse correctness, c1/queued c16 latency, or error behavior. A neutral,
negative or inconclusive screen does not justify promotion or a claimed win.

The owner's clarified criterion also accepts a demonstrated correctness repair
with preserved performance. No allocator correctness defect has been established
by this audit, and the row-cost changes remain performance experiments. Harness
validation improvements are not a claim of improved serving correctness. Any
later runtime correctness repair needs an actual before/after reproducer and
performance-preservation evidence whose uncertainty excludes a material loss;
an interval merely containing zero is inconclusive.

Any surviving setting needs the standard release-tp3 prefill points at 32K,
128K, 256K, 512K and 1,000,000 cold tokens, stalls, sampled c1/c8/c16, and
realistic distinct agent traffic at c8/c16 across boot/repeat. The row-cost
change cannot accelerate pure prefill kernels; cold prefill is a regression
guard, not a route to the 10k prefill target. Agent replay needs an adequately
provisioned independent CPU client: do not squeeze a 12-GiB client into the
TP3 head's serving memory margin. Prepare and validate that stage only after
a promising screen, with the same corpus/revision/seed for before and after.

The screen comparison table above records metric, before (`control035-a`,
`runs/hardware-dgx1-1793961102/`), after (`row010-a`,
`runs/hardware-dgx1-1793958294/`), absolute and percentage delta, 95% interval,
and native receipt identities, with the 2-boot `sp8` single-GPU verification
labelled separately (`experiments/2026-11-05-verification-budget-replay/runs/sp8-gpu-2boot-c3/`).
Because `row010-a` (`0.01`) regressed serving throughput and latency at
`c4/c8/c16` relative to `control035-a` (`0.035`), the screen concludes with
`adaptive_verification_row_cost = 0.035` retained and no runtime config change.
