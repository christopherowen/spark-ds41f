# L2 access retention

Base deployment: `eb8a49f6d92231acaaf3f9fd63a4b7d3a4183854`.
Runtime: candidate-38, vLLM `1f33102547e4da5c64712390604695d1abd577c4`,
tree `c061e5f6931306fa4655ba531428f639c365bf10`, image
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
Promoted reference remains spark-ds41f-1030; do not benchmark it again.

One intended variable: cache retention of a precisely bounded, repeatedly read
region. No changes to weights, arithmetic, topology or promoted configuration.

## Existing evidence and opportunity

- Current side-stream prefetch uses bulk evict-last reads, grid 2, and separate
  WO/COMP/FFN/NEXT/ENGRAM schedules. Owner and Engram scheduling avoids filling
  the next region before its predecessor consumes its lines (0572 and the
  subsequently reported 0635-era work). Candidate-38 selects TP decode shards.
- The October 25 MoE overlap study found substantial evict-last survival at
  set-aside zero. It rejected an evict-first expert stream and large partitions:
  default demand reuse paid for the retained region. Short-context activations
  are used immediately; prefill activations exceed L2. Its timing-derived
  survival estimates are not hardware hit-counter measurements.
- PR 766 was dropped after candidate-28 screening: a 16.8 MB partition made
  prefill 18–24% slower. This is evidence against a large permanent partition,
  not evidence that every small access-policy window fails.
- The October 28 Markov-window probe failed on October 29 before any timing
  because the checkpoint config mount was absent. Preserve that failure; do
  not call it a negative performance result. Its earlier trace bound was at
  most 0.5% of a one-stream step, and its unfused projection predates the
  candidate-38 fused Markov/greedy path.
- Markov is a credible repeated region: one shard is reread at each draft
  position. Router/projection weights are mostly used once per layer per step;
  retention cannot make their first demand access arrive earlier. Small
  metadata is produced immediately before use and has no demonstrated exposed
  eviction cost. A whole-model arena window would retain unrelated streaming
  data and is excluded.
- Candidate-38's available November 4 Nsight/IKET trace is a one-GPU dummy-weight
  run with TP4-shaped dimensions, no distributed peers and no TP>1 fused Markov
  path. It does not replace a real serving trace or establish this component's
  critical-path share. A positive result still needs the latter measurement.

## Gates

First run `capability.cu` in two fresh processes on GB10. Record actual limits,
stream attributes, extended launches, captured and explicit graph attributes,
readback, and cleanup. The default is queried, never inferred from a helper log.
This probe is not a performance benchmark.

Only after capability passes, compare the existing native fused Markov kernel
under ordinary demand, explicit prefetch, access window, and both; include a
separate null arm, multiple partition sizes and TP3/TP4 dimensions at 1/8/16
requests. Run with cold-start and intervening expert-sized streaming traffic.
The retention mechanism tests the interaction of reservation and window:
partition-only arms distinguish their costs, and driver-default prefetch arms
avoid conflating the native prefetch operation with a forced-zero reservation.
Record every sample, exact output comparison, memory and failure. Any positive
component result requires current serving traces and a full-path c1/c8/c16,
prefill and coexistence comparison across two boots before a runtime PR.

Partitions are context-wide and all concurrent streams share them. Setting a
stream window alone does not reserve bytes or prefetch the first read. Verify
attributes on graph nodes; changing a launch stream after capture is not proof
that a graph changed. Synchronize before changing limits, clear windows and
reset persisting lines, restore the original limit, and destroy only resources
owned by the probe. Do not put limit changes or context resets on the serving
step path. No runtime candidate has yet been implemented.

GPU execution requires exclusive ownership of the dgx1 hold and the host GPU
lock. Preserve every preceding request. This experiment has no
authority to change an existing holder's record.

## Commands and results

Preparation: `bin/spark doctor` initially refused the new checkout because it
has no ignored site map. No GPU operation or serving change was attempted.
CPU-only container introspection first used nonexistent `python`; repeated
with `python3` and succeeded. These are preparation errors, not GPU results.
The publication review's bytecode-cache write hit local disk exhaustion;
compiling the Python sources in memory without writing cache files passed.

Prepared in a separate clean deployment checkout on dgx4, detached at each
published experiment commit. The capability binary was compiled without GPU
execution, with:

```sh
flock -n /tmp/spark-cpu-analysis.lock taskset -c 0,1 \
  /usr/local/cuda/bin/nvcc -std=c++17 -O2 -arch=sm_121 \
  /home/swank/spark-lab/l2-access-retention/checkout/experiments/2026-11-06-l2-access-retention/capability.cu \
  -o /home/swank/spark-lab/l2-access-retention/bin/capability-6ad22720
```

That initial preparation binary's SHA-256 is
`99760b979db6dceea6a3f4f2f495be5e16b3c8ca376fc5d583d6a6dc850e5127`.
The subsequent capability revision separately reports whether stream capture
inherits its policy and verifies explicit graph-node attributes. The runner
builds the current published source using the same compiler command before
cooling, saves source and binary hashes, and executes that version. It verifies
the clean checkout revision, exact hold
ownership, GPU lock, an empty GPU, memory floor and cooling/fan curve before
execution. It retains every output, failure and thermal/memory sample under
`/home/swank/spark-lab/l2-access-retention/boot-<boot>-<revision>/` on dgx4.
Each process has an 8-minute deadline, a fresh compilation cache, a two-CPU and
8 GiB container bound, a 16 GiB host memory floor and an 85 C thermal ceiling.
The checkpoint hub is mounted at its original path so blob symlinks resolve.
The native graph probe records captured attributes before any intervention;
if inheritance is absent it explicitly sets and verifies kernel-node windows
before graph instantiation, and records both versions. No policy API is called
on a timed replay path. Verification includes the exact payload address, size,
hit fraction, persisting hit property and normal miss property. The capability
process has a separate 30-second deadline.

Once preceding requests have run, the controller is invoked from a clean,
published checkout on dgx1 using its full 40-character revision:

```sh
python3 experiments/2026-11-06-l2-access-retention/run_window.py \
  --revision "$(git rev-parse HEAD)" --boot 1
```

It imports the original clean serving checkout's lab guards, verifies the live
profile, waits for idle, atomically acquires its own hold, and starts the normal
watchdog. A 20-second heartbeat covers each job. It runs both boots back to back
unless a pending request requires yielding after boot 1; `--boot 2` resumes in
a new window. Closing verifies/restores the original profile once. Serving
checkouts and topology are not changed by this component experiment.

Local checks: `bin/spark doctor` passes after initializing ignored site maps;
`git diff --check`, shell syntax and Python compilation pass. `bin/spark build
check` reports the pinned build tree is not prepared. No upstream patch or new
runtime image is part of this experiment.

### Hardware execution on GB10 (`dgx4`)

Executed under an exclusive node-scoped `dgx4-single` hold (`~/spark-hold.json` on
`dgx4`, `nodes: ["dgx4"]`) with a 15-minute stale-heartbeat watchdog and 20-second
heartbeat updates while `dgx1` had no conflicting global hold.

1. **Initial Boot 1 attempt (`7626c1007efcdc9caf51f588124863fe54d35c23`)**:
   - `capability.cu` passed on GB10 (`SM 12.1`, driver/runtime `13000`,
     `l2_bytes = 25,165,824` [24 MiB], `persisting_max_bytes = 18,874,368`
     [18 MiB], `window_max_bytes = 134,217,728` [128 MiB], driver default
     `original_bytes = 4,718,592` [4.5 MiB], `1,048,576` requested -> `1,572,864`
     [1.5 MiB] applied; `capture_inherits: true`, `graph_node: true`,
     `extended_launch: true`, `extended_capture: true`, `restored_bytes: 4,718,592`).
   - `markov_probe.py` failed before timing with `CUDA_ERROR_INVALID_CONTEXT (201)`
     at `cuCtxGetLimit` because `torch.cuda.init()` + TileLang compilation had not
     bound the device primary context to the host thread, and CUDA 13's Python
     driver bindings expose the stream/kernel-node attribute ID under the
     `CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW` enum alias. Preserved in
     `/home/swank/spark-lab/l2-access-retention/boot-1-7626c1007efcdc9caf51f588124863fe54d35c23/`.

2. **Initial 16-arm 2-boot screen (`13fabf4966e445a014ad11ffd99faf7dc72353f4`, `summary-initial-13fabf49.json`)**:
   - Bound the primary context via `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`
     and ran two fresh-process boots (`boot 1` and `boot 2`, 31 randomized trials
     per case per boot).
   - **Correctness**: 100% bit-identical outputs (`all_bits_equal: true`) across all
     16 arms, `TP in {3, 4}`, `rows in {1, 8, 16}`, and `stress in {False, True}`.
     Original `4,718,592`-byte persisting limit restored on exit.
   - **Memory & thermals**: Peak allocated GPU memory `249,259,008` bytes (~238 MiB),
     peak reserved `299,892,736` bytes (~286 MiB), host `MemAvailable` ~121 GiB,
     starting thermal zone 39.7 C, peak 45.1 C (well below the 85 C guard).
   - **Results**:
     - Fractional windows at the driver-default 4.5 MiB budget (`window_default`,
       `hitRatio = 0.285` on TP3 / `0.380` on TP4) and 1.5 MiB budget (`window_small`)
       showed no gain over `driver_default` (2-boot 95% CIs span zero in 11/12 cases)
       and were slower than `prefetch_default` (including `+11.86 us` / `+3.84%`,
       95% CI `[+7.52, +16.21] us` at `TP=3, rows=16, stress=False`).
     - Full-shard persisting partitions (`window_fit`, applying `17,301,504` bytes
       [16.5 MiB] on TP3 and `12,582,912` bytes [12.0 MiB] on TP4) beat the
       partition-only control (`partition_fit -> window_fit` `-38.21` to `-75.58 us`
       on TP3 because `partition_fit` shrinks normal L2 to 7.5 MiB and regresses
       demand reads by `+9.1%` to `+15.6%`), but `driver_default -> window_fit`
       still spans zero in 5/6 TP3 cases and `prefetch_default -> window_fit` spans
       zero in 11/12 cases across TP3/TP4. On TP4, `partition_fit -> window_fit`
       spans zero in all 6 cases (`-0.54` to `-5.78 us`). Moreover, a 16.5 MiB
       context-wide partition is already ruled out by PR 766's 18–24% prefill
       regression.

3. **Reworked 24-arm 2-boot screen (`2cc085cb8f24dba69a176de4ae4de9af37f08101`, `summary-reworked-2cc085cb.json`)**:
   - Tested contiguous `hitRatio = 1.0` windows phase-scoped to `partial` only
     (`contig_default` at 4.5 MiB, `contig_mid` at 9.0 MiB applied, plus `prefetch`
     combinations `both_contig_default`, `both_contig_mid`), position-0-only
     windows (`pos0_default`, `pos0_fit`), and `STREAMING` windows on the one-use
     `Base` logits (`base_stream_default`, `both_base_stream_default`).
   - **Correctness**: 100% bit-identical outputs (`all_bits_equal: true`) across all
     24 arms on both boots (`capability-boot1.jsonl`, `capability-boot2.jsonl`).
   - **Results**:
     - Under `stress=True` (96 MiB intervening expert-stream traffic between draft
       positions), every TP3 window arm—including `pos0_fit` (`-24.78` to `-34.68 us`,
       95% CIs `[-269.15, +219.59]`, `[-141.03, +76.67]`, `[-91.79, +22.43] us`) and
       `window_fit`—has a 2-boot 95% CI spanning zero against `driver_default`.
     - Against `prefetch_default` at the safe 4.5 MiB default budget, `contig_default`
       spans zero in 10/12 cases, `both_contig_default` spans zero in 11/12 cases
       (and loses at `TP=3, rows=16, stress=False` by `+11.93 us`, 95% CI
       `[+10.62, +13.24] us`), and `both_base_stream_default` spans zero in 11/12
       cases.
     - Only arms that expand the context-wide persisting partition to 9.0–16.5 MiB
       (`contig_mid`, `both_contig_mid`, `pos0_fit`, `window_fit`) achieve isolated
       10–40 us component reductions (~2–8 us per draft position), and even those
       remain inconclusive across boots under `stress=True` while carving 37.5–68.8%
       out of GB10's 24 MiB L2 cache.

### Conclusion

**Negative / do not promote.** Bounded L2 access-policy retention windows on the
packed Markov/greedy path do not deliver a reliable 2-boot speedup over the
existing driver-default (4.5 MiB) + side-stream evict-last prefetch configuration
at safe partition sizes, and full-shard partitions (12.0–16.5 MiB out of 24 MiB L2)
impose unacceptable context-wide normal-L2 shrinkage for a ~2–8 us/position
component effect that does not survive intervening expert-stream traffic with
2-boot statistical significance. Keep the promoted serving configuration unchanged.
