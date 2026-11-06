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
on a timed replay path.

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

GPU results: pending. No speedup, correctness improvement, or performance
preservation is claimed.
