# Independent coding and long-context qualification

Base `39bdd99c`. One intended runtime variable: C36 Native68 versus the existing
block8/query-reuse selector, with unchanged checkpoint, fine arithmetic, budgets
and generation protocol. This separately qualifies the idle Spaceport sp1-sp4
TP4 group. Its numbers are never pooled with the sp5-sp7 TP3 LongBench cohort.
No promoted recipe or published release measurement changes.

The tracked C37 four-node switched recipe is the restoration authority. A new
node map selects sp1-sp4; clean isolated controller/serving checkouts preserve
the original production repositories. All nodes synchronize to one published
commit through named `bin/spark cluster sync` operations. Native token leases,
an idle guard, a 30-second heartbeat and an independent 15-minute watchdog cover
the complete window. Plans retain their exact live/shape review before applies.
All configured nodes restore C37 once and release their holds at the end.

`run_window.py --benchmark code` generates all 164 frozen HumanEvalPlus programs
with one attempt, four concurrent cold requests, temperature0, seed0, high
thinking and 65536 generated tokens. Reasoning and failed requests count in
generation wall time. Grading is a separate CPU-only container job after each
arm: pinned official base and plus tests, no network/GPU, read-only code/cache/
samples, no capabilities, no privilege escalation, 2 CPUs, 4 GiB and 64 processes.
Before timed model requests, the same isolated checker validates all 164
official canonical programs against the recorded reference behavior. This is
oracle validation, not a model result; a reference program need not be perfect.
Programs, full responses and reasoning remain private on the node, outside Git.
Only hashes, counters, timing and executable-test verdicts are public. Grading
time is separately recorded; it is not a serving-implementation operation.

`--benchmark ruler --split pilot|heldout` uses the independently frozen context
manifests and official final-content string-coverage metric. Every fractional
score is compared per case, so aggregate gains cannot hide a lost target. These
are project-protocol RULER tasks, not published leaderboard scores. All input
preparation must finish and be frozen before an arm is eligible.

The arm order is Native A, LSA A, LSA B, Native B. Observed regressions reject
the candidate before further boots; final acceptance checks both repeats and
every task, completion and answer-decision stability. A no-loss pass permits
further evaluation only. It does not prove quality on untested inputs, a speed
win, or promotion. Cooling starts below55C on every node, uses the existing fan
policy, and retains memory, swap, thermal and failure observations for each arm.
The existing TP3 LongBench window runs independently; shared-site activity is
recorded rather than treating measurements on different groups as paired runs.

The first window at `5f36b7b9` stops before any serving replacement or model
request: coordinated sync requires its target Git checkout to exist. C37 remains
live on all four ranks, the restoration check passes and every lease is removed.
An independent later reader verifies all four images and absence of holds and
experimental mounts. All failure/restore receipts remain in `runs/code-window-01`.
The controller now prepares missing clean Git clones from the original clean
repository while holding every node's lease. It then uses the named coordinated
sync to fetch and detach every target at the published experiment commit; it
never copies a working tree or ignored files. The retry uses a new output path.

The second attempt (`9e8aead8`) catches a Python local-name collision in the
new coordinator before identity preflight, lease acquisition or output-directory
creation. No serving action or model request occurs. The original controller
log and launch receipt remain in `runs/code-window-02`. The canonical-program
loop variable and runtime-identity helper now have distinct names; a read-only
CLI preflight test reaches the identity guard without acquiring any machine.

The third attempt (`c1a0f455`) synchronizes every clean target successfully but
stops before replacing C37 or sending any model request. Its full reference
check scores 163/164: the bundled `HumanEval/32` Newton iteration passes 100 base
tests but fails 7 of 788 plus polynomial-residual checks. The official checker
uses a mathematical root property rather than exact agreement with that
reference's output. Independent execution on the pinned public inputs also
finds large residuals/nonconvergence; floating-point details vary by platform.
Requiring perfect reference code was an incorrect preflight assumption.

The revised preflight accepts only all-pass reference behavior or this exact
recorded program hash and base/plus counts; every other reference failure stops
the window. **No model task, input, timeout, tolerance or grading rule changes.**
All 164 generated model programs still face the full official checker, including
every `HumanEval/32` test. A model that solves that property more reliably can
outscore the bundled reference. Earlier failure/restore evidence is retained in
`runs/code-window-03`; C37 remains live and all leases are released.
