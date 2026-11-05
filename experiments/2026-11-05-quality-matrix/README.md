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

The fourth attempt (`a04214b3`) completes all 164 Native A and LSA A generations
and both isolated grading jobs. Both score 152/164 against all official tests,
with exactly the same failed task IDs and zero lost native-correct answers.
Native has 162 natural completions; LSA has 163. Native reaches the 65536-token
cap on `HumanEval/145`; LSA completes that task but its answer remains incorrect.
Both stop for repetition on `HumanEval/64`, and both final programs are marked
incomplete and incorrect. The strict candidate-completion gate rejects this run
on that shared failure. It is not evidence of a newly introduced answer loss.

Observed generation wall times are 1220.67 seconds (Native A) and 1052.83 seconds
(LSA A), including thinking and unsuccessful responses. The coordinator restores
without running LSA B or Native B, so these single-arm diagnostics do not qualify
a speed improvement or a repeated-boot quality pass. Short coding prompts also
mostly exercise the native covered-domain path; they do not establish sparse
long-context quality. Every generation, grade, condition and rejection receipt
is retained in `runs/code-window-04`; public receipts contain no program text.
C37 is restored at 19:06:35 UTC. An independent reader verifies the exact image,
clean equal serving revisions, no experimental mounts/environment and no holds
or pending requests on all four nodes. No candidate is promoted.

`--benchmark longbench --split pilot|heldout` now runs the same frozen
LongBench client and phase-aware quality gate on this cooler TP4 group. It
validates the original selection and input hashes before taking leases, keeps
all serial/cold/cached phases, and records four-node conditions. The TP3 thermal
failure is retained; its partial timings are not reused as TP4 controls.
`--candidate-config` selects a published experimental recipe while native and
restoration authority stay fixed. The out-of-band observer can stop only this
owned evaluator on a proven final-answer loss. Every model candidate still
requires its own kernel/lifecycle qualification before a serving window.

The first compact pilot (`08e6ff32`, `runs/compact-pilot-window-01`) boots Native
successfully but stops before sending any model request: the copied frozen
requests omit the pinned `upstream/pred.py` offline answer extractor. This is
an input-preflight omission, not an answer or performance result. C37 restores
at19:39:01 UTC and an independent all-node reader verifies clean checkouts,
the exact image, no experimental mounts/environment and no remaining leases.
All failed client, boot, condition and restore receipts remain retained.
The complete small upstream cache, including its license, is now copied with
pinned checksums. Preflight validates and exercises the official extractor before
taking leases or replacing serving, so a missing grader cannot cause another
wasted model boot. No input, model protocol or grading rule changes.
