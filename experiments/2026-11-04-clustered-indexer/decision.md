# Retain for evidence; do not integrate or promote

The exact-output requirement stands. All 12 expanded GB10 snapshot cases pass
in two fresh containers, with identical selected-position digests between
processes. Seven CPU arithmetic/selection tests pass. Memcheck reports 0 errors
across 12 cases; racecheck reports 0 hazards, 0 errors and 0 warnings across
10 cases through 16K keys. The initial compile failure is retained. These are
standalone checks, not candidate serving boots or full model qualification.

The speed target is **not achieved or established**. Separable synthetic inputs
retain 576 of 262,143 keys; random inputs retain all 262,143. Full snapshot cost,
including metadata rebuilding, allocation and compaction, remains several
milliseconds. There is no released-build timing comparison or end-to-end claim.
Do not generalize synthetic pruning to actual DeepSeek keys.

Next gates are initcheck and explicitly armed real-model input capture/replay.
The capture arm freshly reproduces the exact source found serving at the start
of the first window; its hardware execution remains untested. If real operands
do not prune enough to cover all costs, reject this grouping and test a different
exact search. Approximate selection must not replace it merely because it is
faster. A changed output requires independent evidence of superior quality.

Both windows closed, at 08:23:17 and 09:07:10 UTC on 2026-11-04, when another
authorized task requested the cluster. The original candidate-36 TP3 profile
was already serving; no serving start, stop, restart or sync was performed.
Holds were removed and foreign requests preserved. Sp8's configured SSH route
was unreachable; dgx4 supplied every GPU result. No promotion is proposed.
