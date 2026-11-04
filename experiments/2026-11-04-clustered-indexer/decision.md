# Retain for evidence; do not integrate or promote

The exact-output requirement stands. The corrected TileLang implementation passes
all 12 expanded snapshot cases on GB10, including 262K-key contexts. Seven CPU
arithmetic/selection tests pass. The earlier six-case hardware run also passed;
its setup failure is retained. These are standalone kernel checks, not two
candidate serving boots or a model-quality qualification.

The speed target is **not achieved or established**. Separable synthetic inputs
retain 576 of 262,143 keys; random inputs retain all 262,143. Full snapshot cost,
including envelope rebuilding, allocations and compaction, is still several
milliseconds at that size. There is no released-build timing comparison and no
end-to-end performance claim. Do not generalize the synthetic pruning to actual
DeepSeek keys.

Next gates are the expanded repeat and sanitizers, followed by explicitly armed
real-model input capture and replay. Capture instrumentation is prepared against
the exact serving source found at the start of the window, but has not run. If
real operands do not prune enough to cover metadata/update/selection costs,
reject this grouping and test a different exact search. Approximate selection
must not replace it merely because it is faster.

The original window closed on 2026-11-04 at 08:23 UTC. The prior candidate-36 TP3
profile was already serving; no serving start, stop, restart or sync was performed.
The hold was removed. A subsequent short idle-dgx4 window is requested; sp8's
configured SSH route is currently unreachable. Foreign holds are preserved.
