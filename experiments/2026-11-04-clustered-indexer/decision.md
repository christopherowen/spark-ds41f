# Retain for evidence; do not integrate or promote

The exact-output requirement stands. All 12 expanded GB10 snapshot cases pass
in two fresh guarded eager runs and two fresh CUDA graph runs, retaining the
archived selected-position digests. Captured replays also pass after reduced
visibility, an empty active domain and restoration. Seven CPU arithmetic and
selection tests pass. These are standalone checks, not candidate serving boots
or full model qualification.

Strict initialization checking found a pre-existing top-k merge read of
unwritten scratch entries. The independent original reproducer fails with
1,536 errors; the guarded reproducer passes with zero. The complete guarded
search passes initcheck (10 cases), memcheck (12 cases) and racecheck (10 cases),
all with zero errors or hazards. Failures and earlier receipts remain intact.
Only the experimental overlay carries the guard; production was unchanged.

The speed target is **not achieved or established**. Separable synthetic inputs
retain 576 of 262,143 keys; random inputs retain all 262,143. Guarded complete
snapshot medians remain about 3.7 ms and 5.8 ms respectively. CUDA graph capture
does not materially change that cost. Candidate-only profiling attributes about
85% of kernel time to envelope rebuilding and 9–13% to bounds. There is no
released-build timing comparison or end-to-end claim. Do not generalize
synthetic pruning to actual DeepSeek keys.

The next causal variable is the envelope reduction's resource use. Separately,
real-model input capture/replay must establish useful pruning before persistent
metadata and CSA ownership are justified. Capture hardware execution remains
untested. If real operands do not prune enough to cover all costs, reject this
grouping and test another exact search. A changed selected key fails the
experiment unless independent evidence establishes superior model quality.

All three windows closed on 2026-11-04 (08:23:17, 09:07:10 and 10:15:21 UTC).
The original candidate-36 TP3 profile was already serving; no serving start,
stop, restart or sync was performed. Holds were removed, foreign requests
preserved, and dgx4 returned for the shared qualification. Sp8's configured SSH
route was unreachable; dgx4 supplied every GPU result. No promotion is proposed.
