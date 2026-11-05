# Rank-order qualification

The source candidate remains the original November 4 rank-order patch. No kernel
is changed by this follow-up. The intended variable is measurement precision and
control of identical programs/bindings, to resolve the earlier mixed timing
report. Historical measurements remain unchanged.

Both boots pass all 24 cases at the strict zero-margin gate: a five-standard-error
upper ratio no greater than one, or exact GPU program and argument equality for
the retained packed BF16 path. The earlier apparent increase on that path came
from measuring identical operations; it is now proved equal by generated CUDA,
bindings and input geometry. All replaced candidates have no ATen compute launch.

`results.json` records both boots, profiler inventories and exact source AST
hashes of every affected factory, wrapper and constant, matched against the
composed source. Raw paired samples, generated CUDA, traces, commands and memory
observations are retained in `combined-v4` in the published
`feat/tilelang-runtime-closure` branch's November 5 experiment. That combined
source passes 1,167 tests on both boots. No distributed timing is inferred from
these isolated local arithmetic tests.

The original source series already passed its fresh apply check. This evidence
update changes no patch or promoted recipe. Built-image and physical multi-rank
serving qualification remain before release. Both serving clusters stay running.
