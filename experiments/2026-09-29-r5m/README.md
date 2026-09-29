# r5m candidate: deterministic top-k ties

Date: 2026-09-29. Base: promoted r5l.

## Change

B12X patch 0003 (`patches/b12x/0003-dsa-topk-position-ties.patch`, tree
`35299956`) breaks exact top-k score ties by lowest logical position in the
DSA radix top-k, so indexer selections repeat exactly.
`experiments/2026-09-29-topk-ties` measured it as an overlay of the same file
on r5l: zero selection differences between full-row runs and against the
split, no prefill cost (a 4,096-token chunk at 8K-200K within 0.7% of r5l over
two rounds), decode matched r5k and r5l in an alternating screen, and draft
acceptance unchanged.

## Method

`build.sh` builds the image from the lock and loads it on every node. `run.sh`
boots it, requires every node's `tiled_topk.py` to be byte-identical to the
measured overlay (SHA-256 `f1bdacb5…`), and runs the quality gate, the needle
check at 180K and `doctor --live`. Because the measured file is the one
shipped, r5l's reference bench stands as the benchmark record; a full
reference run was not repeated.
