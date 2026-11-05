# Verification budget replay

Base deployment: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
Intended variable: the denominator of the expected-token objective, using
recorded step costs instead of the linear 0.035 row cost. This is an offline
screen; no runtime policy, configuration, arithmetic or source series changes.

The existing allocator already ranks survival across requests. Candidate 15
tried graph/tile/dead-row pricing and candidate 16 removed it after regressions.
A revisit requires a new measured reason. Aggregate acceptance counters cannot
stand in for per-request confidence vectors or counterfactual token matches.

`replay.py` compares the two objectives at the *same measured grid cells*,
without interpolating absent graph or tile boundaries. Its restricted linear
control is not the unrestricted serving policy. The historical five-draft
trace is from r4a, before current checkpoint arithmetic, dead rows and copy
drafts. Pairing it with later grids is cost sensitivity, not a speedup verdict.
The script records missing c16 coverage and the fraction of unrestricted
linear selections whose row count is actually measured.

## Execution record

- Read the required repository guides, current lock and immutable 1030 baseline.
- Recovered the existing r4a trace and its benchmark/cost receipts from dgx1
  with read-only copies. Raw token records remain in ignored storage.
- Initial local source preparation ran out of disk; it did not complete.
  Initial replay failed because NumPy was absent; its dependency setup also
  ran out of disk. Both replay error logs are retained under `runs/`.
- Initial local doctor used the example node map and failed the head-address
  check; `runs/doctor.txt` preserves this error.
- The owner then required resource-intensive work to run remotely. No local
  job survived the workstation restart. Further source preparation, environments,
  replay and tests run on Spaceport sp8, in `~/spark-lab/verification-budget-replay`,
  under a separate CPU reservation, two-core/4-GiB limits. Serving operations
  remain unauthorized and are not performed.

Results, exact commands, identities, memory and conclusion will be appended
after the remote run. No candidate performance or correctness claim is made
from the failed local attempts.
