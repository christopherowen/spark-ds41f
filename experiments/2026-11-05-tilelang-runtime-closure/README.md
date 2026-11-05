# Complete the TileLang runtime replacements

Base deployment: `9815f36b72e6caacf8565a181d271e522c8bafc9`, combining
the five open replacement candidates (#885, #886, #887, #893 and #895).
The promoted configuration and published baseline are unchanged.
Exact source/image identities and file hashes are in `base.json` and
`candidate.json`. The ordered original experiment patches remain authoritative;
`vllm/series` adds the sampler correction after their combined source.

The first causal variable is vectorized aligned sampler copies. The previous
scalar TileLang copy was slower than the independent framework copy for padded
FP32 logits at 96 rows. The correction preserves the one-row grid, FP32 owned
output, all payload values, and the native contiguous FP32 DMA path. Narrow and
unaligned inputs keep the existing scalar TileLang copy. Added tests cover the
vectorized partial final block for all three input types.

Subsequent phases requalify the unchanged gather and rank-reduction candidates,
then explicitly test the interaction of all five replacements and inventory
runtime GPU launches. Each code correction is a separate causal variable.

## Reproduce

Publish the deployment commit, create a clean detached checkout on idle sp8,
read the dgx1 and sp8 holds, then run from that checkout:

```
python3 experiments/2026-11-05-tilelang-runtime-closure/run.py \
  --experiments experiments/2026-11-05-tilelang-runtime-closure \
  --run sampler-v1 --family sampler --operations tests pair
```

The runner reserves only the already authorized idle sp8; both serving clusters
remain running. It records exact commands, immutable image identity, two process
boots per operation, host memory and temperature. It uses the shared GPU lock,
20-second reservation heartbeats, a 32 GiB container limit and an 8 GiB active
host-memory guard. The independent reference is not a rerun of a released build.

Paired GPU measurements use fixed seeded inputs and identical output addresses.
Each replay contains eight cold-cache samples timed by CUDA event nodes inside
the graph. Roles are shuffled and mirrored in each of 120 rounds. Raw samples,
the previous candidate, unchanged-reference controls, generated CUDA hashes,
and framework CUDA traces are retained. Acceptance has no slowdown allowance:
each boot must show a five-standard-error upper ratio at or below one, or prove
the same GPU program and bindings. Inconclusive measurements remain unresolved.
This gate covers device execution; host latency and complete model/fabric
qualification are separate requirements. Results and failures will be appended.
