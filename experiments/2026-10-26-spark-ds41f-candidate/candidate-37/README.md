# candidate-37

Base: `candidate-36` (`0001..0673`, `sparknet` `1fd6a1f`).

Added patches (`0674..0676`):
- `0674-feat-v1-make-loop-detection-default-and-harden-windo.patch` (`#845`): make repetition/loop detection default (`window_size=64, max_repetitions=20, min_tokens=512`) and harden windowed reasoning loop detector across multi-line and single-token cycles.
- `0675-perf-dsv41-exchange-directly-at-four-ranks-on-a-swit.patch` (`#846`): direct exchange at 4 ranks on a switched fabric (`+3.3%` to `+3.7%` prefill on `tp4-switched`) and measured `sparknet` switched NCCL profile.
- `0676-perf-dsv41-skip-dead-padding-rows-in-extend-and-inac.patch` (`#778`, reworked): skip dead/padding rows in `extend` and inactive segments/rows in `decode` `sparse_mla_fwd_v41` while keeping `route = "split"` (`block_H < 32`, `rows = 1..4`) 100% branch-free at compile time (`0.00 us` overhead on `1x2`, `1x3`, and 1..4-row drafter steps) and preserving shared-workspace `NaN` protection in `decode_combine`.

vLLM tree: `af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33` (`5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575`).
