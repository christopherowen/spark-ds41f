# Direct exchange at four ranks on a switched fabric

## Summary

At four ranks the rank-order reduce-scatter's chunk exchange (`tilelang/collectives.py`
`exchange_into`, which sequence-parallel prefill's row exchange also uses) and PyNCCL's
`all_to_all_rows` relay half of the opposite rank's block through each neighbour, because on the
switchless four-node ring ranks two apart share no cable. `tp4-switched` keeps that relay although
every pair is one hop through the switch. The patch exchanges directly with every peer, in one group,
when sparknet renders `SPARKNET_ROCE_TOPOLOGY=direct` (as `tp4-switched` does), and keeps the relay
for any other value or none. Only bytes move: the rank-order sum and every part are the same bits.

On four GB10 behind one switch, on **candidate-36** against candidate-36, two boots each at the
recipe's 16,384-token budget:
- **prefill +3.2% to +3.4%** at 32K-256K;
- decode unchanged;
- greedy outputs and logprobs identical;
- no RDMA errors.

The new tests pass on GB10. On spark-ds41f-1030 the same change read +2.7%.

## Patch

- `vllm/0001-perf-dsv41-exchange-directly-at-four-ranks-on-a-switched-fabric.patch`, on candidate-36's
  vLLM series (`experiments/2026-10-26-spark-ds41f-candidate/candidate-36/vllm/series`: vLLM
  `19f2c20e` plus 0001..0673, expected tree `cc0fdf58`).
  - It applies with `git am` onto a tree that reproduces `cc0fdf58`, giving tree `bc7c9cd5`. That is
    the tree built and measured below (`bin/spark build prepare --lock`: patch head `e6ab6a12`).
  - Rebasing it from 1030 had one conflict, in `collectives.py`'s module docstring (0634 had reworded
    "Fewer ranks" as "Other world sizes"), resolved by keeping both. Both code hunks applied
    unchanged.
- `rc14/0001-DS4.1-Exchange-directly-on-a-switched-fabric.patch`: the same code change on
  spark-ds41f-1030's series (rc14, head `b6de07aa`; with it head `cecd9f79`, tree `6ad77da9`), built
  as `spark-ds41f:1030-direct` for the first measurement.

Code changes (24 lines):
- `pynccl.ring4_relay()`, cached, is false only for `SPARKNET_ROCE_TOPOLOGY=direct`.
- `PyNcclCommunicator.all_to_all_rows` and `collectives.exchange_into` take the four-rank relay only
  while it is true.

Every other world size already exchanges directly.

**Interaction with candidate-35's 0669.** 0669 sends SP `reduce_scatter`, `all_to_all_rows` and
`exchange_into` to sparknet's one-shot when sparknet accepts them, and candidate-36 limits that to
single-slot messages. Larger prefill exchanges still take the PyNCCL path this patch changes. The
measurement on candidate-36, with 0669 in place, shows that path still carries the gain.

## Tests

The patch extends two vLLM test modules:

- `tests/kernels/test_deepseek_v41_tilelang_collectives.py`:
  - The threaded fake fabric gains a switched mode, where every pair is linked.
  - `test_switched_four_ranks_exchange_directly_with_the_same_bits`: four ranks under `direct` send
    to every peer, opposite included, in one group per exchange (the relay uses two). Every part and
    every rank-order sum equals the relay's, bit for bit.
  - `test_every_other_topology_keeps_the_ring_relay`: unset, empty, `ring4`, `switched` and
    `DIRECT` keep the ring relay (the ring fabric refuses the opposite rank).
  - An autouse fixture unsets the variable and clears `ring4_relay`'s cache around every test, so
    the ring tests do not depend on the environment they run in.
- `tests/models/test_deepseek_v41_sp_project_reduce_scatter.py`: `test_row_all_to_all_over_cabled_pairs`
  runs on both topologies. Under `direct`, pairs two apart exchange.

**On GB10** (`scripts/release_tests.py bundle` of the 10 release-gate files that exercise the
exchange, `candidate-36/release-tests-manifest.json`, run by `scripts/lab.py kernel-local` on the
built image with serving stopped):

| Image | Result | The two changed modules |
|---|---|---|
| with the patch | 822 passed, 1 failed, 1 skipped | 158 + 173 passed, including all 68 new or changed cases |
| candidate-36's own two files mounted over | 604 passed, 1 failed, 1 skipped, 218 errors | every test errors: their fixture imports `ring4_relay` |

The one failure is the same in both:
`tests/models/test_deepseek_v4_1_engram.py::test_disk_engram_preparation_refreshes_graph_and_rejects_stale_rows`
(`TypeError: argument of type 'NoneType' is not iterable`). It is candidate-36's own and is in a
file this patch does not touch. Verdicts: `candidate-36/runs/tests-*/`. JUnit: `candidate-36/test-results/`.

## Measurement on candidate-36 (`tp4-switched`)

Four DGX Sparks on one MikroTik CRS804-4DDQ (PFC/ECN, traffic class 106), stock 4 KiB kernel
`7.0.0-1024-nvidia`, driver 580.178.04, 2026-11-01, one window.
- **Image:** `spark-ds41f:candidate-36-direct`, built from the series above, distributed to every node.
- **Stock:** that image with candidate-36's own `pynccl.py` and `collectives.py` mounted over the
  patched ones (blobs `64d16a71`, `b5b9fd46`). Checked in the running containers: `ring4_relay` is
  absent.
- **Direct:** the image as built.

Order: stock, direct, …, direct, stock. Each boot cools every GPU to 50 °C, reads every HCA's RDMA
counters, and runs one suite (about 21 minutes):
- greedy outputs of 64 prompts, 256 tokens each, with top-20 logprobs;
- the engine grid;
- closed loops at 16 and 1 streams (60 s warm-up, 300 s measured);
- cold real-text prefill at 32K, 128K and 256K, three repeats each.

It then reads the counters again. Changes are against the first stock boot. Decode is servebench's
paired per-request decode rate.

| Boot | Prefill 32K / 128K / 256K tok/s | Prefill | c16 paired | c1 paired | Grid (decode geomean) | RDMA errors |
|---|---:|---:|---:|---:|---:|---:|
| stock (reference) | 7,241 / 7,277 / 6,948 | 0 | 0 | 0 | 21.61 ms | 0 |
| stock, repeat | 7,288 / 7,302 / 6,976 | +0.46% | -0.42% | -0.07% | +0.18% | 0 |
| direct | 7,505 / 7,553 / 7,197 | +3.67% | -0.11% | +0.09% | +0.08% | 0 |
| direct, repeat | 7,486 / 7,525 / 7,176 | +3.36% | -0.40% | -0.11% | -0.22% | 0 |

- **Prefill:** two-boot means +3.18% at 32K, +3.43% at 128K and +3.23% at 256K. The arms do not
  overlap (3.36-3.67% against 0-0.46%).
- **Decode:** closed-loop paired decode unchanged. The grid reads -0.16% over boots
  (`servebench.engine.spread`, boot as the sample), ±1.05% at 95%: no detectable change.
- **Outputs:** greedy token ids identical on every boot. Teacher-forced, every top-20 distribution
  is identical to the reference (KL 0, top-1 agreement 100%), from
  `python3 -m servebench compare <stock>/model.json.gz <direct>/model.json.gz`.
- **RDMA:** on every node's switch port, none of the 10 error counters grew on any boot
  (`out_of_sequence`, `packet_seq_err`, `implied_nak_seq_err`, `out_of_buffer`,
  `local_ack_timeout_err`, `rnr_nak_retry_err`, `req_transport_retries_exceeded`, `req_cqe_error`,
  `resp_cqe_error`, `duplicate_request`). Each node's port appears as two PCIe functions
  (`rocep1s0f1`, `roceP2p1s0f1`), which read identical counters; the `f0` functions are not cabled.
- **ECN-marked packets** per boot, summed over the four ports and counting each port once: stock
  49,972 and 43,151, direct 28,479 and 36,083. `hca-counters.json` lists both functions, and
  `metrics.json`'s `hca_ecn_marks` sums both, so it is twice these figures. The switch's own drop
  counters were not read.

Receipts: `candidate-36/runs/<boot>/` holds:
- `suite.json` (suite definition and deployment identity);
- `conditions.json`, `metrics.json`, `grid.json`;
- `c16.json` and `c1.json` (request tables);
- `prefill.json`;
- `hca-counters.json` (each counter before the bench and its growth);
- `arm.json`, the arm the boot ran.

The model steps' text (3.8 MB each) is not kept.

## Earlier measurement on spark-ds41f-1030

The same change on 1030, same site and suite, two boots each. Stock was the `spark-ds41f:1030-direct`
image with 1030's two files (`stock/`) mounted over the patched ones.

| Boot | Arm | Prefill 32K / 128K / 256K tok/s | Prefill | c16 paired | c1 paired | Grid |
|---|---|---:|---:|---:|---:|---:|
| ref | stock | 6,796 / 6,835 / 6,565 | 0 | 0 | 0 | 24.75 ms |
| A/A | stock | 6,813 / 6,851 / 6,581 | +0.23% | -0.20% | +0.01% | +0.19% |
| A | direct | 7,021 / 7,049 / 6,754 | +3.10% | -0.01% | -0.01% | +0.58% |
| B | direct | 6,990 / 6,998 / 6,725 | +2.55% | +0.09% | -0.16% | +0.53% |

Prefill +2.7%. The grid reads +0.46% ±1.05 over boots. Outputs and logprobs are identical. Receipts:
`runs/<run>/`. These arms extend that site's configuration, and the stock arm mounts `stock/` from
the site's experiment path.

## Still open

- The switch's own drop counters (the HCA error counters are covered above).
