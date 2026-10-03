# Balanced four-node collective policy

The selected experimental policy keeps neighbours direct, divides opposite-rank
relay traffic between both directions, and uses both PCIe roots on each cable.
Across the final 78 timed cases on four ranks, distinct RDMA interface counters
range from **24.805% to 25.175% per interface**, and clockwise shares from
**49.715% to 50.233%**. Payload counters and compiled NCCL channel plans verify
that this is payload distribution, not merely acknowledgment traffic.

This follows the [bidirectional relay](../2026-10-03-ring4-bidirectional/README.md)
and [NCCL direction audit](../2026-10-03-nccl-bidirectional/README.md).
`selected.json` is the best balanced policy tested here. It is unpromoted;
`config/cluster.json` remains unchanged. See [decision.md](decision.md) for the
serving result and remaining limits, and [hardware/README.md](hardware/README.md)
for the complete native evidence, including failed attempts.

## Policy

| Operation | Eligible input | Backend |
| --- | --- | --- |
| All-reduce | Contiguous local CUDA BF16/FP32, positive size divisible by 16 bytes, at most 1 MiB | RoCEnante bidirectional relay |
| All-reduce | Larger or unsupported inputs | NCCL Ring |
| All-gather | Supported contiguous CUDA input, dimension 0 or last, input shard at most 2 MiB | RoCEnante bidirectional relay |
| All-gather | Larger or unsupported inputs | NCCL Ring |
| Reduce-scatter | All tested sizes | NCCL Ring |

The gather backend rejects scalar, boolean, complex and sparse tensors. Its
unaligned rows use padded scratch and a reshape. All-reduce capacity stays
**2 MiB** even though dispatch changes at **1 MiB**. This preserves preparation,
priming and vLLM's **205-token sequence-parallel prefill threshold**. Explicit
prepared operations may use the full registered capacity. The dispatch limit
is validated collectively and included in peer agreement; ABI 10 rejects mixed
implementations even when an older rank zero would ignore the new field.

A source-qualified [successor contract](../2026-10-03-collective-contract/README.md)
corrects the reporting/API and makes the selected collective policy fail-stop.
It is not built or deployed, and does not replace this measured image.

The legacy vLLM startup message in this measured image still reports the registered all-reduce capacity.
Use `B12X_ROCE_ALLREDUCE_DISPATCH_MAX_BYTES` and B12X's `dispatch_max_bytes`
statistics for the dispatch ceiling. The hardware probe asserts that observed
RoCEnante payload counters agree with the actual backend predicate.

NCCL uses four channels, ordered **CW/root0, CCW/root1, CCW/root0, CW/root1**,
a **4 MiB buffer**, a **512-byte allocation floor**, and Ring thread thresholds
`-2 -2 -2 1 1 1`. In mode 2, tiny Ring calls try fewer threads down to 128
before dropping channels. Large calls retain their normal thread count.
Defaults and mode 0 retain the previous behavior. No Linux route change is
required; NCCL communicates with physical neighbours, while RoCEnante relays
opposite-rank fragments.

Payloads below the splitting/alignment granularity can use fewer paths. The
scalar and odd-size fallback tests establish correctness, not equal distribution
of an indivisible payload. All 78 timed cases (128 bytes through 20 MiB per
rank, depending on dtype) use all four interfaces. This does not assert that
every conceivable tiny size activates four channels.

## Why this combination

RoCEnante has lower small-message latency, while NCCL scales better for larger
reductions. At 480 KiB, RoCEnante's exact payload counters show 1,474,560 bytes
sent per rank per call: three input payloads, split equally across four HCAs.
Its gather-and-sum reduction exchanges peer inputs; NCCL's reduce-scatter and
all-gather phases reduce bulk all-reduce traffic. All-gather has its own crossover.

The handoff screen compares NCCL-only and mixed backends over 480 KiB–4 MiB
BF16 and 960 KiB–8 MiB FP32. All-reduce is tied near 1 MiB; at 2 MiB the
NCCL measurements are 235/243 us versus RoCEnante 314/324 us (BF16/FP32).
All-gather favors RoCEnante through 2 MiB and NCCL above it.

Final same-image comparison, BF16, median of the slowest rank in each sample:

| Operation and size | Clockwise control | Balanced selected | Latency change |
| --- | ---: | ---: | ---: |
| All-reduce, 2 MiB | 320 us | 232 us | -27.4% |
| All-gather, 4 MiB | 706 us | 618 us | -12.5% |
| All-gather, 10 MiB | 1,550 us | 1,426 us | -8.0% |
| Reduce-scatter, 10 MiB | 1,571 us | 1,431 us | -8.9% |
| Reduce-scatter, 128 B | 48 us | 65 us | +34.3% |
| Reduce-scatter, 480 KiB | 122 us | 152 us | +24.1% |

All-reduce/all-gather sizes are input bytes per rank. Reduce-scatter sizes are
**output bytes per rank**; its input is four times larger. These are transport
microbenchmarks, not model TPS gains. Small reduce-scatter results vary across
screens: an earlier bracketed 480 KiB comparison was 134 us in both arms, and
the 128-byte penalty was about 6 us. Both repeats are retained; there is no
universal latency improvement claim.

A 256 KiB buffer hurt bulk traffic. Eight channels with a 1 MiB buffer offered
no clear advantage over four channels with 4 MiB and consume more GPU blocks.
Globally forcing 128 threads balanced tiny calls but slowed larger calls; the
adaptive rule avoids that blanket setting. This is a finite tuning screen,
not proof of a global optimum or every buffer/channel combination.

## Reproduction and measurement

The final image is
`vllm-ds41f-kkref:04c30fa98e79-r5o-roce-balanced-dispatch-v1`, with image ID
`sha256:6e03995e36aac90c578fe2d00796068a32ecb8316ed0bae201352d1486272d4d`.
It was built from deployment `6b6861b66b3c2f10da56fadbcf452b705fc31c3c`, GPU
smoke-tested, and copied by digest to all four ranks. Source trees and patches
are pinned in `source-dispatch.json` and `upstreams-dispatch.lock.json`.
The same image serves both `selected-control.json` and `selected.json`.
The control uses clockwise NCCL, four channels, a 1 MiB buffer, and the old
2 MiB/4 MiB handoffs; RoCEnante is bidirectional in both arms.

The shared window was owned by `balanced-policy`, with all four nodes idle on
entry and an agreed idle return. Runs use the repository's published-checkout,
startup-memory, steady-memory, cooling and telemetry guards. Obtain a window
before reproducing; `screen.py` and `run_command.py` require that owned hold.

```sh
TMPDIR=/tmp bin/spark3 --cluster-config experiments/2026-10-03-balanced-policy/selected.json build prepare
TMPDIR=/tmp python3 experiments/2026-10-03-balanced-policy/test-dispatch.py
TMPDIR=/tmp python3 experiments/2026-10-03-balanced-policy/test-relay.py
TMPDIR=/tmp python3 experiments/2026-10-03-balanced-policy/screen.py \
  experiments/2026-10-03-balanced-policy/selected.json new-run \
  --lengths 64 1024 5120 30720 245760 393216 524288 655360 786432 1048576 1572864 2097152 5242880
python3 experiments/2026-10-03-balanced-policy/analyze.py .work/balanced-policy new-run
```

The microbenchmark captures 16 calls per CUDA graph and replays it 16 times
per sample, with five samples per case. Hardware counters are outside GPU-event
timing. Analysis takes the maximum rank latency per sample, then the median;
it does not average away the slow rank. Checks include exact eager data,
changing graph inputs, dispatch/capacity/gather boundaries, unsupported sizes
and dtypes, cancellation-sensitive inputs, and RDMA error-counter deltas.
`channel-plans.py` parses actual NCCL payload partition counts. Physical-port
counters expose duplicate views of the same cable; only the separate RDMA
vport counters are summed as interface traffic.

Serving uses the existing `bin/spark3 bench`, the same client and image, seed 0,
256-token prose/code decode, and a fixed DSpark verification-cost table:
`dspark-costs-8b0313ede293a6de.json`, SHA-256
`1c66058404523da5323fe46c0a467b1448a39a2eaa5d7cfb1f39e5dffa8d3a0e`.
Every boot logs that it reused the table. Actual accepted and verified rows can
still vary with generated text; pinned policy is not fixed realized work.
`workload-identity.py` fingerprints the source corpus and exact prefill prompts.
Source-text size labels are estimates; `results.json` and native reports retain
actual token counts. Unique cache salts keep the source-prefill requests cold.
Prefix replay is a separate filler-text workload with intentional cache reuse.

The serving sequence is control/candidate/control/candidate. The first pair
covers concurrency 1/8 with three decode samples and two prefill repetitions;
the second pair covers 1/2/4/8 with three decode samples and four prefill
repetitions. Prefill/prefix is cooled separately after decode in every arm.
`summarize-serving.py` verifies matching workload identity, topology, image IDs,
clean client state, live doctor and prompt token counts. It reports each
matched pair separately rather than manufacturing a pooled confidence interval
from different boots.

## Validation and boundaries

The source tests cover actual ring construction, channel allocation, adaptive
threads, capacity-independent dispatch, ABI agreement and relay progress with
sanitizers. All 176 repository tests pass. Sixteen transport launches pass on
all four ranks, with zero tracked RDMA errors; the final pair covers 78 cases
per rank. Native records retain the failed ABI-loader build and the client LAN
routing failure before any benchmark request. Patch 0011 fixes the loader's
ABI check; all measured serving runs use the working `dgx1` client hostname.

Ring ordering and backend choice can change floating-point rounding. This work
does not establish deterministic or batch-invariant generation. It also does
not replace full long-context admission, multimodal, end-to-end agent or
sustained thermal qualification. The hotter dgx1/dgx2 pattern remains present.
