# Collective policy decision

Select the RoCEnante neighbor relay for small collectives and NCCL Ring with
four initialized channels for the larger transfers. Retain the 1 MiB Simple
buffer, automatic LL/Simple selection, 2 MiB custom all-reduce capacity and
4 MiB custom all-gather shard capacity. This is a measured TP4 candidate;
production configuration and the accepted TP3 baseline are unchanged.

The four-channel profile improves the actual 4K–64K filler-prefill screen by
about 23% against the same four-node service with one NCCL channel. It does not
establish a decode improvement. The fixed 16,321-token source-prefill trace has
the same 284 NCCL calls: rank 0's summed NCCL kernel duration falls from
1,333.54 to 642.01 ms. The trace span falls from 4,056.75 to 3,436.30 ms.
These are profiled GPU durations, not API throughput, and sums include waits
and possible overlap. Separate unprofiled benchmarks establish the speed gain.

Eight channels reduce the 5/10 MiB collective microbenchmark times by another
6–9%, but the serving prefill improvement over four channels is only 0.7–1.2%
at 4K–64K, with intervals spanning no gain. The eight-channel screen also stops
before prefix replay at the thermal guard (dgx1: 83 C, reported margin 4 C;
no accumulated GPU thermal slowdown). Keep four channels, which passed that
screen and uses fewer communication CTAs. Sixteen channels do not improve the
bulk probe over eight. A 4 MiB buffer produces only a 2–6% bulk-probe gain over
the 1 MiB buffer at four channels; it was not screened in serving and is not
selected. These are bounded choices, not a claim of a global optimum.

A final one-channel repeat reproduces the original 32K/64K prefill within
0.15%/0.03%, so the four-channel gain remains about 23%. The same repeat flags
an eight-stream JSON regression of 3.5% despite unchanged serving settings and
pinned costs. All requests and quality checks pass; the CLI's nonzero result is
retained. This bounds the confidence in small decode differences and supports
selecting on the much larger, repeatable prefill gain.

## Policy and scope

| Traffic | Selected path | Evidence and boundary |
| --- | --- | --- |
| Small decode all-reduce | RoCEnante ring4, fixed neighbor routes over two PCIe roots | About 17–19 us at 10 KiB versus 75–86 us for tested NCCL arms; about 91–94 us at the 480 KiB maximum captured decode shape versus 143–157 us for four-channel NCCL. |
| Custom-range all-gather | Existing RoCEnante all-gather | Keep its separately defined 4 MiB **input shard** limit; the output is four times that on TP4. Do not reuse the all-reduce crossover. |
| Reduce-scatter and larger all-reduce/all-gather | NCCL Ring, four initialized channels | Full prefill shapes and actual serving traces show a substantial gain. NCCL may use fewer channels per call; broadcasts in the trace still use one. |
| Control and management | Existing TCP/Gloo paths | No routing, firewall or interface-policy change. |

The all-reduce crossover is near 1–1.25 MiB, below the current 2 MiB custom
capacity. This is not yet a reason to lower that environment variable:
vLLM patch 0018 also uses it to select sequence-parallel prefill. Reducing it
would change model execution plans as well as transport. A future transport
cutoff experiment must first separate these two decisions and retain the
same SP behavior; it is unnecessary for the demonstrated channel-count gain.

Both physical PCIe-root interfaces share outgoing bulk data evenly. The current
NCCL ring sends payload toward one neighbor and receives from the other;
reverse-direction traffic on those ports is mostly acknowledgements. The large
max/min **transmit-only** ratio across all four ports is therefore expected,
not evidence that one of the two transmitting rails is idle. More channels
increase injection parallelism, not cable capacity. All completed probes use
legal neighbor edges and report zero tracked retransmissions, packet sequence
errors, out-of-sequence events and receive-buffer drops.

## Qualification limits

The selected profile passed the 5/5 LRU gate, c1/c8 prose/code/JSON screen,
prefix replay and four concurrent approximately 64K prompts with 2,048-token
replies. The latter used actual inputs of 63,595–63,596 tokens, had all four
requests active, no preemption, about 13% peak KV use, no swap growth and at
least 29.15 GiB MemAvailable. Its synthetic three-turn read-only tool workflow
passed 3/3; this is a narrow integrity check, not tool-eval-bench.

The extended c2/c4 plus approximately 128K **source** prefill screen stopped
at the thermal guard after dgx2 accumulated 120 ms of GPU thermal slowdown.
The guard worked. Its two prefill timings are retained as failed-run evidence,
not accepted sustained-throughput results. dgx1/dgx2 were much hotter than
dgx3/dgx4 despite identical fan software; the physical cooling cause is not
established. Separate cooled admission passed without throttling. A long-load
thermal check is required before promotion; raising limits or suppressing the
guard is not the remedy.

No full 524K-per-request capacity, multimodal, long-duration soak or
batch-invariance qualification is claimed for TP4. The image retains production's
atomic model reductions. Exact representable transport checks pass, while the
cancellation-sensitive diagnostic distinguishes NCCL BF16 arithmetic from the
custom fixed-rank FP32 reference. A strict numerical reference needs its own
qualification before changing reduction paths.

Do not replace the relay with NCCL for decode: tested NCCL protocol, channel
and thread settings did not close the small-message gap. Do not adopt the
NIC-forwarding candidate: prior large-message tests showed retransmissions and
poorer latency. No new native collective is needed to obtain the measured gain.

The experiment is complete as a transport selection. The cluster was returned
to its idle entry state and the shared window released at 01:52:17 UTC on
2026-10-03. Promotion and the remaining deployment qualifications are separate.
