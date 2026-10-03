# Matched three- versus four-Spark serving comparison

Run the published TP3 baseline's exact `prose`/`code` prompts, reasoning mode,
256-token decode at concurrency 1/2/4/8, seed 0, three samples, source-text
prefill at nominal 1024/32768/65536/262144 tokens with two repeats, and 32K
prefix replay on the selected TP4 relay/four-channel NCCL profile. Quality is
checked first. Keep the same client host (dgx1), Python source corpus and request
sequence; compare actual input tokens against the historical samples. The earlier
portable/filler TP4 screen is not used for this comparison.

The physical ring has no dgx1–dgx3 cable. A fresh TP3 triangle needs recabling;
using a fourth forwarding host would change the transport and is not a fair
reproduction of the old triangle. Until the owner chooses recabling, report TP4
against the stored TP3 baseline explicitly as a historical-control comparison.
This compares useful deployment configurations, including their transport and
TP-dependent verification plans; it does not isolate GPU count alone. Do not
reuse a TP3 speculative cost table on TP4.

The serving image and existing TP4 cost table remain unchanged. Use published
isolated checkouts, normal coordinated start/stop, all-rank memory guards and
the owned hold file. Begin cooled, retain the thermal guard and every failure.
If long prefill reaches the guard, preserve the incomplete report and do not
claim sustained throughput at the unqualified sizes. Restore the idle entry
state and release the hold after the run.

Also audit padding against the exact served vLLM/B12X sources. Distinguish
TP3 divisibility padding from quantization/kernel alignment, graph capacity and
sequence-parallel row padding. Do not remove shared TP3 support while measuring.
