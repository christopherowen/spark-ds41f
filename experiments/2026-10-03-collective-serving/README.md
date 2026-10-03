# Four-node collective selection in serving

The selected candidate combines the existing **RoCEnante neighbor relay for
small messages with NCCL Ring at four initialized channels for bulk traffic**.
It improves 4K–64K filler prefill by about 23% over the same TP4 service with one
NCCL channel. Decode remains within this screen's variability. No new native
collective, Linux routing policy or production configuration change is needed.
See [decision.md](decision.md) for the exact policy and qualification limits.

## Serving comparison

All arms use the same image and native checkpoint, target/draft TP4, five draft
tokens, verification policy, 48-row graph limit, 4,096-row prefill chunks,
524,288-token configured request limit and 3.5 GiB KV allocation per rank.
Only NCCL channel count changes. Timed clients use `http://dgx1:8000` over the
same Tailscale route. The full commands are retained in the native receipts.

| Workload | One channel | Four channels | Change |
| --- | ---: | ---: | ---: |
| Prose single-stream step | 30.43 ms | 30.62 ms | +0.6% |
| Code single-stream step | 43.71 ms | 43.35 ms | −0.8% |
| JSON single-stream step | 43.21 ms | 42.88 ms | −0.7% |
| Prose eight-stream decode window | 193.1 tok/s | 194.8 tok/s | +0.9% |
| Code eight-stream decode window | 271.5 tok/s | 266.3 tok/s | −1.9% |
| JSON eight-stream decode window | 345.4 tok/s | 340.7 tok/s | −1.4% |
| About 4K filler prefill | 4,358.7 tok/s | 5,339.8 tok/s | +22.5% |
| About 32K filler prefill | 4,463.9 tok/s | 5,499.3 tok/s | +23.2% |
| About 64K filler prefill | 4,458.5 tok/s | 5,473.5 tok/s | +22.8% |
| Prefix replay, cold / warm | 7.103 / 0.261 s | 5.777 / 0.258 s | Warm replay unchanged |

These are three-sample decode screens and two-sample prefill points, not tight
proof of decode equivalence. Read intervals, acceptance, verification work and
actual token counts in the reports. `essay`, `tasks` and `rows` disable reasoning
and force 256 output tokens with distinct prompts per stream. Decode-window TPS
excludes first tokens and the arrival stagger. The nominal 2K/4K/32K/64K filler
sizes produce 2,000 / 3,988 / 31,798 / 63,583 actual prompt tokens. Do not compare
these filler results directly with the historical TP3 **source-text** baseline.

The final one-channel repeat returns 4,470.5 / 4,459.8 tok/s at 32K / 64K,
only +0.15% / +0.03% from the first control. Against this independent control,
the four-channel gains remain +23.0% / +22.7%. The short 4K repeat is noisy
(4,102.8 tok/s, two samples); it does not provide a tight drift bound.
The repeat completes every suite and passes quality 5/5, but the CLI returns
nonzero because its **unchanged-control** eight-stream JSON aggregate drops
3.5% (336.1 to 324.3 tok/s). JSON accepted drafts change from 3.111 to 3.046;
that is a contributing variable, not a complete causal explanation. Single-stream
steps remain within 1.1%. No thermal slowdown, failed requests, foreign load
or changed cost table is observed. Retain the regression flag: the result shows
why three within-boot samples cannot prove that 1–2% decode changes come from
the transport. Do not pool boots or suppress an unfavorable control.

A separate fixed 16,321-token **source** prefill profile attributes the gain:
the same 284 NCCL calls consume 1,333.54 ms with one channel and 642.01 ms with
four on rank 0 (summed GPU kernel durations). Full trace spans are 4,056.75 and
3,436.30 ms. Large launches actually have grid size four; the small broadcast
still has grid size one. Generic NCCL kernel symbol names contain `LL` even
where runtime plans use Simple, so the symbol is not protocol evidence. The
microbenchmarks retain NCCL's actual plan logs. Profiles were collected outside
timed throughput runs; summed durations can include waits and overlap.

## Channel and size screen

All eleven new collective launches pass on every rank, including BF16/FP32,
changing CUDA-graph inputs and numerical diagnostics. Every ring stays on
neighbor edges, channel counts match the requested configuration, and tracked
retransmissions, sequence errors and receive-buffer drops are zero. Runs 06/07
have invalid fan-policy timing and are excluded from performance selection;
runs 08/09 repeat them after the cooling helper fix.

Median time of the slowest rank, BF16, microseconds per call:

| Operation and size | 1 channel | 4 channels | 4 channels, 4 MiB buffer | 8 channels | 16 channels |
| --- | ---: | ---: | ---: | ---: | ---: |
| All-reduce, 5 MiB input | 855.5 | 429.9 | 423.0 | 402.1 | 430.1 |
| All-reduce, 10 MiB input | 1,590.1 | 817.7 | 779.9 | 758.6 | 765.3 |
| All-gather, 5 MiB input shard | 1,667.3 | 843.8 | 800.9 | 797.8 | 841.2 |
| All-gather, 10 MiB input shard | 3,255.6 | 1,559.3 | 1,523.8 | 1,454.5 | 1,484.1 |
| Reduce-scatter, 5 MiB output shard | 1,772.6 | 851.5 | 802.7 | 776.5 | 869.4 |
| Reduce-scatter, 10 MiB output shard | 3,455.3 | 1,584.8 | 1,537.7 | 1,453.9 | 1,465.9 |

All-gather output and reduce-scatter input are **four times the shard size**.
The 5/10 MiB shards cover 2,048/4,096-row, 5,120-wide BF16 TP4 hidden-state
traffic. The raw runs also retain FP32 cells; they are not mixed into this table.
Each median uses five samples, each containing 256 graph calls.

The repeated all-reduce crossover lies near 1–1.25 MiB: custom relay is faster
at 640/768 KiB, comparable around 1 MiB and slower above it. All-gather has a
different crossover. The 2 MiB custom limit also controls vLLM's SP-prefill
transition; changing it during this screen would confound model scheduling
with transport. It remains unchanged, as does the 4 MiB all-gather shard limit.

Eight channels' microbenchmark advantage shrinks to 0.7–1.2% over four in
4K–64K serving prefill, with intervals spanning no gain. That arm's full screen
also aborted before prefix testing at the thermal guard, so it is not selected.
Sixteen channels do not improve the bulk microbenchmarks over eight. The 4 MiB
buffer was not screened in serving; a small microbenchmark win is insufficient
to select it.

## Reproducibility

- Image: `vllm-ds41f-kkref:04c30fa98e79-r5o-roce-mesh4-fourpaths-v1`,
  `sha256:9671c96903dc7cc5a9bbae4249e7b7aa9efa550980f1dab5fa142dac3ebe842d` on all four ranks.
  The image supports forwarding experiments, but these profiles select CPU
  relay and install no forwarding rules.
- vLLM tree `c108cd6d1fe8e2d3b91c065feefe818742159020`, B12X tree
  `2077a53bfb2a993b385f1a1bd9799b0a89694976`, NCCL tree
  `47687d2a75b06fdff1b752dbf08bb87f12ca98bb`.
- Kernel `7.0.0-1019-nvidia-64k`, NVIDIA `580.178.04`, hairpin queue size 1024;
  unchanged across arms. No clock, power, driver, firmware or persistent fan-policy change.
- First serving control creates `dspark-costs-8b0313ede293a6de.json` in the
  experiment-specific shared cache directory. Later boots log reuse; retained
  SHA256 is `1c66058404523da5323fe46c0a467b1448a39a2eaa5d7cfb1f39e5dffa8d3a0e`.
  This controls startup curve noise, not batch-dependent numerical differences.
- Initial c1/c4 serving uses deployment commit `edd91f9`; c8 and repeated c1
  use `f3e0e0a`. Changes between them are the fan-restoration helper, probe
  arms, experiment utilities and channel validation. The serving image and
  pinned costs are identical. Benchmark identities confirm clean clients.
- All 48 checkpoint shards pass matching metadata/header/extent checks on all
  ranks. Payload hashes were **not** recomputed; this is not full payload
  equality verification.
- Shared deployment checkouts stay at `cfa6e5c`; isolated qualification checkouts
  receive published commits through the normal coordinated CLI.

## Evidence and failures

`hardware/runs.tar.gz` preserves native commands, logs, benchmark reports,
profiles' summaries and raw-trace paths/hashes, all failed attempts, host/network
receipts and window restoration. `hardware/sha256.json` hashes the archive and
its members. `results.json` keeps boots separate; `hardware/collectives.json`
retains per-rank correctness, actual NCCL plans, port traffic and every timing
sample. Raw profiler traces remain at their recorded host paths; they are not
embedded in Git. `summarize_serving.py` reproduces the serving summary from an
extracted archive.

A repeated-control start also stopped at preflight because the previous exited
containers still existed. The next attempt used the normal coordinated
`--replace` option; no workers launched in the rejected attempt.

The first c1 bench was blocked by failed fan-service restoration; the next
attempt failed on this Mac's LAN route, before measurement. Switching the
client to `http://dgx1:8000` solved the latter without any server/network change.
The fan issue was a repeat-cooling bug: systemd's start-rate limit was exhausted,
and the old helper proceeded after reporting failed restoration. The fix resets
the intentional restart budget, verifies the service is active and aborts on
failure. All affected performance runs are retained and repeated. Four tests
cover failure and recovery; no fan service or curve was changed persistently.

The c4 extended test stopped before admission after dgx2 reached a reported
2 C GPU thermal margin and accumulated 120 ms thermal slowdown. The separate
cooled four-request admission passed. The c8 screen stopped before prefix
replay at a reported 4 C margin on dgx1, with no accumulated slowdown. These
are failed complete screens. Do not hide them or use their long-prefill timing
as a sustained-throughput qualification. Physical cooling differences remain
unresolved; the guard is retained.

The first profile collector lacked permissions on root-owned output. Raw
traces remained intact; the subsequent c1-c8 trace was separated from c1-decode
by timestamp during recovery. The earlier profiler text summary was overwritten,
but the original raw c1-decode trace is retained with SHA256. Later collection
moves one trace per rank and checks the expected count. An initial c8 boot
receipt assertion incorrectly expected rank-zero cost-table logging on every
rank; inspection corrected that local assertion without a serving restart.

## Qualification status

Four-channel quality 5/5, c1/c8 decode, short/4K/32K/64K filler prefill, prefix
replay, cooled four-request admission and the synthetic tool workflow pass.
The extended screen contains c2/c4 decode but aborts later on thermal headroom.
At least 29.15 GiB host memory remains under the admission load; no swap growth,
preemption or request failure occurs in the passing arms. This does not qualify
full 524K contexts, image inputs, batch invariance or sustained long-source
prefill on the new topology. Promotion requires owner acceptance and those
remaining deployment checks; the selected optimization itself is measured.

## Completion and restoration

The synthetic workflow also passes 3/3 on the repeated control, using the same
workload hash as the four-channel candidate. Repository checks pass all 176
tests, candidate doctor and `git diff --check`. CI prepares the promoted and
experimental patch stacks afresh and runs native proxy sanitizer checks. No
new image was built for channel tuning; complete image rebuild inputs are not
present locally.

The window ends at **2026-10-03 01:52:17 UTC**. The coordinated CLI stops and
removes only the experiment's serving containers and memory guards. All four
nodes return to the idle entry state, original Created containers are preserved,
fan services are active, and shared checkouts still match the entry revision
and status. Final NIC counters show no new retransmissions, packet sequence
errors or out-of-sequence events since the final bulk probe. The hold is removed;
there is no pending GPU task from this experiment. The historical TP3 production
profile is not started on the physical four-node ring.
