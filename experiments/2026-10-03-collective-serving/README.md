# Four-node collective selection in serving

Continue the completed NCCL/relay screen from `5331c96`. Select the simplest
measured fast path for the actual four-node serving workload. No new native
collective implementation is proposed. Production configuration remains the
accepted baseline until a candidate is explicitly promoted.

## Candidates and controls

All arms use the same verified image, target/draft TP4, model arguments,
RoCEnante relay, 2 MiB all-reduce capacity, 4 MiB all-gather shard capacity,
KV allocation, graph shapes, and fresh shared TP4 speculative cost-table path.
The first control boot creates that table; later arms must report reusing the
same file/hash. Existing TP3 cost curves are not reused. Profiling remains
inactive during timings; bounded profiles run separately using vLLM's existing
profiler endpoint.

| Profile | Initialized NCCL channels | Simple buffer | Intended comparison |
| --- | ---: | ---: | --- |
| `cluster-relay-c1.json` | 1 | 1 MiB | Four-node serving control |
| `cluster-relay-c4.json` | 4 | 1 MiB | Channel count only |
| `cluster-relay-c4-buffer4m.json` | 4 | 4 MiB | Buffer size only, if useful |

The topology validator accepts equal native min/max channel settings of 1, 2 or
4 on four nodes. It retains all neighbor, algorithm and reachability constraints.
The default generator still chooses one channel. There is no duplicate custom
channel setting and no production environment change.

Do not change the RoCE size limit while screening NCCL channels. vLLM patch
0018 also uses that limit to select sequence-parallel prefill, so doing both
would confound transport choice with a different model execution plan.

## Completion criteria

1. Refine the all-reduce crossover using the existing bounded GPU probe at
   640/768 KiB, 1/1.25/1.5/1.75/2 MiB BF16 inputs. Retain corresponding FP32
   cells and label actual NCCL fallbacks. Compare relay and automatic NCCL
   with four channels, in both orders, without overlapping host transfers.
   Separately check 5/10 MiB BF16 shards (2,621,440/5,242,880 elements) with
   one and four NCCL channels. These represent 2048/4096-row, 5120-wide TP4
   hidden-state gather/scatter shapes; reduce-scatter inputs are four times
   the shard size. Keep FP32 results separately labelled. The largest graph's
   sixteen FP32 all-gather outputs occupy 1.25 GiB inside the existing 12 GiB
   probe limit. These points are bulk NCCL tests, not a proposed RoCEnante
   capacity increase.
2. Qualify coordinated TP4 model startup with the existing memory guards,
   identical source/image, readable complete checkpoint metadata and live doctor.
   Keep the entry idle state recoverable; do not start the old triangle profile
   on the four-node physical ring.
3. Use the existing bench client for quality, prose/code/JSON c1/c8 decode,
   short/4K/32K/64K prefill and prefix replay. Start with three samples; repeat
   complete arms when variability or a regression prevents a decision. Track
   verification work, acceptance, output lengths, failures and step time as
   well as aggregate TPS. All timed arms share the pinned cost table.
4. Attribute the result with bounded GPU traces of c1/c8 decode and prefill.
   Separate profiling from throughput measurements. Graph-capture counts are
   prepared shapes, not executed-call counts; report that distinction.
5. For the surviving candidate, check c2/c4, long prefill and four-long-context
   admission, memory/swap/error behavior and output-integrity gates. Accept a
   simpler configuration if a more complex one has no repeatable serving win.
6. Record the chosen path, unsupported cases, numerical limits, native receipts
   and a precise next action. Microsecond wins alone do not justify promotion.

A fixed rank-order FP32 reference and NCCL reduction can differ numerically.
These profiles start from production's arithmetic, which is not batch-invariant.
The task does not silently claim that a faster transport makes the model
fully deterministic. A strict-reference deployment requires its own numerical
qualification before adopting a changed reduction path.

## Window discipline

Request the shared four-node window and wait for its current holder to release.
Require idle GPU/NIC state and no checkpoint transfers before taking it. Use
clean published isolated qualification checkouts and the normal coordinated
cluster commands. Record every unsuccessful attempt. After an allocation or
management-path problem, investigate the specific failure before another boot.
Use the repository's fail-closed startup guard and do not bypass it.
