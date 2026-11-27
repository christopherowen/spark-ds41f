# External site receipt: spark-ds41f-1126 on tp3-switched and tp4-switched (DeepSeek-V4.1-Flash-1017)

The site of [2026-11-03-site-tp4-1030-1017](../2026-11-03-site-tp4-1030-1017/README.md) moved from 1107 to release
**spark-ds41f-1126** (`890f8654`) in one window on 2026-11-27. It measured three configurations with the same client
battery, on the same day:

1. 1107 on three nodes (the configuration that was serving), as the baseline;
2. 1126 on `tp3-switched`, the same three nodes;
3. 1126 on `tp4-switched`, with the fourth node added.

The site now serves 1126 on tp3-switched, and keeps tp4-switched ready to switch on. This is a receipt from another
site, not a promotion request. Both switched shapes stay unqualified here (`--unqualified-shape`), since upstream
qualified 1126 on the switchless shapes.

**Not stock 1126.** Every measured run carried the site's C1 overlay of `vllm/v1/core/sched/utils.py` (below). The
throughput, prefill and memory numbers are not expected to depend on it. The loop-probe rows do depend on it, and
are 1126 + C1, not stock 1126.

## Site

| | |
|---|---|
| Nodes | 4 x ASUS Ascent GX10 (GB10), Secure Boot on, `host.fan_control: false`, the recipe's 64 KiB host as in the earlier receipt |
| Fabric | one switch, both ConnectX-7 paths per port, two rails, MTU 9000 |
| Weights | DeepSeek-V4.1-Flash-1017 `1bae0e10`, on the head's disk; the other ranks read the snapshot over read-only NFS, except shards 47/48 (the Engram tables), local on every node |
| Image | `spark-ds41f:1126`, `sha256:0fe3e0aea413bd8c2aa37204c80b9d3a3bd69404f480277d562bff6476bf0eb4`, built once with `bin/spark build image --apply` (589 s), then copied to the other nodes; vLLM patch head `d8784b84` |
| Deployment | a site branch over the tag. It adds only this experiment's files and `scripts/memguard.sh` warn-only (a site policy: memguard logs instead of killing the container) |
| tp3-switched | three nodes, the shape's own sizing: 16 sequences, 6,144 batched tokens, 4.91 GiB KV; arm [`candidate-arm-tp3-c1.json`](candidate-arm-tp3-c1.json) |
| tp4-switched | four nodes, 32 sequences, 16.5 GiB KV, site batch budget 8,192 with threshold 7,936; arm [`candidate-arm-tp4-c1.json`](candidate-arm-tp4-c1.json) |
| Serving | one name (`glp` unset), 1,048,576 context, 64 images per prompt, no display carve-out (`Errno 38` on these boards) |
| Baseline | spark-ds41f-1107 on the same three nodes, 1107's tp3 sizing (8 sequences), with the site's DSML parser overlay (this repository's #949, which 1126 carries as `0768`) and the same C1 hunk |

Node names in `runs/` are `node1` (the head, rank 0 in both shapes) to `node4` (the node tp4 adds).

### The C1 overlay

[`overlay/c1-utils.diff`](overlay/c1-utils.diff) is the whole change against 1126's own file (sha256 `1859400d`). It
removes the branch that stops an answer at its first loop once the reasoning has looped. Such an answer is then
judged by the same persistence rules as any other answer. The site added it on 1107 because the stock rule cut
legitimate repetitive output that the model had drafted in its reasoning (a zero-filled file written through a tool
call). 1126's only change to this file, `forced_reasoning_end_grace`, is untouched. The served file has sha256
`49750acf`. The copy here, [`overlay/vllm/v1/core/sched/utils.py`](overlay/vllm/v1/core/sched/utils.py), differs from
it only in one comment line.

## Findings worth knowing upstream

1. **The threshold check refuses 1107's 8,192 / 8,096 pair.** Startup checks that the threshold plus the decode rows
   (203 on tp4-switched) fit the batch budget (`_check_aligned_prefill_chunks`). The site's 1107 arm had carried
   8,192 / 8,096 since 1030, so tp4-switched refused at startup with a `ValueError`. 7,936 starts. A site that lowered
   the budget on an earlier release should lower the threshold with it.
2. **`engram_on_disk` needs the disk space before the first start.**
   - The first start writes the packed Engram tables per shape and rank, under `cache/spark-ds41f/engram-264-v1/`:
     about 66 GB per node for tp3 and 49 GB per node for tp4. The tables of both shapes stay on disk.
   - The container writes as root, so it also fills the root filesystem's reserved blocks. Two nodes reached 100%.
   - At 100%, `bin/spark cluster stop` found no usable temporary directory. The `git checkout` of a rollback then
     left a half-written tree. Every start then refused it, because start refuses a dirty checkout, untracked files
     included.
   - A free-space check for the tables in `doctor`, or before the write, would have caught this.
3. **The 16 sequences of tp3-switched are the C16 gain.** At C16 the 1107 baseline had 8 sequences, so half the
   streams queued: aggregate 275 tok/s and TTFT 2.7 s. 1126 runs all 16 at 430 tok/s and TTFT 0.7 s (+57%).
   - That is a sizing change. Upstream's own experiment is `2026-11-22-tp3-16-seats`.
   - The cost: each of 16 streams is 3-29% slower than each of 8 was, by category, and aggregate C4 is 5.5% lower.
   - The engine's own gain shows in prefill: +6-13% cold, +31% on the ~545K needle and +57% on the ~990K needle.
4. **The DSML fixtures are identical on tp3, and one case differs on tp4.**
   - On tp3-switched, the quoted and thinking fixtures reproduce the token IDs of the site's 2026-11-06 reference
     exactly. `0768` replaces the site's overlay without any change.
   - On tp4-switched, one quoted fixture (`n1_file_of_closers`, whole and streamed) produces different token IDs, and
     the client check fails on that case. That is 6/8 clients and 8/10 fixtures. All other cases are equal.
   - It is one temperature-0 sample per shape. A different reduction order across four ranks moving one greedy token
     would look the same.
5. **Loop probe on 1126 + C1** ([`runs/loop_probe.py`](runs/loop_probe.py), temperature 0):
   - On tp3, the plain answer of 256 zeros at max effort is now cut by the repetition stop. 1107 + C1 completed it.
   - At effort none, 1126 completes the zero-filled file and that answer, which 1107 cut.
   - On tp4, the zero-filled file is cut at both efforts, and the plain answer at both efforts too.
   - These are single samples of the overlay's behaviour, not of stock 1126.
6. **Operations:**
   - A warm switch between the two shapes takes about 3 minutes (stop, then start with `--change-shape`), once both
     shapes' tables and JIT caches are on disk.
   - The first 1126 boot, which writes the tables and compiles, took about 22 minutes.
   - A stop refuses when the running engine's node count differs from the profile's. The site's switch script stops
     with the running shape's arm.

## Results

The same head-side client on every run, thinking off unless a task enables it:
[`runs/trial_battery.sh`](runs/trial_battery.sh). It wraps the battery, salted sweep, ~1M needle, stall probe and
memory sampler of [`../2026-11-03-site-tp4-1030-1017/runs/`](../2026-11-03-site-tp4-1030-1017/runs/), and adds the
loop probe and the DSML checks ([`runs/overlay_check.py`](runs/overlay_check.py),
[`runs/capture_fixtures.py`](runs/capture_fixtures.py)). It is the site's own battery, not `bin/spark bench`.

| Measure | 1107, 3 nodes, 8 seqs | 1126 tp3-switched | 1126 tp4-switched |
|---|---:|---:|---:|
| v41bench aggregate tok/s, C1 / C4 / C8 / C16 | 80.5 / 207.3 / 312.1 / 274.8 | 86.8 / 196.0 / 310.5 / 430.5 | 109.5 / 262.2 / 378.4 / 577.3 |
| v41bench TTFT at C16 (s) | 2.68 | 0.70 | 0.58 |
| v41bench cold prefill tok/s, 2.9K / 11.6K / 46.8K / 93.3K | 4,462 / 6,218 / 6,967 / 6,985 | 4,930 / 6,866 / 7,366 / 7,903 | 5,440 / 8,356 / 9,604 / 9,727 |
| salted needle, ~545K | PASS, 5,867 | PASS, 7,676 | PASS, 9,440 |
| salted needle, ~990K | PASS, 4,709 | PASS, 7,368 | PASS, 9,062 |
| idle decode tok/s after first token, runs A-E | 146.7 / 148.3 / 118.2 / 147.9 / 125.7 | 153.5 / 152.7 / 131.1 / 153.8 / 121.4 | 196.8 / 190.0 / 161.6 / 192.3 / 164.2 |
| short request beside a ~545K prefill: TTFT / max chunk gap | 1.66 s / 1,132 ms | 1.20 s / 912 ms | 1.16 s / 1,087 ms |
| qeval, 75 tasks, greedy | 70/75 | 72/75 | 70/75 |
| vision + tools | 7/7 | 7/7 | 7/7 |
| DSML: clients / agents / quoted fixtures / thinking fixtures | 8/8, 8/10, 10/10, 8/8 | 8/8, 8/10, 10/10, 8/8 | 6/8, 8/10, 8/10, 8/8 |
| lowest MemAvailable through the battery, per node (GiB) | 6.8-9.5 | 6.3-8.4 | 16.5-19.2 |
| RoCE retransmits / out-of-sequence, all rails | 0 / 0 | 0 / 0 | 0 / 0 |
| warm ready (s) | - | 148 | 155 |

- qeval misses `code_interval_intersect`, `code_word_wrap` and `math_m9` in every run. The 70/75 runs also miss
  `json_count` and `json_escape`.
- Both agent failures are `multiturn_n2_max` in every run, whole and streamed.
- The lowest MemAvailable on tp3 is on the head, as on 1107. The full rows are in the two comparisons below.
- Full side-by-side tables, per category: [`runs/compare-tp3-1107-vs-1126.md`](runs/compare-tp3-1107-vs-1126.md) and
  [`runs/compare-1126-tp3-vs-tp4.md`](runs/compare-1126-tp3-vs-tp4.md).

## Limits

- one boot per configuration and one sample per probe at temperature 0;
- the site's own battery, not `bin/spark bench`;
- every 1126 run carries the C1 overlay, and the baseline carries the DSML and C1 overlays;
- no soak and no agent replay.

The receipt carries no private addresses, hostnames or serials. The head address in the arms is a documentation
address, and nodes are numbered.
