# External site receipt: tp4-ring on a stock 4 KiB kernel, DeepSeek-V4.1-Flash-1017

A second four-Spark ring ran this repository's TP4 recipe (r6u image + r6v's vLLM 0180 `pynccl.py`) **without the 64 KiB
kernel or the memory saver**, and compared it with the SGLang line it replaced on the same 1017 weights, the same day, with
the same client battery. It is a receipt from another site, not a promotion request: the shape stays unqualified here.

## Site

| | |
|---|---|
| Nodes | 4 x DGX Spark (GB10), DGX OS OTA 7.5.0, kernel `6.17.0-1032-nvidia` (4 KiB pages), driver 580.173.02, Secure Boot on |
| Memory saver | not installed (Secure Boot would need key enrollment at each console) |
| Ring | switchless, **like-port** cabling (p0-p0 / p1-p1 between neighbours), both PCIe functions per cable, one /24 per stripe; per-node RoCE GID index (5/3/3/5) through `roce_gid_index` |
| Weights | DeepSeek-V4.1-Flash-1017 `1bae0e10`, all 49 LFS files sha256-verified; on the head's disk; workers read the snapshot over read-only NFS from the head, except shards 47/48 (the Engram tables, 94.6 GiB each), which are local regular files on every node for the io_uring row reader |
| Image | `vllm-ds41f-kkref:19f2c20ed4d6-r6u` built once with `bin/spark build image --apply` (630 s, lowest MemAvailable 73.7 GiB), image ID `sha256:ef19f052fe4e...`, copied to the other nodes |
| Arm | [`candidate-arm.json`](candidate-arm.json): `config/cluster.json` at `tp4-ring` with the host kernel at 4 KiB / no memory saver, a snapshot+blobs model profile, per-build DSpark curves, and r6v 0180's `pynccl.py` mounted read-only (sha256 `e3990cd5...4bbc`); started with `--unqualified-shape` |
| Sizing | unchanged from the recipe: 1,048,576 context, 16 sequences, 8,192 batched tokens, KV 10.5 GiB |

The mounted `r6v-0180/.../pynccl.py` is site-local and not in this repository; r6v and later images include vLLM [0180](../2026-10-17-r6v/vllm/0180-PyNCCL-row-all-to-all-over-a-four-node-ring-the-far-.patch) (its series builds the same file, sha256 `e3990cd5...4bbc`), so on them the arm needs no such mount.

## Findings worth knowing upstream

1. **r6u alone does not boot on this ring; r6v's 0180 fixes it.** The first boot loaded the weights and prepared 188
   TileLang calls per rank, then NCCL tried the head's neighbour-facing NIC towards the opposite rank (QP INIT->RTR
   timeout 110) in the Engram row exchange - the failure `experiments/2026-10-17-r6v` describes. With 0180 mounted the boot
   was ready in 216.7 s (warm restart 214.1 s).
2. **A stock 4 KiB kernel is enough for TP4 at the recipe's sizing.** MemAvailable stayed at 18-21 GiB per node idle and
   through a 993K-token prefill (head 17.9 GiB before, 19.0 GiB after); no allocation failures, no memory-guard warnings.
3. **Like-port cabling and per-node GID indices work** with the node map's `roce_peer_hcas` / `roce_subnets` /
   `roce_gid_index`; all 16 NIC-function RDMA counters moved during a completion.
4. **Weights over NFS with local Engram shards** load without trouble; only shards 47/48 need to be local for the reader.

## Results

Same client, prompts and protocol on both engines; thinking off unless a task enables it ([`runs/battery.sh`](runs/battery.sh)).
SGLang = MiaAI-Lab DeepSeek-v4.1-Flash-DGX-Sparks production line (cad252b, TP4/EP1, 716,800 context, DSpark 5), the same
1017 weights, measured the same day. spark-ds41f = this recipe as above, two battery runs on one boot.

| Measure | SGLang | spark-ds41f run 1 | spark-ds41f run 2 |
|---|---:|---:|---:|
| idle step, counting / code (ms) | 43.4-44.4 / 46.1-46.9 | 36.7-37.1 / 38.7-39.1 | 36.8-36.9 / 38.9-39.1 |
| v41bench aggregate tok/s, C1 / C4 / C8 / C16 (8-category mean, counting ceiling excluded) | 70.2 / 188.1 / 285.3 / 445.1 | 87.3 / 219.6 / 329.7 / 465.2 | 87.5 / 219.9 / 330.4 / 463.4 |
| salted cold prefill tok/s, ~3.5K / 13K / 54K / 109K tokens (mean of 2) | 3,042 / 4,806 / 4,326 / 3,925 | - | 4,458 / 5,744 / 6,231 / 6,148 |
| salted needle, ~545K tokens | PASS, 2,444 tok/s | PASS, 5,239 tok/s | PASS, 5,225 tok/s |
| salted needle, 992,974 tokens | - (716,800 limit) | - | **PASS**, 4,413 tok/s, 225 s |
| qeval, 75 tasks, greedy | 72/75 | 71/75 | 71/75 |
| vision + tools | 7/7 | 7/7 | 7/7 |

Losses: one reproducible extra qeval miss (`json_count`: 11 for 10); at C16, coding / math / format 1-7% below SGLang in
both runs. The two spark-ds41f runs agree within 0.4% on decode and produce identical qeval text (temperature 0).

Follow-up (2026-10-18): DeepSeek's reference implementation answers `json_count` with 11 as well, and production
matches it; see [2026-10-18-json-count](../2026-10-18-json-count/README.md).

Limits: one boot per engine, no confidence intervals, no soak or agent workload; the battery is this site's own, not
`bin/spark bench` (no fan-floor control installed); run 2's v41bench prefill cells hit the prefix cache and are not used -
the salted sweep is the prefill comparison.

## Files

- `base.json`, `candidate.json`, `candidate-arm.json`, `results.json`, `decision.md`
- `runs/sglang-1017`, `runs/r6u-0180-run1`, `runs/r6u-0180-run2`: idletest, v41bench, needle, qeval (text + JSON),
  vision/tools, the salted prefill sweeps and the 993K needle
- `runs/battery.sh`, `runs/compare.py`, `runs/needle_salted.py`: the client side
