# Apple's RSI-Inference DS4.1 checks on the previous checkpoint

Owner, 2026-10-17: explore everything apple-aiml-research/ml-rsi-inference-bench offers, run
against the previous DeepSeek release, since our changes that follow DeepSeek's current
reference may affect its checks.

The benchmark's `support_deepseek_v41_flash_4xb200_vllm` task grades day-0 vLLM support of
DeepSeek-V4.1-Flash at revision `dba1be0a` on four B200s. Its reference outputs come from
the model author's `inference/` code at that revision, before DeepSeek's 2026-10-12 WO-A
input quantization and the 2026-10-16 encoding fix. It scores 17 checks; the ones that read
only a running server are run here, attached to our deployment:

| Check | What it compares | Its bar |
| --- | --- | --- |
| `test_fidelity_kl` | teacher-forced log-probabilities of 32 held-out sequences (prose, math, code, structured; 512-20,480 tokens) against the author's | per-band mean k3: 0.121 / 0.103 / 0.120 / 0.090 |
| `test_text_history_fidelity` | 4 records at 8,192 and 20,480 tokens: fresh; after an unrelated request; 512 generated tokens' decode log-probabilities against the server's own prefill rescoring | the 20,480 band's bar |
| `test_chat_streaming_and_isolation` | chat, a streamed chat, a second turn, and concurrent teacher-forced readings against the same readings alone, under a running companion stream | each band's bar |
| `test_gsm8k_hidden` | GSM8K, the held-out 200-question slice, vLLM's evaluator, 5-shot, greedy | 174 of 200 |

Beside them: the task's API examples, the public fidelity checker on the four public and the
32 held-out sequences, and its serving client (4,096-token natural prompts, 512 tokens; one
stream and 16), whose numbers have no bar here (the task's bar is a share of a B200
reference).

Beside them, the benchmark's generation judge from its DSpark optimization task
([judge_rsi.py](judge_rsi.py)): each DSpark arm's server generates 32 prompts x 256 tokens,
greedy and sampled (temperature 1, seeded); a server of the same model with speculation off
teacher-forces every arm's tokens. J1 counts greedy tokens more than 0.5 nats below the
target's top-1 (a candidate fails beyond the pristine arm's count by
max(0.005 n, 4 sqrt(c + p + 1))); J2 compares the sampled sets' top-1 calibration
(|z| > 5 fails). Our determinism gates prove greedy outputs unchanged by DSpark only in
the sense that they repeat; this asks whether speculation shifts the sampled distribution,
which DeepSeek's sampling defaults (temperature 1.0 on 1017) now make the common case.

## The sparse-indexer checks

The task also grades the model's two-level sparse indexer as a component: the candidate
source (layer 20) publishes up to 2,048 blocks of 8 positions by block maximum, the newest
block always kept, and each candidate-consuming indexer (layers 24, 28, 32, 36) selects its
own top 512 within that pool. Its checks compare against FP32 mathematics with explicit
allowances (logits: per-row error over the row's largest score, 99th percentile below 0.01
and maximum below 0.02; selections: no excluded position more than 2 x 0.02 x the row's
largest score above the smallest selected one; structure exact), on held-out inputs:

- `two_level` (13 cases): the source's pool (newest kept, a strictly lowest block excluded,
  its size) and the consumers within it, packed and paged; exact ties;
- `packed` (15): unaligned row starts (residues 0/1/5/7), 1-512 rows, 0-32,768 visible
  positions, strided candidate views, empty rows and batches, guarded buffers, and the
  logits operations directly;
- `paged` (2): the fused page layout at page sizes 64 and 128, permuted pages with holes and
  a wider page stride;
- `replay_spec` (11): verification groups of 1-6 rows, accepted drafts with a poisoned
  rejected tail, reordered rows, a reused slot, and CUDA-graph replay with changed inputs.

[bundles/indexer-checks/rsi_indexer_shim.py](bundles/indexer-checks/rsi_indexer_shim.py)
puts serving's kernels behind the names the checks import (the reindex layers'
`IndexerV41Plan(16384, page, candidates=True)` in serving's row chunks, and the source's
`IndexerTopK(width, 2048, 8)`), adapting only argument layout: a pool of block ids becomes
the candidate positions the source publishes; packed keys become index-cache pages
addressed by absolute position; the source's FP32 logits are rounded to BF16, the dtype
our source scores in. `indexer.json` runs the public check, the four families and the
component measurement (times only; its bar is a share of a B200 reference) on r6v (dgx1)
and r6n (dgx2); the kernels are the same in both.

## Arms

- `r6n-tp4-dba1.json`: r6n on the ring with the original checkpoint, as it served in
  production. Its arithmetic and chat encoding predate both revisions, so it is the
  comparison the reference outputs were made for: a failure here is ours.
- `r6n-tp4-dba1-nospec.json`: r6n with speculation off, the judge's pristine arm and judge
  server.
- `r6v-tp4-dba1.json`: production r6v with the original checkpoint. A difference from r6n
  measures what following DeepSeek's current reference costs against an oracle that
  predates it.

## Running

The task's files stay outside this repository (Apple Sample Code License): a local copy of
the task directory goes to `~/spark-lab/apple-rsi-src/` on the head node, and
[prepare.py](prepare.py) builds the attached copy in `~/spark-lab/apple-rsi/` (the served
model name, a dataset directory for the evaluator's own GSM8K download, and a session
server that is the running deployment instead of a launch at the task's B200 recipe).
[run_rsi.py](run_rsi.py) runs every step in a container of the serving image against the
booted arm; `run.json` boots each arm and runs it. Full outputs land in
`results/private/apple-rsi/<run>-<label>/`.
