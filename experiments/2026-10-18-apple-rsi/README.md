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

## Arms

- `r6n-tp4-dba1.json`: r6n on the ring with the original checkpoint, as it served in
  production. Its arithmetic and chat encoding predate both revisions, so it is the
  comparison the reference outputs were made for: a failure here is ours.
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
