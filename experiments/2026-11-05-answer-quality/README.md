# Completed-answer quality and useful throughput

Base: `6cc2d6b1`; serving source and immutable C36 image are inherited from the
page-owned selector experiment. Promoted configuration and release evidence are
unchanged. One intended variable: native selection versus the current aggressive
centroid selector. Reasoning may differ. The measured outcome is a completed
answer graded against independent benchmark ground truth, plus total request and
workload time including reasoning and summary maintenance.

The first benchmark is LongBench v2. `sources.lock.json` pins the official code,
zero-shot prompt, pure answer extractor, dataset revision/hash, model tokenizer,
settings and outcome-independent task selection. The original benchmark license
is cached with its source. Raw dataset, generated text and token IDs remain
node-local. Do not score reasoning text or infer a final answer from it.

The protocol keeps temperature 0, seed 0 and native high-effort thinking on both
arms, allows 8192 generated tokens, and requires natural final-answer completion.
Truncation, missing final content and missing/incorrect choices score zero; their
categories remain separately visible. No native answer is used as ground truth.
The pinned official answer extractor runs on final content only, outside timing.
There are no grading calls, comparison passes or host activation copies in the
serving implementation.

The pilot freezes one example per domain and untemplated-token bin (8K–64K and
64K–256K), covering all six domains. Up to four independent document contexts per stratum form the
held-out confirmation set (all remaining independent contexts when a stratum is
sparse, at least one). No document context crosses the split. Selection uses a pinned SHA256 ordering, never model
outcomes. Prompts are complete: no truncation, RAG, extra hints or task rewrites.
Actual server prompt lengths and cached-token usage are recorded separately.
Token counts used for selection omit the fixed chat template overhead.

The pilot measures serial cold requests. Held-out confirmation additionally
measures four concurrent cold requests and their cached replay. Native-A, LSA-A,
LSA-B and native-B use two clean fresh serving boots per arm.
Answer accuracy, paired wins/losses, invalid/truncated responses, wall time,
correct answers per second, reasoning/output counts, cold/cached usage and
per-domain/context results remain separate. Repeated boots do not double the
number of independent quality tasks. A superiority claim needs held-out evidence
and uncertainty; a pilot win alone is not sufficient. No quality loss or promotion
is authorized by a faster aggregate score.

Prepare only pinned CPU-side benchmark inputs before opening a coordinated GPU
window. The sp5–sp7 window uses their native reservations, heartbeat, idle guard,
all-node clean commit sync, named cluster profiles and C37 restoration recipe.
Read the live/shape plan before every apply. Restore once after the full window,
then independently verify all nodes and release all reservations. Record exact
commands and every run/error under `runs/`; unfavorable outcomes are retained.

EvalPlus executable coding tests and fresh RULER task generation are subsequent
checks, not claimed results. They will be independently pinned before use.


Input-preparation errors are retained in `runs/prepare-longbench-01` and `02`:
the first CPU container could not write its output under the image default user;
the retry used the host UID/GID and exposed a sparse short structured-data
stratum (two independent contexts). Before any model output was generated, the
selection rule was revised to keep one pilot and up to four independent held-out
contexts per stratum, with at least one. Repeated questions from the same source
context cannot cross pilot/held-out or inflate independent task counts. No GPU
or serving operation occurred in those preparation attempts; both leases were
released and both CPU containers removed.


The frozen selection is 12 pilot and 45 held-out independent document contexts;
`selection.json` and its SHA256 in the source lock are authoritative before any
model evaluation. Successful CPU-only export is retained in
`runs/prepare-longbench-03`. Input artifacts are SHA256-checked on the serving
controller; the full dataset is not loaded beside the serving engine. The input
preparation leases are released and no CPU containers remain.

Exact pilot command from the published, clean sp5 controller checkout:

```sh
python3 experiments/2026-11-05-answer-quality/run_window.py \
  --out /home/chris/spark-lab/results/answer-quality/pilot-window-01 \
  --cache /home/chris/spark-lab/datasets/longbench-v2/ac35867b9c9436dc81c4c7535fe640ba7cf47877 \
  --split pilot
```
