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
arms, allows 65536 generated tokens, and requires natural final-answer completion.
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
  --out /home/chris/spark-lab/results/answer-quality/pilot-window-02 \
  --cache /home/chris/spark-lab/datasets/longbench-v2/ac35867b9c9436dc81c4c7535fe640ba7cf47877/policy-65536-v1 \
  --split pilot \
  --candidate-config experiments/2026-11-05-answer-quality/native-source-fast.json
```


The next predeclared operating-point probe changes only source/reindex candidate
selection back to native (`native-source-fast.json`), keeping dense budget 64
and row capacity 256 unchanged. This tests whether the source selection shared by
later layers accounts for quality differences or query overhead. It will use the
same pilot; the held-out set remains unseen. No favorable result is assumed.
`--candidate-config` names the committed profile. `--controls-from` may reuse
published native receipts only on the same UTC day, for the identical split,
source lock, native recipe, mounted kernels and benchmark clients; otherwise it
refuses and requires fresh controls. Two fresh candidate boots remain required.


Owner budget revision, 2026-11-05: allow substantially more thinking/output tokens
and judge completed answers. The new shared allowance is 65,536 tokens with an
1800-second request timeout; natural completion ends sooner. Every token and all
elapsed time count. A finite common watchdog prevents a non-terminating request
from occupying the shared lab indefinitely. The original 8192-token window is
cancelled through its benchmark client and restored normally; its partial and
completed runs remain historical diagnostics. No comparison mixes its settings
with the larger-budget cohort. The exact same pilot/held-out document IDs and
input hashes are retained; only the allowance and timeout change. Reference-
control guards require fresh matched native runs because the protocol changed.
The new input-cache namespace is `policy-65536-v1`.


The owner-requested cancellation of `pilot-window-01` occurs during the second
LSA boot. Native-A and LSA-A each finish 12 questions at 7/12 correct, with one
truncation each; their wall times are 490.70 and 494.67 seconds. LSA has one
independently correct answer native missed and one loss in the other direction.
The partial repeat reproduces the first three LSA decisions. Native-B is not run.
These are incomplete-cohort diagnostics at the former cap, not a superiority
claim. C37 is restored at 09:58:26 UTC and independently verified on all three
nodes with clean checkouts, no experiment mounts/environment and no reservations.
Every partial/error receipt and the explicit owner budget-revision reason remain
in `runs/pilot-window-01`. The new protocol starts fresh matched controls.


The larger-budget `pilot-window-02` uses the already prepared native-source
variant. Relative to its matched native controls, its intended variable is dense
centroid selection plus required dense-summary maintenance; source/reindex
selection remains native. Budget 65,536 is shared by both arms. The prior
aggressive, lower-budget cohort is not a performance or quality comparator for
this window. This is a new pilot operating point, not held-out confirmation.

HumanEvalPlus preparation is separate from inference. `evalplus.lock.json` pins
v0.1.10, the official checker source archive and all 164 tasks before any coding
model output. `code_quality.py prepare` verifies the dataset/source hashes and
exports the standard instruction prompts; `grade` uses the official base and
plus executable checks on a single completed final program. One Python code
fence or a plain script is accepted; ambiguous multiple blocks remain unparsed.
No program is recovered from reasoning. Run grading only in a CPU container with
no GPU, no network, a read-only root/source/inputs, a temporary work directory,
limited CPU/memory/processes and host UID/GID. Canonical solutions provide the
independent expected outputs. Short coding prompts are a regression check and do
not demonstrate long-context LSA quality gains. No coding model result is claimed.

CPU-only checker preparation succeeds on sp8 in `runs/prepare-evalplus-01`:
the official canonical HumanEval/0 solution passes all 7 base and 999 additional
tests, and a deliberately wrong program fails both groups. The container has no
GPU or network access; it is removed and the preparation reservation released.
Eight local oracle/selection tests pass. This validates the grading path, not
model coding quality.

The next predeclared source-retention probe is `source-1024-fast.json`: dense
budget 64 and row capacity 256 remain fixed, with source retention 1024 buckets
rather than the original aggressive 512. It tests the source-selection operating
point between aggressive pruning and the current native-source pilot. It uses
this same 65,536-token pilot protocol, with two fresh candidate boots; same-day
native controls may be reused only after the full matched window is published,
restored and accepted by the strict reference guard. The old 8192-token cohort is
not a comparator. The 45 held-out documents remain unseen. No outcome is assumed.
The separate mechanical check covers 98,305 visible keys, both page sizes,
9 owners/rows, graph replay, source handoff and selected native score bits.

The source-1024 mechanical check passes on sp8 in
`runs/source-budget-check-01`: both page sizes, 9 rows/owners, visible 98,305-key
pruned domains, native selected-score bits, source handoff/counts, partial pages,
graph replay and recycled pages. All six hot mounted sources remain identical
to the previously fresh-applied C36 patch series (`runs/local-source-1024`).
No source code inside the timed implementation changes. The standalone test
reservation and container are removed. Model quality and request timing for this
new operating point have not been measured.

The complete larger-budget four-boot pilot is published in [results.md](results.md)
and `runs/pilot-window-02`: native 8/12 on both boots, LSA 9/12 on both, all
responses naturally completed. Total request time is 3.6% higher with LSA; useful
correct-answer throughput is 8.56% higher in this pilot. The one paired win has
p=0.5, so general quality superiority is not established. C37 restoration and
independent all-node verification pass. Every run and its conditions are retained.

Next source-retention pilot command, after publishing that restored reference:

```sh
python3 experiments/2026-11-05-answer-quality/run_window.py \
  --out /home/chris/spark-lab/results/answer-quality/pilot-window-03 \
  --cache /home/chris/spark-lab/datasets/longbench-v2/ac35867b9c9436dc81c4c7535fe640ba7cf47877/policy-65536-v1 \
  --split pilot \
  --candidate-config experiments/2026-11-05-answer-quality/source-1024-fast.json \
  --controls-from experiments/2026-11-05-answer-quality/runs/pilot-window-02
```
