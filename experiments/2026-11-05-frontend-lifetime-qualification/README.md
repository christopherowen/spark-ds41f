# Frontend lifetime cleanup qualification

Prepare PR890's two existing request/utility cleanup patches on the corrected,
default-enabled C40 vLLM source. Purpose: correctness and bounded ownership on
failed sends/cancellation, with neutral serving performance acceptable and no
material core regression. This is not a serving-speedup or general memory-leak
attribution claim. The request and utility lifetimes form an explicit frontend
ownership interaction; healthy request results, model/GPU math and policy remain
unchanged.

The two authored patches are copied byte-for-byte with original license/message
and SHA provenance in the canonical replay receipt. Fresh576-patch application
from pinned upstream reproduced C40 first, then applied only those two fixes.
The new vLLM head is99029e6cbdb966ff198dd13902a0825e8d507e27, tree
082bda0f13e94290f462eaf55d143cdb7931f9e5. Only async_llm.py/core_client.py and
their two new test files differ. Changed Python files parse and whitespace passes.

Both disabled TP3/TP4 source profiles pass offline doctor. Experiment-local
release classification retains every global gate plus both CPU regression files
(138 gate files total; selected-source classification has199 files,137 gate and
62 excluded, zero problems). Original inherited lock-metadata failures remain
beside the corrected checks; no implementation was changed to fix that metadata.

The author's existing CPU-only candidate38 container checks reproduce8/10 request
assertion failures and14/20 utility assertion failures in original code; corrected
sources pass10/10 and20/20. Those are authored native results, not independently
repeated here or new serving-image qualification. No GPU/model arithmetic change
is intended, but a new source/image still needs its own full release, functional
and both complete metric boots. Never remeasure an old image/released build.

Compiler/source-base selection remains unfrozen until C42's measured review.
These source profiles currently preserve C40's old compiler and recorded four
images/request; the global67-image default is untouched. If C42 is accepted,
explicitly rebase the packet onto that best validated default-enabled base and
publish/verify its complete identity before any image. Do not silently inject
this cleanup into C42 or another active task's frozen matrix.

Primary acceptance includes exact failed-send/cancellation/late-reply and reused
external-ID behavior, normal results, source/image/rank identity and no material
core regression. Show1030,C36,previous,current with signed changes, both boots,
both reference/candidate repeat margins and every adverse row. One focused rework
on a real miss, then highlight/drop another miss; no speculative retry. No
promotion or bypass of outstanding TP4 gates follows from this source proof.

Official local build preparation exited1 when its temporary canonical fetch ran
out of disk space. The original receipt and compressed log are preserved. Fresh
patch replay and offline source checks remain valid; complete official preparation
and build checks are still required before any image. No image or GPU work ran.

A separate second official vLLM-only preparation subsequently passed and
reproduced the declared head/tree. Its original log retains an intermediate
patch whitespace warning; final source whitespace passed. The disk failure
remains unchanged. Complete Linux inputs/checks and actual-image gates are pending.

C42 subsequently failed its required IKET prepared/frozen check. The VERSION-only
C44 proposal cannot fix that source branch and is withdrawn unmeasured. C43 now
selects C40's unchanged old compiler. Its forthcoming release gate uses an isolated
fresh directory, preserving the shared test cache C42 touched. Serving-rank
caches were never touched by C42 because no image was distributed or booted.
