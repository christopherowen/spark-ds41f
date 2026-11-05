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

C43 selects C40's unchanged validated compiler. C42 built but failed a required
IKET integration test before serving; that compiler idea is under separate
focused rework and is not a qualified base. No compiler fix is injected into
this frontend cleanup. Source TP3/TP4 profiles remain disabled; separate named
TP3 qualification profiles allow this owner-authorized, unpromoted shared-window
round. Four images/request and the recorded model/draft/topology policy stay
unchanged; the global67-image default is untouched.

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

C42 subsequently failed its required IKET prepared/frozen check; a focused source
correction is pending separately. The VERSION-only C44 proposal cannot fix that
branch and is withdrawn unmeasured. C43 now
selects C40's unchanged old compiler. Its forthcoming release gate uses an isolated
fresh directory, preserving the shared test cache C42 touched. Serving-rank
caches were never touched by C42 because no image was distributed or booted.

## Qualification controls prepared

The new image is unknown until built once on guarded idle DGX4. Its official
release manifest must include all138 files, including both frontend CPU
regressions. The whole gate cache, including pytest temporary directories and
loader fixtures, must start absent at an image-digest-specific path under
`~/.cache/spark-lab/qualification/candidate-43-<digest>`. Preserve C42's shared
cache. Strict actual-image top-k initcheck, source labels and all eight ordered
image/gate exits are required before distribution. A failed release gate keeps
its native verdict and copied XML/JSONL before the coordinator refuses serving.

Both functional boots require all40 ordered script exits to pass independently
of their parent exit. The unchanged full release-tp3, single-stream and sampled
families run on a fresh boot and repeat (six suites/26 ordered steps). Explicit
current stream policy matches the saved full protocol; no lean/full mixing.
The coordinator owns heartbeat/guard checks and restores named C36 once at the
end. Management/telemetry loss retains a hardware-review hold and forbids a new
launch. No concurrent DGX4 GPU work during TP3 timing.

Nine offline control tests, six named profile doctors, both read-only controller
plans and Python3.12 parsing passed. Identity-only normalization confirms both
workload specs match the prior complete protocol; actual prepared top-k bytes
match the intended hash. Original preparation/setup failures stay recorded.
Full Linux preparation/build checks, actual image gates and serving measurements
were pending at the prepared control packet. The sole C43 controller has now
started as recorded below; the actual image and all gate results are pending.

The sole planned command, after publishing and guarded all-node source staging,
is `python3 -u experiments/2026-11-05-frontend-lifetime-qualification/run_round.py --apply`.
Default attempt1 is unused. Native head output will be
`results/private/lab/candidate-43-round/receipt.json`; builder output will be
`results/private/candidate-rounds/candidate-43-image/receipt.json` in the owned
DGX4 checkout. Actual launch/exit receipts must use fresh paths under
`~/tl-logs/frontend-lifetime-qualification/setup/`. Never repeat a completed gate
or measured arm. A safely scheduled passed-gate-only handoff may use the tested
immutable receipt resume in a distinct attempt; a failed or measured arm cannot.

## Current execution

Attempt1 started2026-11-05T22:20:01UTC on DGX1, controller1125694 and
supervisor1125693. Exact clean published execution commit is
2c4b23afcd834ba2adf6a15ca434e59fa4f49f74 on all three serving checkouts and the
owned DGX4 build checkout. Windowdgx1-1793917240 opened22:20:40UTC. Initial
compact check has live heartbeat, both processes alive, image-and-gate running,
no final exit or serving measurements. No duplicate or active checkout sync.

Source staging used its own window, retained both the expected existing-container
plan refusal and the local operator parse error, then used the named sync command.
Independent C36 doctor and unchanged all-rank container IDs passed before closing
without a serving restart. All11 native setup receipts and hashes are in the
attempt directory. Control-packet hashes describe the prepared snapshot at the
execution commit; later status/analysis evidence does not change active nodes.
