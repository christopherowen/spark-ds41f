# CopyDrafter layout divergence: source review

Read-only review, 2026-11-03. This records the path from rank-local host limits
to unequal collective lengths. It does not identify the original first NaN
instruction or claim the original receive-buffer tail has been captured.
No runtime was changed or GPU work performed for this review.

## Exact sources

All line references below are from `git show` of these commits, not the current
contents of prepared worktrees:

- vLLM candidate-37: `5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575`,
  tree `af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33`.
- Sparknet candidate-37: `1fd6a1fbd8dc4884057046d1ebd93d585c9e4bcf`,
  tree prefix `8cc28a5c`. The older `ed6af327` revision is not the candidate's
  source pin. `_oneshot_tilelang.py` is identical between these two revisions;
  the changed device helpers add exchange/reduce-scatter operations without
  changing ordinary all-reduce's sequence-only receive flags.

| Source path | SHA-256 of exact file |
| --- | --- |
| `vllm/v1/worker/gpu/spec_decode/copy_drafts.py` | `caed58e8c641cae78626fb8dab34758573bac407fc092915691bd9ac6e9e1cc8` |
| `vllm/v1/worker/gpu/spec_decode/tilelang_copy.py` | `89cd4cf39ad75b8992504afd5bc7d91a088dda0c64a66d314126553a4379187d` |
| `vllm/v1/worker/gpu/async_utils.py` | `427e4cef724734ddca163185f69ec7f42e9a89900905e3199907c9f64c3d7b8b` |
| `vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` | `851eebdde10371bcf4c3d2c5a0d86a68ee917c70cd6d86f12e8386164eb11295` |
| `vllm/v1/worker/gpu/model_runner.py` | `55841d2d94c4540f05421a4d8a6d1d25cbbd6eafe0d439fd81e94eafdb94c8cc` |
| `vllm/v1/worker/gpu/dp_utils.py` | `9d6e84041464723de303605ad332a5de2fe080dd8f0791cca53664579be65ddc` |
| `vllm/model_executor/layers/vocab_parallel_embedding.py` | `42fc00ad520e9c0c65a701f4b84377d9e5143311e962ac585be4045d5b4d510d` |
| `vllm/models/deepseek_v4_1/nvidia/model.py` | `45b630b279e4ef2657e651cfb4d30322d2b98cc1b72c8462efe4644ed37c5fe9` |
| `vllm/v1/executor/multiproc_executor.py` | `d5eb18335c91c8a3ebd00167f069d49a802c62c186613ac39e3dac7c3f085567` |
| `sparknet/oneshot/_device.py` | `fcb626dd8a202a708a6bba3eb0e828a89eb6681d26bfd3fa68f9bc073e461eb7` |
| `sparknet/oneshot/_oneshot_tilelang.py` | `33e147fefb052d7e7f6ec1a705b99f550e80afd247293758c820aeca19a7fbde` |
| `sparknet/oneshot/_proxy.py` | `7468fe4b3d8b9ed83615b4c3bbd3e0eb2e1c07de8244bfd0c8878595f0fb1968` |
| `sparknet/oneshot/_roce_proxy.c` | `256c711c72bc15fd51ed5177edb777742d2123398221e64790d5cfe5a34083f7` |
| `sparknet/oneshot/runtime.py` | `d87e4d0f0596b038c7d749309c70b40e4e6df5cbae275a58fac1bee4d4eef8ad` |

## The source of different host limits

`copy_drafts.py:80–107` creates a three-slot allocation/staging/event ring but
one shared GPU limit tensor and one published host mirror. The host mirror is
initialized to the full lookahead, 16 in the failing configuration. Request
addition/reset also publishes the full lookahead (`113–138`). Thus an observed
16 is not proof that a 16-token copy proposal was produced: it can be the
initial/reset value or a previously completed value.

At `prepare_alloc` (`154–196`), each rank matches its histories then runs
`copy_alloc`. The integer kernel at `tilelang_copy.py:307–332` computes a source,
accepted-proposal cap growth and offered count; it writes `block` when no copy
source exists. A five-token DSpark block therefore naturally produces limit 5.
An available sufficiently long copy can produce 16. These computations are
local, with no cross-rank broadcast of the limits. Equal input history, state,
verified counts, cut and sampled counts should produce equal limits; those
device inputs and per-epoch results were not captured in the failing run.

`model_runner.py:3643–3655` immediately queues their D2H transfer before DSpark
executes. `async_utils.py:238–246` makes the copy stream wait for the main
stream, copies the GPU limits to that step's pinned staging buffer, and records
an event. `copy_drafts.py:220–221` stores only the latest staging/event pair.
`sync_slot_draft_limits` (`113–120`) publishes that staging buffer only if the
rank's local `event.query()` succeeds; otherwise it returns the old host mirror.
`prepare_alloc:166` and budget selection both call this nonblocking publisher.
An unready event can be replaced by the next pair without publishing its epoch.

There is no requirement that two ranks publish the same completed epoch. Even
if every GPU proposal is identical across ranks and every D2H transfer is free
from races, different local completion/read timing can select different host
values. Matching cumulative read/stale counts do not identify the epochs or
values used. In contrast, planned confidence buffers explicitly synchronize a
fixed previous event and rotate slots (`adaptive_verification.py:218–226`),
with TP agreement of confidence data handled separately (`253–274`).

There is also a **separate, unproven device-race concern**: the D2H source is
the singleton limit tensor, and the next main-stream `copy_alloc` or staged
reset writes it. The methods above do not insert a reverse copy-stream-to-main
dependency before reuse. The output thread does synchronize the allocation
event (`async_utils.py:277–278`; asynchronous delivery at
`multiproc_executor.py:1189–1215`), so establishing actual overlap requires the
whole pipeline's ordering, not just these functions. The observed host values
do not prove a GPU read/write race. PyTorch CSAN visibility tests concern this
second question; they cannot detect incorrect host epoch selection.

## How limits become 17 versus 6 target rows

`model_runner.py:833–837` passes the limit callback into adaptive verification.
`adaptive_verification.py:306–312` clamps the common scheduled draft count by
the local mirror. It selects a width of 5 or 16 (`313–318`), limits the budget
by eligible rows (`324–328`), and returns non-draft rows plus the selected budget
(`370–375`). `compact_batch` applies that budget to request row counts.
For the observed single request with 16 scheduled drafts, eligible/budget 16
gives 17 rows and eligible/budget 5 gives 6 rows.

`model_runner.py:1738–1746` records that count; `2941–2945,2979–2983` uses it
for graph dispatch. `gpu/dp_utils.py:286–295` dispatches the local count and
returns directly for DP=1; it does not reconcile TP ranks. CPU query offsets
are then built from compacted counts (`model_runner.py:1834–1852`). The layout
record is host evidence; it does not capture the final GPU offsets.

The independent repeat receipt in `layout-repeat-evidence.json` records one
divergence among 6,373 consecutive per-rank records: step 6,372 has
17/6/17 tokens and selected padded counts, 16 common scheduled drafts, and
eligible/budget 16/5/16. It establishes unequal host layouts in the last
preparation record before termination. It does not contain the limit
publication epoch, GPU limit values, or first nonfinite activation.

The recorder is before launch, not an execution acknowledgement. In the
layout-ring overlay, the record is written at `1904–1910`, before query-offset
transfer (`1922`) and GPU reallocation (`1931` onward). FULL graph replay is
later (`3334`; piecewise `3379`, eager `3384`). It records the selected padded
extent but not `cg_mode`. Earlier asynchronous GPU work may fail before this
prepared forward is enqueued. Thus neither the original failing GPU step nor
execution of the selected unequal graphs is established by the host record.

## First target collective and the short-payload consequence

In the target decode path, `nvidia/model.py:802–829` reaches the embedding
before transformer layers (`embed_input_ids:732–733`). The TP embedding calls
all-reduce at `vocab_parallel_embedding.py:727–766`. A 5120-wide BF16 output
therefore requests 174,080 bytes for 17 rows and 61,440 bytes for 6 rows.

At the exact Sparknet pin, `runtime.py:1021–1041` derives byte count solely from
the local tensor. `_oneshot_tilelang.py:130–142` stages that local extent and
rings its doorbell. `_roce_proxy.c:1066–1069` reads the local per-slot size;
`949–955` validates only nonzero/alignment/capacity. `759–825,844–860` posts
the sender's payload stripes, each followed by a four-byte sequence flag.
There is no length in that peer-visible flag and no per-operation rank-size
agreement in this path.

`_device.py:265–277` waits for the peer sequence. Once it arrives,
`_oneshot_tilelang.py:165–170` reduces the receiver's local extent;
`_device.py:350–357` reads each peer at those offsets without checking that
the peer sent that many bytes. A 17-row receiver can therefore consume the
old 11-row tail of a 6-row sender's receive slot after valid current flags.
These are allocated, previously initialized bytes: ordinary memcheck/initcheck
success does not establish that they belong to the current collective.

This proves an invalid collective input contract and explains a concrete
corruption route. It does **not** prove that finite current inputs must produce
NaNs: the old tail may contain finite values, nonfinite values, or bytes from
an earlier different dtype. In particular, a legal old int64 -1 has BF16 NaN
bit patterns, but that is a controlled mechanism test, not the observed
production tail. A first-fault payload/tensor capture is still needed to name
the original first NaN producer. The later `route_partials` reduction is local
split-K gate work, not the TP collective itself.

## Review of the shared-budget fix

Reviewed the separate worktree's
`experiments/2026-11-03-copy-drafter-shared-budget` patches and minimal overlay.
Patch 1 SHA-256 is
`3be2b4a7cf42cd31fe82a8b6305a08ed21b3ec8c46312b909e1122747705733c`;
patch 2 is
`a652d61f543137511d2158496ffab47e9112f59e9d78f71c0348f6b37a5e6053`.
The minimal overlay runner SHA-256 is
`a4f2271aa3215db9bfd61f84454175768eb19d76a1b1bb658ac627741d62362b`.
Its exact diff against candidate-37 matches patch 1's only runtime change:
the callback at `model_runner.py:833–837` becomes `slot_draft_limits=None`.

**Recommendation:** patch 1 is the narrow fix for the diagnosed rank-layout
defect. It selects the manager's existing scheduled-drafts branch, removing
rank-local completion timing from its budget inputs. It adds neither a host
wait nor a device length clamp. This is stronger than making an invalid
cross-rank layout's suffix harmless on one rank. The existing confidence
synchronization remains; this is not a claim that the whole runner has no
host waits.

Rank agreement still relies on matching scheduler request/count order, slot
mapping, confidence epoch/values, recipe constants and manager state. The
confidence mechanism already synchronizes a defined previous copy and either
broadcasts or uses its configured rank-invariant path. Removing the callback
does not repair an independent violation of those contracts, but it removes
the specifically observed 16-versus-5 input to graph selection.

The extra rows have an existing sentinel contract, not an added acceptance
shortcut. `tilelang_copy.py:245–274` writes -1 and zero current confidence
through the proposal's unused suffix. The actual candidate-37 verifiers reject
that sentinel:

- Greedy: `sample/tilelang_sampling.py:3077–3089` requires a nonnegative draft
  and matching target token; the first mismatch stores the target token.
- Block: `spec_decode/tilelang_rejection.py:624–652,759–802` reads bounded
  addresses, requires a valid draft, and stops at the placeholder. Resampling
  treats it as the end of the block (`1017–1029`).
- Standard/synthetic and the fallback block path:
  `spec_decode/rejection_sampler_utils.py:585–618` masks negative tokens before
  acceptance, including synthetic acceptance. The rejected placeholder uses
  target logits directly during resampling (`859–876`). Synthetic acceptance
  is a synthetic benchmark mode, not an inference-exactness claim.

The first placeholder is judged using the preceding valid-prefix input row's
logits, so that decision does not require logits conditioned on the placeholder
itself. With dead rows enabled, zero confidence makes the suffix padding via
`tilelang_inputs.py:1499–1522`, retaining the required bonus row. Without dead
rows, extra suffix rows execute through the target; finite/safe suffix execution
and causal isolation remain requirements. History updates append only sampled
tokens (`tilelang_inputs.py:1723–1731`) and move the computed frontier by query
length minus rejected rows (`1752–1754`); draft preparation similarly removes
rejected target suffix rows (`2306–2332,2581–2608`). No concrete consumer
violating this contract was identified in this review. This is not a new
end-to-end proof of every attention/compressor/cache path.

Patch 1's two simulated-rank test exercises actual budget/reallocation kernels;
its growth/shrink/slot-reuse test exercises real combine and verification with a
`ToyTarget`, not the actual target forward. Those are useful tests of the changed
contract, with that limitation. The isolated exact-source CPU fixture also
reproduced 16→17 and 5→6 rows, and None→17. Full-model repro and performance
receipts remain separate evidence; neither speed nor bit-identical outputs for
all workloads follow from this source review. Extra verification rows may cost
work even though the patch adds no synchronization.

Patch 2 removes the now-unused limit buffers, queries and D2H transfer, while
preserving allocation delivery to the scheduler. It also removes the separate
singleton-buffer race concern from this path by removing the buffer entirely.
It is a sensible follow-up cleanup but is a different final source tree from
the measured patch 1; its own validation must identify that tree.
