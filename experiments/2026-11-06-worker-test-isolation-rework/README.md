# Worker-test lazy-reference isolation rework

Candidate46 preserves C43's two frontend lifetime corrections on the validated
default-enabled C40 runtime and compiler. Its only additional source change is
one test-fixture correction. This is correctness and maintenance work: neutral
performance is acceptable without a material core or serving regression.
Compilation or preparation trouble does not constitute a performance miss.

Canonical source preparation, an actual-image CPU fixture proof and both static
profile doctors have passed. The image is not built; full native gates, both
functional boots and both complete metric boots remain pending. C43's original
failed image and verdict remain unchanged and are never resumed or re-gated.

## Preserved failure and correction

C43 built successfully, then recorded8103 passes,360 skips and one failure in
its138-file release gate. The failure was
`test_gpu_verified_logprob_token_ids.py::test_tilelang_worker_modules_hold_no_live_triton_reference_kernels`:
`logprob._topk_log_softmax_kernel` was unexpectedly a real module global. The
two new frontend regression files passed their10 and20 cases. No C43 functional
or performance phase started; independent all-rank C36 restoration passed.

Worker runtime and relevant test bytes were unchanged between C40 and C43.
Adding two frontend files shifted the ten-file process groups: the logprob
producer moved from C40 chunk011 into C43 chunk012 before the lazy-global
consumer. Its `monkeypatch.setattr` resolves a lazy reference, and cleanup can
restore that reference as a real global. Runtime `logprob.__getattr__` does not
cache it. A read-only scan of all78 worker test files found one matching setter.

The correction uses `monkeypatch.context()` and dictionary-based poisoning,
then explicitly asserts that cleanup preserved laziness. All original
numerical, routing, empty-shape and launch-count assertions remain. The consumer
invariant,138-file coverage, process grouping, runtime files and two frontend
patches are unchanged. No tolerance, deselection or runtime fallback is added.

## Completed preparation evidence

The bounded SP8 canonical replay reproduced all576 C43 prefix patches and then
independently reapplied the formatted fixture patch. All1767 source steps
exited0; the actual supervisor wait exited0 at2026-11-06 01:12:22 UTC.

- vLLM head: `d92f9c26e7e749ed15850d24a9e419ebad90a91f`.
- Tree: `8056f1ef9761065bf9c66cb82b008ff71b6f310a`.
-577-patch fingerprint: `aa971421ba2bb74223b8dbc046222700ec9dc2049b0547ae1da686e3ced94ffe`.
- The C40 TileLang compiler remains `a0e66d71fe7617216f2de5d86c3b088595367387`.

The source job enforced one CPU,8GiB memory/address space, no swap and threads1,
with both shared CPU locks and an owned host claim. A separate one-CPU/4GiB
metadata job verified every prefix patch byte and the resulting series identity.
Both claims were removed. See
[canonical-source-proof.json](runs/preflight/canonical-source-proof.json).

The immutable C43 image's actual dependency is pytest9.1.1, monkeypatch module
SHA256 `b5c405ed57dbd58adfb7f0fae44f84d62402993c50058e07f03387285c03df33`.
A CPU-only proof using that dependency reproduces the original lazy-global
failure, passes the corrected fixture and still detects preexisting pollution.
Three actual-image runtime file hashes match canonical source. Both ephemeral
containers exited0; the actual supervisor wait exited0 at01:30:45 UTC. Image
identity and labels were unchanged. Enforced bounds were one CPU,2GiB memory,
no extra swap, no network and no GPU devices. Synthetic module/reference objects
isolate the fixture mechanism; this is not a numerical or native gate pass.
Original small logs and hashes are in
[actual-image-fixture-proof](runs/preflight/actual-image-fixture-proof/).

Both static TP3 and TP4 doctors passed at published commit `3de1bbba` under a
one-CPU/8GiB/no-swap SP8 scope. These used ignored copies of committed example
node maps, never real site configuration or deployment authority. Actual wait
exit0 is retained in [attempt4](runs/preflight/build-controls-attempt-4/).
All19 unique affected build-control tests are covered: the first run passed18
and encountered one missing sparse-checkout fixture; the two affected checks
passed after the fixture and file-handle correction. These tests do not prove
an actual build worker or C46 image. No passed native gate or benchmark was rerun.

All preparation errors remain recorded: the initial inspection helper's tuple
assignment error, vanished-hold and series-path probes, incomplete draft hunk,
pre-SSH transport parsing error, sparse offline fixture omissions, the manifest's
576/577 count typo and a later offline-fixture quoting error. The count correction
changed no source head/tree/runtime bytes. See the preflight receipts and
[static-profile-preparation-issues.json](runs/preflight/static-profile-preparation-issues.json).

## Qualification still required

The control packet retains all138 native files and eight image/gate steps,
40 ordered functional exits and26 ordered performance steps across the unchanged
full `release-tp3`, `single-stream` and `sampled` families. No C46 image ID or
serving result exists. [control-packet-status.md](control-packet-status.md)
records the remaining integration and remote control-test work. The unchanged
canonical indexer hash was independently read on SP8 and is recorded in
[topk-source.json](runs/preflight/topk-source.json); the new image must still pass
its own actual-source, exact-output, zero-error sanitizer check.

Use the published declared-evidence API after review. Require the same isolated
sampled repeat as C40 and preserve sampled C8 batch differences as a limitation;
do not turn an observation into a retrospective hard gate. Forty zero process
exits alone are insufficient. Temperature-zero, model and every other declared
required functional property must pass before timing. Final local semantic and
hash audits remain required after the complete round and independent restoration.

All substantial preparation, tests, builds and analysis run remotely with hard
resource bounds and shared locks/claims. The Mac holds lightweight edits and
small evidence only. Re-read actual pool holds/requests before any hardware work;
preserve earlier Engram and expert-route claims. No acquisition waiter or new
hardware request is running here. A full, exact, published DGX1–4 lease and named
same-commit staging are required before one new build/gate/boot-repeat round.
No topology, hardware, cooling, limit or release-policy bypass is authorized.
