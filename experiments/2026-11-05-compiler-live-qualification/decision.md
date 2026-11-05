# Reject C42; retain C40

The corrected compiler stack fails its retained-feature gate. All136 manifest
files are covered:8073 passed,360 skipped, one failed, no errors. Native XML
independently matches those totals. The sole failure is IKET's
`test_on_matches_off_and_keeps_its_own_cache`: an explicitly prepared kernel is
ignored when #880's wrapper-cache bypass is active, then frozen compilation
correctly refuses. The focused review and independently repeated source-method
witness identify this branch. This case has its own temporary cache, so the
separate shared-cache provenance gap does not explain it away.

The C44 package-identity proposal cannot repair that branch; its JIT source is
byte-identical. Withdraw it unmeasured, keep its source proofs and disabled
profiles, and do not build it as a proxy fix. Under the declared bounded
corrected-stack rule, drop this compiler integration rather than add another
algorithmic retry. The standalone positive evidence for the other compiler
fixes remains archived; it does not qualify this combination.

C42 never seeded/distributed its image or ran functional/serving/performance
phases. There is no throughput over/under or performance-regression claim.
The native release gate really failed: actual wait exit1 and
`RuntimeError: image-and-gate exited 1`. It is not classified as a scheduled
sharing exit, even though a prior closure-only request was pending. No safe
passed-image resume is permitted. Top-k initialization and fresh compiler
feature processes did not run.

The image70e9abba... and exact original logs/receipts/XML/JSONL are preserved.
All29 builder/gate and21 head artifact hashes verify locally. Original failed
verdicts and control/restore JSON are published byte-for-byte; raw logs stay
private under the failed export. Independent named doctor and all three stable
C36 containers confirm the same image/profile, running state and zero restarts;
both controller/supervisor are absent. Only the matching own closure request
was fulfilled/removed. No hardware-review hold or native guard halt occurred.

C40 remains the best validated corrected default-enabled base. Continue the
separate C43 frontend cleanup on its old compiler. C42 did not touch serving
rank caches; isolate all subsequent DGX4 release-gate caches from the shared
path C42 used. No old image, completed gate or timed point is remeasured. No
promotion or immutable baseline change follows from this rejected round.
