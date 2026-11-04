# Real indexer input capture (not yet run)

This arm is instrumentation, not a performance candidate. It extends the exact
candidate-36 profile the lab window originally found serving. Its patch adds an
explicitly armed one-shot hook before the dense indexer search. It copies one
query from each of three distinct cache pools and only written key/scale prefixes. Duplicate physical pages are
deduplicated using their maximum readable prefix; the saved remainder is initialized
without reading its device tail. It remaps the compact saved page table and atomically saves the tensors **outside Git**. Each rank gets
one file per captured pool. Native planned width, active limit, source-block
interface, physical page IDs, byte stride and alignment are retained. Replay
reconstructs that sparse address geometry within an explicit 8 GiB span budget;
a larger span is a reported refusal, never silently compacted. It changes no scorer arithmetic or model weights. Capturing and
synchronizing perturb timing; no performance conclusion comes from this arm.

The candidate-36 series was freshly applied: 558 patches reproduce tree
`cc0fdf582d0bddff71c6a853e237bc9e87587832`. The ordered `series` applies after
that tree. The executable `overlay` copies match that patch. The capture
profile uses an isolated checkout and passes local doctor; synthetic capture/layout
checks are prepared, while real hardware capture remains untested.

Use this only in an owner-authorized window. Restore the production profile
recorded by that window, which may have changed since this arm was prepared.
The API is isolated on port 8001. Its startup memory guards remain unchanged.
Do not arm the capture during startup's dummy/profile inputs.

1. Publish the complete commit. Inspect the capture profile's coordinated
   `cluster sync` and `cluster start --replace` plans and their `live:` and
   `shape:` lines. Use only the explicit profile with `--apply`, covering all
   configured ranks at the same clean detached commit. Do not change shape.
2. Wait for API readiness and all startup guards. Then create the empty `arm`
   file on each node in its ignored `cache/clustered-indexer-capture-20261104/`
   directory. Before arming, ensure the directory contains no prior capture;
   use a new directory/arm for a later run rather than overwriting evidence.
3. Send one public-source prompt long enough to produce at least 8,192
   compressed index keys (for example a 64K-token source-text prompt). Keep
   the prompt identity and hash. This request collects operands, not timings.
4. Collect each `rank-*-pool-*.pt` into the idle replay node's
   `~/spark-lab/inputs/clustered-indexer/`. Record SHA-256 values; retain the
   raw tensors node-locally. Do not put activations or prompts in this repo.
5. Close the window through the lab, verifying the prior serving profile and
   clearing the hold. In a subsequent authorized idle-node job, run the
   `replay` bundle twice in fresh containers. It compares all selected keys,
   checks each cluster bound and evaluated BF16 score bit, verifies source blocks
when published, and tests graph visibility rollback/empty/restore. It times only the
   candidate, including envelope rebuilding, selection and compaction.

A positive result justifies designing persistent metadata and CSA state ownership.
If real keys do not prune, retain that unfavorable result and revise the search
before any serving integration. A changed selected key fails the experiment.

The diagnostic now captures up to three distinct eligible owned pools per rank,
identified only by stable device allocation identity within that process. This
samples multiple source caches, rather than treating three replicated ranks as
three independent layers. Filenames and `pool_ordinal` retain that distinction;
no layer identity or decode coverage is inferred. CUDA graph capture is skipped,
so the long eager prefill is the intended source. No graph mode is changed.
