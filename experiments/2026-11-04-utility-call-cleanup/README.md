# Release abandoned utility calls

Deployment base: `a9ff19416fcd182eb8b971c00cec6907e4757a93`.
One variable: pending utility-call ownership in the engine client. This is
separate from the failed request-submission leak in the preceding experiment.

`SyncMPClient.call_utility()` and `AsyncMPClient._call_utility_async()` insert a
future into `utility_results` before encoding or sending. Only receipt of a
reply removes it. If encoding or sending fails, or cancellation interrupts the
send before the engine receives it, no reply will remove that entry. Cancelling
a caller waiting for a reply also retains its future until a reply arrives,
which may never happen. Repeated failures accumulate entries indefinitely.

The correction bounds each entry to its caller's lifetime with `finally`,
cancels an unfinished future, and consumes an already delivered error if the
send itself fails. A reply can arrive after cleanup, so the receiver must
tolerate absent entries. It also handles the synchronous future-cancellation
race without terminating the output receiver. Normal results and utility
errors are preserved; token selection, model execution and GPU work are
unchanged. This is not a serving-performance claim.

The affected methods are identical in candidates 36, 38 and candidate-39 plus
the prior request cleanup; the module also has identical bytes in the latter
two sources. The source patch, original/corrected module hashes,
test hash and immutable candidate-38 image are recorded in [base.json](base.json).
This source defect is not evidence that utility calls caused #884's overnight
memory loss or its reported candidate-38/candidate-36 difference.

## Isolated validation protocol

Use published inputs, idle Spaceport sp8 and bounded CPU-only containers. Do
not stop or restart a serving cluster. The host applies the published patch to
the hash-verified original image module; the disposable containers recheck all
hashes before running the actual client methods with deterministic transport
stubs. No GPU device, network or model download is needed.

Twenty cases cover encoding/send/setup errors, cancellation during send and
reply waits, 64 repeated failures per client type, late and duplicate replies,
synchronous/asynchronous cancellation races, concurrent callers and ordinary
success/error results. Expected control: 14 failures and six passes, no
errors/skips. Expected correction: twenty passes, no errors/skips. Do not count
collection errors as reproduced failures. Retain every attempt and native
pytest exit beside its controller receipt.

Exact controller invocation after publishing `<commit>`:

```
python3 run_spaceport.py --commit <commit> --apply
```

The script is restricted to `spark-0016`; it exports the named commit without
changing the site's checkout, checks reservations and idle containers, records
exact container commands, and removes its temporary containers. Each uses an
8-GiB memory limit, two CPUs, no network and no GPU devices. Source and runtime
qualification results will be recorded separately; no image is promoted here.

## Results

Spaceport sp8 reproduced the expected control: 14 failed assertions and six
passes, with zero errors or skips. The corrected source passed all twenty.
Each repeated-failure control leaves 64 pending entries; corrected clients
leave none. The ordinary success/error controls preserve their results.
The published input commit is `79e96d550c9dbac37f47892d0f1b55b371648a61`.
Exact commands, hashes, native pytest exits, JUnit and container states are in
[runs/spaceport-attempt1](runs/spaceport-attempt1). Neither container was
OOM-killed. Host MemAvailable minima of 115.205/115.200 GiB are test conditions,
not a claimed memory saving. No containers were left running.

The complete 575-patch source series replayed in a fresh clean worktree from
the pinned upstream revision. It reproduces head
`07733cc7611ad3822b313778bc3504f4233709d0`, tree
`ff08b5ed9f909de67fa74590e321ed634d2e54a3`, carrying both this correction and
the earlier request cleanup. The new profile passes doctor, and the local
release-test classification has 199 selected files (137 gate, 62 excluded,
zero problems). The first doctor run found an inherited patch count of 574;
it was corrected to 575, with the failed check retained. No build or runtime
source was changed to resolve that metadata error.

[AUDIT.md](AUDIT.md) records the other ownership paths inspected and conditional
retention outside the serving recipe. The new candidate image, full release
gate, serving performance and complete attribution of #884 remain pending.
Full build inputs are unavailable for `build check`. No prior candidate's
serving measurements qualify this new source.

Upstream status: local, not submitted.
