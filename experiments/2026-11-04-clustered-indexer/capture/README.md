# Real indexer input capture

Status: collection completed in window `dgx1-1793801035`, closed14:25:30UTC
after verified C36 restoration. Twelve operands (three ranks, dense2/8/14 and
source20) replay exactly twice on sp8, but retain every visible key. See
`../native-input-results.json` and `../runs/capture-01/`. Synthetic capture/layout
checks pass for both native key-page sizes and strict initialization checking.
This arm retains exact C36 scoring/attention arithmetic and checkpoint weights.
No benchmark result comes from capture: copies and synchronization perturb timing.

The ordered series applies after C36 tree
`cc0fdf582d0bddff71c6a853e237bc9e87587832`, reproduced by all 558 source patches.
`base.json` pins both original source hashes; `check_sources.py` reverses every
capture patch and recovers them exactly. No source-layer or caller identity is
inferred from allocation order.

The hook requires an explicit `arm` file, skips graph capture/startup, and defaults
to at least 32,769 visible keys. It reserves independent budgets for three dense
pools and two source pools per rank, so early dense layers cannot exhaust source
coverage. Each sample retains caller prefix/layer identity, source interface,
native planned width and active limit. It copies only written key/scale byte
prefixes, deduplicates physical IDs by maximum written prefix, and initializes
saved tails without reading device tails. Raw activations stay outside Git.

Replay reconstructs original physical page IDs, byte stride and alignment with
unwritten unreferenced pages. It refuses an address span above 8 GiB rather than
silently switching to compact timing geometry. Every selected key, evaluated BF16
score bit, full bound, source handoff/count and following reindex output is gated.
Graph rollback, empty domain and restoration are checked. Only the candidate is
timed; native fallback is correctness-only.

The completed collection used the owner-authorized coordinated window after
the previous holder released it. Any future collection must establish the
current holder and authorization again; saved-input replay needs no serving change. This profile uses an isolated published checkout on every
rank, the same topology, an isolated API on port8001, and existing node-local
compile caches. Startup memory guards remain unchanged. Native serving checkouts
remain at their current published commit for restoration.

1. Read the native hold/request and verify the actual serving profile. Open the
   window from the clean production checkout through `scripts/lab.py window open`
   with note `owner-authorized clustered-indexer-input-capture` and that exact
   `--production-config`. Never treat another `spark-lab` holder as this window.
   Keep its heartbeat current throughout startup, capture and restoration.
2. Publish this whole commit. Read the explicit capture profile's coordinated
   `cluster sync` plan, then apply the exact published revision on all three
   isolated checkouts. Read the `cluster start --replace` plan's `live:` and
   `shape:` lines before applying; topology must remain unchanged. Use explicit
   `--cluster-config` on every apply, and never use a one-rank serving launch.
3. Wait for API readiness and startup guards. Refuse existing snapshots/partial
   files, then create `arm` in each node's ignored
   `~/projects/spark-ds41f/cache/clustered-indexer-capture-20261104/` directory.
4. On dgx1, run the pinned `capture/request.py` with the exact window ID and a new
   node-local `--out` directory. It generates a seeded public Python source prompt,
   retains source/prompt hashes, checks the actual served token count, and sends
   one greedy completion request. The native checkpoint's compression ratios1/2
   make the default approximately131K-token prompt cover long dense/source keys.
   Prompt contents and response remain node-local; this is not a timed baseline.
5. Collect all `rank-*.pt` into sp8's
   `~/spark-lab/inputs/clustered-indexer/`, retaining original files and SHA-256
   values. Source/dense filenames and saved caller metadata distinguish samples.
   Disarm every node. Stop the capture profile coherently and restore the window's
   recorded profile with named-profile commands. Check `doctor --live`, then close
   the lab window and verify the hold is removed.
6. In the next isolated sp8 reservation, run `sp8-replay` twice in fresh pinned
   C36 containers. Preserve every refusal/error and every unfavorable pruning/cost
   result. A positive kernel result still requires full serving qualification.

The initial request has124426 actual prompt tokens and saves one query per
distinct pool/interface and rank. Execution mode was not saved; original
query-row count is recorded and is not used to infer decode or prefill coverage.
It establishes neither mode coverage nor model quality across a prompt corpus.
If real bounds do not prune enough to cover cost, reject this grouping and revise
before serving integration. The historical replay checks exact keys; future
candidates use the generated-token gate in `../acceptance.md`. Changed keys and
intermediate values are allowed when the generated token sequence matches.
