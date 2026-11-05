# Native operand batches for selector fitting

Base `65c6e62f`, immutable C36 image and source pin
`6d4e49d1cdbb6513819e47145e0e52fdab327b6b`. This is separate from the
completed-answer benchmark. One intended variable for this preparation is
collecting 64 native query rows per layer instead of a single row. Native model
weights, scoring and attention remain unchanged; the live-list scratch guard is
retained. No selector map is trained or qualified yet.

The diagnostic hook requires an explicit case/prompt hash in `arm.json`, skips
startup and graph capture, and records three dense layers plus one source layer
per rank and case. It waits for 64 queries with at least 32,769 visible keys,
retains each query's table/length and copies only written native key and scale
prefixes. Distinct query tables are supported. A 512 MiB host-side union limit
refuses excessive captures. Raw inputs and future fitted weights remain
node-local, outside Git.

Use separate seeded public source prompts for fitting and validation. No
LongBench pilot or held-out questions may supply training activations. Separate
documents, not adjacent query rows, define validation independence. The first
mechanical checks must cover both native page sizes, multiple query tables,
partial/aliased pages, skipped startup, explicit caller identity and overwrite
refusal. Full native replay must validate every captured row before fitting.

This profile is untimed: copies and synchronization perturb execution. Never
mount this capture hook in a timed LSA or native control profile. No capture run,
performance result or quality improvement is claimed. A future collection must
use the authorized, reserved all-node window and restore C37 after its jobs.

Preparation applies the scratch guard and this experiment's ordered patch in a
fresh named C36 tree; all three overlays match byte-for-byte and `diff --check`
passes (`runs/local/source-check.json`). An initial attempt used a repository
without the C36 object; that refusal and the correction are retained. The
CPU-only pinned-container copy checks pass on sp8 for 128/256-key pages, 64 rows,
different query tables, physical aliases, partial pages, zeroed saved tails,
arming/graph/decode skips and overwrite refusal (`runs/copy-check-01`). The
container and reservation are removed. These checks do not replace native GPU
replay, which is still required before any fit.

All-row synthetic native replay then passes on sp8 (`runs/replay-check-02`),
including exact visible BF16 score bits, selected keys and source handoff/counts
for both page sizes and 64 rows. A separate process with the CUDA allocator cache
disabled and strict compute-sanitizer initcheck reports zero errors
(`runs/initcheck-01`). The first replay launch used the wrong working directory
and did not start a GPU container; that error is retained in `runs/replay-check-01`.
Every standalone reservation is released and no containers remain.

`collect.py` prepares two fitting documents and one validation document from
entire, SHA-deduplicated Python standard-library files assigned to disjoint
cohorts before capture. It uses no benchmark questions. The original file hashes,
actual tokenizer counts, exact commit and every response/capture identity are
recorded; prompts, responses and tensors stay private on the nodes. Replicated
ranks are mechanical checks, not independent training examples: use rank 0 for a
fit and keep document-level validation separate. Collection is not yet run.

Future collection command, from the clean published sp5 controller checkout
after the current answer-quality window restores C37 and releases all nodes:

```sh
python3 experiments/2026-11-05-selector-fitting/collect.py \
  --out /home/chris/spark-lab/results/selector-fitting/capture-01
```

The runner performs all-node preflight, a quiet-service guard, exclusive native
reservations, heartbeat/watchdog, published commit sync and named coherent
capture startup. Each explicitly armed document has one untimed generation
request. It disarms every node and restores C37 once before releasing holds.
Every actual captured batch still needs exact native replay before fitting.
