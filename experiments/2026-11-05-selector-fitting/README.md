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
