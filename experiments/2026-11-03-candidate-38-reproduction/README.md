# Candidate-38 reproduction

Base: `6d59eaebf4ae569075f88b0466051dfcca053699`. Host-only checks completed;
full-model tests prepared, not run. Runtime authorization is still required.

Use the unchanged final candidate-38 image `213ba4df68d0`, vLLM head
`1f33102547e4da5c64712390604695d1abd577c4`, tree
`c061e5f6931306fa4655ba531428f639c365bf10`. No source overlays or effort clamp.
The tracked base reconciles the installed site profile. Its local node map must
name sp5–sp7, head 10.30.0.13; preserve other workstations' same-named maps.

Run `lab-loop.json` first, then `lab-copy.json`, each on a boot and repeat boot.
The loop matrix tests numeric effort 100/75 with default/off chat detection:
two seeded 64-token continuations, four seeded 1,024-token active-turn requests,
four clean 6,000-token onset probes. Cases and source hashes are self-contained.
Ten requests/boot, 28,224 output-token cap; sequential, greedy, fresh cache salts,
600-second request deadline, 32 MiB response cap. Returned tools are never executed.
The seeded screenshot transcription tests sustainment, not onset; short completion
requests do not support the chat detector-off override.

CopyDrafter: `copy_probe.py --prompts 3`, then single-stream on the SAME boot, then 8/16 distinct
streams (three samples, 1,024 tokens). Original trigger settings: min-match 8,
drafts 8–16, dead-row cutoff 0, budget 6155, capture 59. Old source overlays are omitted.
Historical crashes are positive controls; no released baseline is remeasured.

Publish one checked commit and inspect sync/start plans' live/shape lines.
After explicit authorization, use the lab window, publish and idle guards,
sync all three clean checkouts together, and retain one window across both phases.
The verified live restoration profile is config/candidate-37-tp3-switched-dev.local.json.
Restore it once at the end. Review both loop summaries and runtime receipts before
starting the CopyDrafter phase; a zero lab process exit is insufficient. Commands
(first add --dry-run):

```sh
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-reproduction/lab-loop.json --production-config config/candidate-37-tp3-switched-dev.local.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-reproduction/lab-copy.json --production-config config/candidate-37-tp3-switched-dev.local.json
```

Preserve every request, response, error, token/logprob/usage/stop record, rank log,
image identity, memory observation and artifact hash. Check each workload's exit;
the lab may continue after failures. No GPU or speed measurement has run here.
A bounded clean result cannot exclude longer loops or all NaNs. The stock image
has no per-step layout recorder, so no crash is not proof every rank agreed.

Host-only results and exact commands: runs/2026-11-03-host-budget/.
check_host_budget.py executes SHA-verified pinned host budget/compaction bodies:
mirrors 16/5/16 produce rows 17/6/17 on candidate 37 and 17/17/17 on candidate 38.
Five cases pass, including absent/forbidden obsolete callbacks. Simulated CPU
ranks and representative finite confidences; no GPU, collective or model forward.

Offline check: python3 experiments/2026-11-03-candidate-38-reproduction/loop_probe.py --check.

Preflight receipts: `runs/2026-11-03-preflight/`. Four doctor checks and both lab
dry-runs passed in an isolated checkout using the actual site map. Start plans
show live candidate-37 and unchanged shape; sync targets sp5, sp6 and sp7.
Initial macOS SSH socket-path failures were retained in private logs and resolved
with `TMPDIR=/tmp`; no runtime changed. Repository suite: 626 tests, 5 skipped.
