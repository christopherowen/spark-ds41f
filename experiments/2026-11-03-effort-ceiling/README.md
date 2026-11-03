# Reasoning effort: serve max and ultra aliases at 85

Status: candidate patch prepared and locally validated; **not deployed**. No
cluster sync, start, stop, or restart was performed. The served cluster still
runs candidate 36, which renders `max`/`ultra` (and any explicit integer) at
their requested budget today.

## Why

A repo-wide search found no reasoning level set to 100 in this repository:
`config/cluster.json`'s serving args set no effort default (unset requests get
the wrapper default `high` = 75), the bench tooling sends `thinking: false`,
and no gateway/host/docker config carries an effort. The 100 observed in the
incidents enters per request from clients, which send the named aliases
`max`/`ultra`. The owner chose to lower exactly those two names; explicit
integer efforts are served as requested, so a client that deliberately sends
`reasoning_effort: 100` still gets 100 (known open hole - the incident regime
remains reachable that way).

Measurements behind the value (experiments/2026-11-03-loop-onset, live,
greedy, fresh cache salts): identical tasks reason 5-13x longer at effort 100
than at 75 and restate instead of concluding; the degenerate short-restatement
regime, once present, is continued by the checkpoint's own distribution at
p ~ 0.9-1.0 per token at every context scale and effort tested. 85 sits above
the named `high` (75) so ordinary traffic is untouched while the named
maximum-thoroughness regime is closed at the boundary.

## The patch

`0001-feat-dsv41-serve-max-and-ultra-effort-aliases-at-85.patch`, prepared on
the candidate-37 series head `5b7cfa37fe` (branch `effort-ceiling` in the
prepared tree; prepared trees are disposable, the patch is the durable
source).

- Upstream base: candidate-37 complete vLLM series at `5b7cfa37fe`.
- Change: `vllm/tokenizers/deepseek_v41.py` maps a validated `max` or `ultra`
  to `REASONING_EFFORT_ALIAS_CEILING = 85` before encoding. The encoder
  (`deepseek_v41_encoding.render_reasoning_effort`) is untouched and still
  renders any budget it is given; the remap is request-path policy only.
- Quality implications: clients asking for `max`/`ultra` get 85 silently -
  observed behavior changes (shorter reasoning, conclusions instead of
  endless restatement). Explicit integers 1-100, `none`, thinking on/off, and
  invalid-effort rejection are unchanged.
- Tests: in-tree `tests/tokenizers_/test_deepseek_v41.py` parametrizations
  updated (max/ultra -> 85; explicit 86/95/100 pass through) plus a new
  `test_max_and_ultra_aliases_serve_the_deployment_ceiling`; these need the
  full test environment (torch) and belong to the candidate-image run. An
  earlier draft of this patch capped every value above 85; it was replaced by
  this alias-only remap per the owner's direction before any use.
- Local validation without torch used the stub-adapter harness (same loader
  as the 2026-11-03 loop-cause probes) through `check.py`:
  - `baseline-check.json`: candidate-37 base commit `5b7cfa37fe` (clean
    worktree, no patch) renders max/ultra at 100 and passes integers through -
    the control. (An earlier baseline receipt was accidentally taken against
    the first draft patch and showed 85s; it was discarded and rerun against
    the true base.)
  - `ceiling-check.json`: patched tree renders 85 for max/ultra, exact
    budgets for None/low/high/xhigh/minimal/medium/1/42/85/86/95/100, and
    still rejects every invalid spelling.
  - The pure encoding test file could not be collected in the torch-less
    venv; the encoding module is byte-identical in this patch (only the
    wrapper and its test changed), and `check.py` exercises the same
    spell-to-budget table through the real wrapper.
- Upstream status: not submitted. This is deployment policy for this stack;
  upstream has no such cap and the 100-budget behavior is the vendor's.

## Applying and deploying

Not done here. Next steps if accepted: fold the patch into the next candidate
series after candidate-37, run the full in-tree effort tests on the candidate
image, qualify, and deploy through the coordinated `cluster sync` + `start`
with `--apply` under a hold window. The remap takes effect for every node at
once or not at all; mixed rank state is invalid.

Commands used (deployment repository root):

```sh
.work/loop-analysis/venv/bin/python experiments/2026-11-03-effort-ceiling/check.py --baseline --source <candidate-37-base-worktree> --receipt experiments/2026-11-03-effort-ceiling/baseline-check.json
.work/loop-analysis/venv/bin/python experiments/2026-11-03-effort-ceiling/check.py --receipt experiments/2026-11-03-effort-ceiling/ceiling-check.json
```

No performance or GPU measurements apply. `bin/spark doctor` and
`git diff --check` pass for this change (recorded in the session log for
2026-11-03).
