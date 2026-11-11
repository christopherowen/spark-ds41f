# tp4-ring temperature-0 determinism: the root cause

The determinism gate failed intermittently at `tp4-ring`, about one boot in three, on candidate-54 and candidate-58.
A failing boot could diverge in its first, cold `reference` run. This folder is the investigation's record and its
tools. It was candidate-58's (`boot-differences/`, `receipts/determinism-investigation/`, `receipts/tp4-ring/gate-repeat/`)
until candidates 55-58 were removed on 2026-11-11 (owner); it moved here then, unchanged except for the three path
fixes below.

## Result

**Cause:** `moe_combine_v41` (`vllm/models/deepseek_v4_1/tilelang/moe.py`, from 1107's `0756`) loaded the shared
expert's output (`Shared`) before `pdl.chain()`, its `griddepcontrol.wait`. The shared expert runs on the aux stream,
and the combine joins it with an event wait just before its launch. In a captured graph with programmatic stream
serialization, that dependency can be a programmatic edge, which orders only what follows the wait. So the combine
could read shared rows the shared expert had not yet stored, and that layer's MoE output then differed on one rank.

**Fix:** candidate-54's `0799` moves the wait above the shared load (owner, 2026-11-11). Its verification, with the
comparison against the record below, is
[candidate-54/receipts/gate-0799](../2026-10-26-spark-ds41f-candidate/candidate-54/receipts/gate-0799/README.md):
0 of 12 failing boots at `tp4-ring` after it (4 of 12 before), and 0 of 4 at each switched testbed.

**The same race split the drafter's decisions between boots.** The drafter's layers are `DeepseekV4DecoderLayer`s, so
its MoE runs the same combine inside its captured query graph. In candidate-58's c1 arms at `tp4-switched` (3 boots
each), the boots' accept decisions agreed only where the hazard was removed (shared expert on the main stream; query
forward without its graph) or row taps were on, at unchanged acceptance. On candidate-54's rebuilt image, 4 of 4 c1
boots at `tp4-ring` agree on every request (candidate-54's `c1-decisions-tp4.json`).

## The record before the fix

**Single-GPU row invariance** ([receipts/determinism-investigation](receipts/determinism-investigation/result.json),
sp8): no kernel is convicted. #961, #963 (both versions), #987 and #1001 (both versions), the WO-B and lm_head paths,
sparse MLA, the indexer and the rest of a TP4 decode layer give a fixed row the same bits at every M from 1 to 128 and
with any neighbours, on one GPU. That covers each kernel alone, not the ordering between streams in a captured graph.

**Gate repeat** ([receipts/gate-repeat](receipts/gate-repeat), `tp4-ring`, 2026-11-10): twelve boots interleaved, four
each, with the gate's records kept per call by [token_determinism_record.py](token_determinism_record.py).

| Arm | Boots identical | Hashes (logprobs) |
| --- | ---: | --- |
| candidate-58 | 3 of 4 | 18d61961 / d6568649; boot 3's reference f06c3f3c |
| candidate-58 without candidate-51's patches | 4 of 4 | 18d61961 / d6568649 |
| spark-ds41f-1107 | 4 of 4 | 8e46794a / 11ec073c |

candidate-58's boot 3 failed in its first c1 run (`reference`). One request diverged, prompt 7: its logprobs are equal
to token 113 and differ from token 114. The row trace (candidate-58's `rowtrace-tp4.json`, 8 boots, 3 failing) then
placed the first difference in layer 30's MoE output, on one rank.

## Contents

| Path | What |
| --- | --- |
| [token_determinism_record.py](token_determinism_record.py) | The gate with each differing request's outputs kept (`--record`); candidate-54's gate specs run it |
| [boot-differences/](boot-differences) | Trace collection and diffs: `collect_traces.py`, `diff_step_traces.py`, `diff_draft_traces.py`, `first_divergence.py`, and `c1_decisions.py` (each request's per-step token counts compared between boots) |
| `boot-differences/overlay/`, `make_overlay.py` | The trace taps: candidate-58's vLLM files with the step and row traces. They apply to candidate-58's tree only |
| [collective-rowinv/](collective-rowinv) | The fabric bit test for the `tp4-ring` collectives |
| [receipts/](receipts) | The row-invariance record and the gate repeat |

**The runs' specs and profiles** (`gate-repeat-tp4.json`, `gate-arms-tp4.json`, `rowtrace-tp4.json`,
`c1-steptrace-*.json`, `c1-rowtrace-*.json`, `c1-drafter-arms-tp4-switched.json`, `collective-rowinv-*.json`, and the
arm profiles in `boot-differences/`) extended candidate-58's profiles and image, so they were removed with it. They
are in Git at commit `07b2adfe`, under `experiments/2026-10-26-spark-ds41f-candidate/candidate-58/`. Their outputs stay
on the heads under `results/private/` (`determinism/`, `boot-differences/`).

**Path fixes in the move:** `token_determinism_record.py` finds `experiments/2026-10-12-r6g/token_determinism_text.py`
as `parents[1]` (was `parents[2]`); `collect_traces.py` finds the repository as `parents[3]` (was `parents[4]`);
`c1_decisions.py` takes any `<arm>-boot<N>` label (was only `c1-<arm>-boot<N>`).
