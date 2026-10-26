# The cluster program after spark-ds41f-1025 (2026-10-26, CET)

The owner's order (2026-10-26, about 22:00 CET): quick experiments that unlock research
first; then validation benching that needs no multi-hour release suite; full release benching
only once the quality, correctness and performance goals are met with enough gain for a
release.

- **Base:** `origin/main` 9275a790. Production is spark-ds41f-1025 (image
  `spark-ds41f:19f2c20ed4d6-next-6`) on the ring `tp4-ring`, deployed from 6af41e64.
  `main` renders the same launch (`bin/spark render dgx1`, byte-identical), so a queue's
  sync to `main` leaves production unchanged.
- **Cluster:** the ring (dgx1-4). The triangle is not cabled, so TP3 checks wait.

## Phases

1. **Quick probes on 1025** ([phase1.json](phase1.json)), then #620's
   `stall-probe-tp4.json`:
   - draft acceptance on GSM8K-200 at production's settings and with every draft verified;
   - per-position acceptance on the single-stream prompts with every draft verified;
   - an all-rank torch profile of c1 and c8 decode steps, with the engine grid's dummy step
     at their rows on the same boot;
   - #620's stall probe: the decode stall and short-request TTFT behind a long prefill.
2. **Lean validation of candidate 9** (vLLM `next-candidate` e2b2b0b59, image
   `spark-ds41f:19f2c20ed4d6-next-9`): the determinism gate against 1025, a lean screen,
   startup memory on every node, 32K and 131K prefill.
3. **Lean A/Bs of the ready features on candidate 9.**
4. Report to the coordinator, who folds the winners into candidate 10.
5. Full release benching of the final candidate, only on the coordinator's word.

## Arms ([make_arms.py](make_arms.py))

| Arm | What changes from 1025 |
| --- | --- |
| [arm-1025.json](arm-1025.json) | nothing |
| [arm-verify-all.json](arm-verify-all.json) | every draft verified: `enable_adaptive_verification` false, its row cost dropped |
| [prof-1025.json](prof-1025.json) | `--profiler-config` (torch, no stacks or shapes), `VLLM_SERVER_DEV_MODE=1` |

## Phase 1 workloads

- [gsm8k_acceptance.py](gsm8k_acceptance.py): vLLM's GSM8K evaluator prompts (the first 5
  training examples as shots, the first 200 test questions), raw `/v1/completions`,
  temperature 0, max 512 tokens, 8 in flight. It reads the speculative-decoding counters
  before and after and reports tokens a step (1 + accepted / drafts), survival by position
  (accepted at k / drafts) and accuracy. The data is the evaluator's own download, staged on
  dgx1 at `~/spark-lab/inputs/gsm8k` (test.jsonl sha256 3730d312…, train.jsonl 17f347dc…,
  from experiments/2026-10-18-apple-rsi's runs).
- [acceptance-suite.json](acceptance-suite.json): servebench `model` on the single-stream
  prompts at one stream with per-position counters (#598's c1-acceptance suite without its
  closed loop).
- [profile_decode.py](profile_decode.py): windows `c1-greedy`, `c1-sampled`, `grid-c1`
  (1 request at 3 and 6 rows, 4K context).
- [profile_c8.py](profile_c8.py): windows `c8-greedy` (8 distinct prompts), `grid-c8`
  (8 requests at 2 and 3 rows a request).
- `summarize_kernels.py` and `analyze_costs.py` are experiments/2026-10-25-request-inventory's,
  which the lab's profile job runs.

## Results

(Filled in as each phase lands.)
