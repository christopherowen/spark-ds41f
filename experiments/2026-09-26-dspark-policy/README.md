# DSpark verification policy and draft cost

Base: the promoted `2026-09-25-karmic-kraken-r3-vision-kernel70`
configuration, on candidate image `vllm-ds41f-kkref:01f1b874c774-r4a`
(vLLM series 0001-0008, tree `0debe853`). Patches 0005-0008 are off unless
their variable is set, so one image serves every arm.

## Hypotheses

1. Pinning adaptive verification's cost curves
   (`SPARK3_DSPARK_COST_DIR`, patch 0005) removes most of the ~3%
   boot-to-boot throughput variation, which comes from the startup profile.
2. The marginal-rate verification rule (patch 0006) raises accepted tokens
   per step at unchanged step cost, most on code, where confidence varies
   most between steps.
3. The gathered Markov bias (patch 0007, `dspark_draft_topk` 1024) shortens
   the drafter by reading 1,024 rows of `markov_w2` per draft position
   instead of all 129,280, without lowering acceptance.
4. A five-draft trace at temperature 0 (patch 0008) prices draft counts,
   rules and confidence calibration offline.

## Arms

`make_arms.py` derives every arm from `config/cluster.json`. All arms use
the r4a image and its own compile cache, drop
`--default-chat-template-kwargs {"thinking":true}` (thinking is already the
default when a request names neither key, and the explicit default
overrode a client's `enable_thinking: false`), and profile 15 replays into
a pinned cost directory.

| Arm | Change from `base` |
|---|---|
| `base` | none (all new paths off; pinned curves in `dspark-costs/k3`) |
| `marginal` | `SPARK3_DSPARK_VERIFY_RULE=marginal`, cost scale 1.0 (the marginal rule prices drafts directly; scale 2.0 would double-count) |
| `topk` | `dspark_draft_topk` 1024; own pinned curves and compile cache |
| `fastcores` | container restricted to the ten Cortex-X925 cores (`--cpuset-cpus=5-9,15-19`); the RoCE proxy threads have no affinity and can otherwise run on the slower A725 cores |
| `det` | `B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1` and `B12X_DENSE_SPLITK_TURBO=0`: routed-MoE combine and dense split-K in a fixed order; own pinned curves |
| `k5trace` | 5 drafts, capture sizes to 48, trace to `cache/dspark-trace/k5.jsonl`; replay only, never timed |

## Workload

`run_arm.sh ARM LABEL --suites quality,decode --decode-cases
prose,code,prose-nothink,code-nothink,json-nothink`: the LRU gate and the
temperature-0 decode matrix at 1/2/4/8 streams. `prose` and `code` are the
reference cases (reasoning on, the server default); the `-nothink` cases
turn reasoning off to measure answers, and `json-nothink` is structured
output. Arms alternate (base, marginal, base, marginal) so each pair spans
two boots.

## Gates

LRU 5/5 on every arm; no failed requests; memory guards unchanged (5 GiB
startup, 3 GiB steady), and dgx1's lowest MemAvailable recorded per arm.
`--allow-mismatch` because `doctor --live` reports only the nvidia-drm
modeset host setting (display), which does not touch inference.

## Results

Runs: `results/private/bench/dsp-{b1,m1,b2,m2,t1,f1,k5trace}` (4 samples per
point, 8 for pooled arms). Every arm passed LRU 5/5 with no failed requests;
dgx1's lowest MemAvailable was 6.3-6.5 GiB (k5trace: 6.48 GiB with graphs
captured to 48 rows, 1.45 GiB of graphs).

- **Base (r4a, all new paths off)** matches the r3 reference at every
  reference point. The reference `prose` and `code` cells are 100% reasoning
  text; with reasoning off, accepted drafts per step rise from 1.19 to 1.48
  (prose) and 1.89 to 2.28 (code), and `json-nothink` accepts 2.25.
- **Pinned cost curves** do not remove the boot-to-boot spread: b2 reuses
  b1's curves and still moves up to 8% at single points. The spread follows
  acceptance, because temperature-0 outputs differ between identical
  requests (4 distinct outputs of 4 at one stream on every case). Single-
  stream step time (1 + accepted per step) / tok/s is stable to ±0.1-0.5 ms
  and is the sharper measure of cost changes.
- **Marginal rule** (pooled m1+m2 against b1+b2): reasoning text 2-5% slower
  (prose c2 -5.2 ±3.0%, code c8 -4.3 ±2.5%), answers unchanged; it verifies
  more rows (+0.1 to +0.9 ms per step). Rejected.
- **Gathered Markov bias** (top-k 1024): the drafter falls from 5.38 to 4.47 ms
  at one request (pinned curves), steps are 0.6-0.9 ms shorter, but accepted
  drafts fall 2-9% (code c8 1.72 to 1.65, prose c8 1.13 to 1.03): throughput
  neutral. The biased argmax sometimes lies outside the top 1,024 base
  candidates.
- **Fast cores** (container on the ten X925 cores): neutral everywhere;
  host thread placement is not limiting.
- **Five-draft trace** (replay.py): the logged accepted counts equal the
  matching prefix of the replayed stream in 19,654 of 19,654 steps. Raw
  confidences are calibrated (mean confidence against conditional
  acceptance by position: 0.78/0.77, 0.72/0.71, 0.71/0.73, 0.71/0.74,
  0.73/0.75), so calibration buys nothing. Depth 5 commits 3.26 tokens per
  request-step against 2.72 at depth 3 on the same drafts; priced with the
  pinned curves, verification policy moves throughput by under 5% even for
  an oracle.
- **Five drafts live** (the trace arm, pessimistic because it copies tensors
  every step): code and JSON answers +5-13% (accepted 2.9-3.2 per step),
  reasoning text and prose answers -3-8%. Single-stream steps grow by 5-7 ms
  for about one more verified row plus 0.9 ms of drafting, while the pinned
  curves price a verified row at about 0.4 ms: the startup profile's
  identical dummy tokens route to the same experts and under-price
  verification. Patch 0009 profiles on distinct tokens; the follow-up is in
  `experiments/2026-09-27-lil-head` (`k5real`).
