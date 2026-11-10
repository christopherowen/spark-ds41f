# candidate-55 with #978 round 2 at tp4-ring (dgx1..dgx4), 2026-11-10

[../../validate-978r2-tp4.json](../../validate-978r2-tp4.json), run in dgx1's window (10:30-10:43Z): the candidate-55
image with #978's round-2 files mounted (`../../tp4-978r2.json`), so the code is candidate-54 + #1008 + #978 r2 +
#988's dead-row part.

| Run | Result |
| --- | --- |
| Boot | ready in 109.9 s |
| Image chat (`check_978_image_chat.py`), against candidate-54 in the same window | passes: both turns' greedy outputs match candidate-54's (`1.1`, `1.2`, `2.1`, `2.2`); turn 2 0.51-0.57 s |
| Gate | **fails**: reference hashes 18d61961/d6568649 (candidate-54's), but the `again` scenario (the same prompts re-run after other traffic, on prefix-cache hits) differs in one request, in both the logprob run (logprobs from token 176, a token at 189, prompt 2) and the greedy run. Every other scenario is identical |

candidate-55 without #978 on the same image passes every scenario, `again` included (`../tp4-ring-no978/`), and so
do candidate-54's gates at `tp4-ring` and `tp4-switched`. So the difference is #978 round 2's: its scheduler and
prefix-cache changes reach text requests re-run on cached blocks. #978 declares no output change.

**Verdict: #978 misses at the gate after its second and last rework, and is dropped from this release.** Its record
stays: the serving-path gain it showed on sp8 (image turns 2 onward 4.2-9.5x) is in `../rework/pr-978-r2/`.

## Correction (2026-11-10): the `again` difference is not shown to be #978's

The verdict above drew too much from one boot. Counting every recorded gate on the nodes: of about 500 `again`
runs, 3 differ, and every other scenario of those runs is identical:
- `candidate-51-tp4/greedy-candidate-51.jsonl` (one token): candidate-51 carries no #978;
- this run's two (`determinism-` and `greedy-candidate-55-978r2.jsonl`), from the same boot.
So the `again` scenario has a rare difference in the stack without #978 (about one boot in 250 on this record). #978
round 3 ([../rework/pr-978-r3](../rework/pr-978-r3/result.json)) also shows round 2 never changed the text-only path:
every 0793 hunk in `scheduler.py`, `core.py` and `async_llm.py` is unreachable or inert for a request without
multimodal items, the gate's requests never hit the prefix cache (each carries a fresh `cache_salt`), and a replay of
the gate's requests through the real scheduler and KV cache manager is identical on candidate-54, round 2 and round 3.

**Revised verdict:** #978 is not dropped. Its round 3 (granted by owner exception) is judged in the next round on
candidate-58, with a gate on two boots, which tells a repeating difference from the stack's rare one.
