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
