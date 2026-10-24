# Every request type's kernels: what the TileLang migration has left

Owner, 2026-10-25: complete the TileLang port and include the night's fixes before blessing
another candidate.

The migration so far ported every kernel that a temperature-0 decode step launches
([experiments/2026-10-24-tilelang-completion](../2026-10-24-tilelang-completion/README.md)).
The deployment serves more than that:
- sampled requests (temperature, top-p, top-k) through the rejection sampler;
- structured output;
- logprobs and prompt logprobs;
- penalties, stop strings, thinking budgets and parallel sampling;
- long prompts;
- images (`--limit-mm-per-prompt {"image":4}`).

Those requests' kernels were never inventoried. This run profiles each request type alone, plus
eight mixed requests at once, on the ports build:
- vLLM `tilelang-ports` 1537606b0;
- the prompt-logprobs fix (0266) included, so its scenario does not crash the engine;
- Python stacks on, so every kernel that is not TileLang can be traced to the line that
  launched it.

- [prof-ports.json](prof-ports.json): the ports arm with the torch profiler and its stacks.
- [profile_decode.py](profile_decode.py): one profiler window a request type. It covers
  greedy, sampled, top-p, top-k, both, structured, logprobs, prompt logprobs, penalties, stop
  strings, a thinking budget and n=2, then eight mixed requests at once.
- [profile_prefill.py](profile_prefill.py): a 16K-token prompt and an image request.
- [inventory.json](inventory.json): the run, `request-inventory`.
