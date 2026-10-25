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

## Inventories 3-5

Inventory 3 never started: the ports build failed to start (undeclared draft-budget kernel;
see the tilelang-completion README).

Inventory 4 (vLLM 9554a406c's package files, the sixth serving check's build) started.
It then timed out on the decode workload's second window:
- **The cause:** `/stop_profile` waited for the profiler's summary table
  (`torch_profiler_dump_cuda_time_total`). The greedy window's trace was written within
  a second, but the table took 12 minutes, and the sampled window passed the client's
  900 s timeout.
- **Why the table is so slow:** with Python stacks, the greedy window's rank-0 trace holds
  1.39M events for a 0.55 s request. Of these, 1.29M are Python calls of the worker's
  idle wait on the shared-memory queue (`shm_broadcast` wait, `memory_fence`,
  `sched_yield`, 112k iterations); the table walks all of them.
- **The fix:** [prof-ports.json](prof-ports.json) turns the table off. The inventory reads
  the traces.

The same trace shows the JIT monitor (`--jit-monitor-mode warn`, also in production) at
work. It recomputes the TileLang cache key through the eager builder's `parse_args` on
each eager TileLang launch: 1238 calls in the window's 15 steps, about 82 a step, mostly
in step preparation such as `ced.requests`. Under Python tracing that is 20.6 ms of
216 ms of `execute_model`. The frozen kernel cache already refuses a miss by name, so for
TileLang kernels the monitor repeats a check. Its untraced cost per step, and whether
async scheduling hides it, is still to be measured.

Inventory 5 profiles the same build without the table.
