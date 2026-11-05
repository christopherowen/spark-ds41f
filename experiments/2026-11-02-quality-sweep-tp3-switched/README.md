# Quality sweep: the checks we don't run every release, on tp3-switched

Owner, 2026-11-02: run a NaN scan, AgentX, the image-stack checks, and every other
quality measurement we have but don't run regularly.

**Status:** queued in the sp5–sp7 window `spark-0013-1793654131`, after #856's
limit-sync runs. Not run yet. Health checks of the serving build, not a
candidate: no recipe change.

- Build: candidate-37 at tp3-switched, the profile sp5–sp7 serve
  (`config/candidate-37-tp3-switched-dev.local.json`, node-local), image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b1953b1…` on sp5, sp6 and sp7). vLLM
  patch head `5b7cfa37f`, tree `af0fb152`. Deployment commit: the queue's sync
  to origin/main.
- No variable: two arms of the same build.
  - [arm-c37.json](arm-c37.json) adds only the weight digest
    (`SPARK_DS41F_WEIGHT_DIGEST=1`).
  - [arm-c37-poison.json](arm-c37-poison.json) adds the KV poison and NaN-logit
    counting.

## The runs

[lab-quality.json](lab-quality.json), run `quality-sweep-sp`:
1. **Fabric** (serving stopped): `scripts/probe_collectives.py --port-samples`
   on every node. It checks exact all-reduce, all-gather and reduce-scatter,
   eager and graph replays, and that every HCA carried traffic.
   `docs/switched-fabric.md` lists TP3 switched as unqualified until probed.
2. **Both arms, one boot each, the same workloads:**
   - the lean bench's `quality` and `compliance` suites;
   - the weight digest per rank (`loader_report.py`), compared across ranks and
     both boots;
   - the gate's output digests:
     - `token_determinism_text.py --wide`, with and without logprobs;
     - `prefix_extension.py --threshold 6096`;
     - `controls_digest.py`;
     - `sampled_digest.py`;
   - `vision_check.py`;
   - `acceptance_sweep.py`;
   - `determinism.py --repeats 5` and `determinism_concurrent.py`;
   - the prompt-logprob and logprob-token-id checks;
   - the sink batch and deep-context probes (64K, 512K);
   - `long_context.py` at 960K tokens, needles at 10% and 90% depth;
   - `kv_counters.py --nodes sp5,sp6,sp7` last: NaN-logit requests, preemptions,
     prefix-cache and draft counters, and every node's device asserts.
3. **The NaN scan** is the poison arm.
   - `VLLM_KV_CACHE_POISON=1` (vLLM 0500) fills every new KV block, compressor
     rings included, with all-ones bytes: NaN in every float format.
   - A read of a slot nobody wrote becomes NaN, and the router or the logits
     catch it.
   - `VLLM_COMPUTE_NANS_IN_LOGITS=1` counts requests whose logits held a NaN
     (`vllm:corrupted_requests`).
   - Pass: every digest equals the plain arm's, no NaN-logit request, and no
     device assert. `experiments/2026-10-25-kv-zeroing` notes that
     NANS_IN_LOGITS had not run in a lab boot before. If it stops the boot, that
     is recorded and the arm reruns without it.

[lab-suites.json](lab-suites.json), run `quality-suites-sp`: servebench
`screen-tp3` and `sampled` on one boot of the plain arm. There is no switched
suite reference, so it records levels only. The fan-control service is not
installed on sp5–sp7, and two release-tp3 runs earlier today stopped on the
bench's thermal guard. These suites may stop the same way, which would itself
be recorded.

[tables_arms.py](tables_arms.py) prints each workload's summary per arm.

## Not in this window, and why

- **AgentX** has never run a window. `scripts/agentx.py install` and `fetch`
  refuse under a hold. They need `uv`, which sp5 doesn't have, and a 1.72 GiB
  dataset download. Then its job (a port-8001 arm, points 4 and 8, 3,600 s
  each) runs in a window of its own, about 2.5 to 3 hours.
- **IKET NaN hooks** (`iket_step.py --nan-hooks`) run with dummy weights and
  need a hand-made bundle. They test kernel plumbing, not the served weights.
- **compute-sanitizer initcheck** (`kv-zeroing/bundles/indexer-initcheck`): the
  bundle pins an older image and has to be re-pointed to candidate-37.
- **The validate job**: its trace arms are B12X-era overlays.
- **Data the window doesn't have:** the vision eval (2,283 items, needs a
  Hugging Face download), GSM8K acceptance (no `~/spark-lab/inputs/gsm8k` on
  sp5), and tool-eval-bench (its venv is gone).
- **Too heavy for a window:** the reference parity probes (CPU, about 150 GB
  read per case).
- **Already in every release:** the release-test gate (121 files) and the
  candidate gate.
- **The lab's `image` job** can remove a stale tag, so it isn't run. The image
  identity is the digest above, checked on every node.

Offline image-stack checks, outside the window: `bin/spark doctor` (and
`doctor --live`, which the window runs at open and close),
`bin/spark build smoke --image spark-ds41f:candidate-37` on sp8, and
`scripts/kv_capacity.py --shape tp3-switched`.
