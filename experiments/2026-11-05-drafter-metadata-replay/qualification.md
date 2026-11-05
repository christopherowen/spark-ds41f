# Remaining qualification plan

This plan is not an instruction to run while another holder owns the cluster.
Before any GPU job or node operation, read dgx1 `~/spark-hold.json`; yield to its
holder. Preserve an existing `~/spark-request.json`. At preparation time both an
active frontend qualification hold and another task's request existed.

1. After a coordinated GPU slot becomes available, verify the clean source
   checkout is exactly the `candidate_source` commit in `base.json`. Use the
   immutable candidate-38 image with only this committed source delta overlaid,
   or a newly built candidate image with matching source labels. Keep resource
   limits explicit, a fresh task cache, and every exit/log. Run the new metadata
   graph cases in two separate processes, then the existing DFlash input,
   causality, DCP, CED metadata and sampling gates. The focused command inside
   the prepared environment is:

   ```sh
   .venv/bin/python -m pytest --noconftest -q tests/v1/attention/test_b12x_v41_ced_metadata.py -k draft_metadata_graph
   ```

   Require all three cases to execute and pass, exact integer equality on every
   readable row, and no compilation miss after cache freeze. Add a graph replay
   trace to confirm metadata kernels are inside the query graph and the eager
   rebuild is absent. Record graph memory before/after and startup compilation.
   If a gate fails, retain it and investigate before another candidate.

2. Complete all pinned build inputs on an idle DGX/Spaceport builder; run the
   candidate's `bin/spark build check`. If an image is built, build once and record
   its digest, source labels and whole release-gate result. Never overwrite an
   image tag or infer source identity from it.

3. Serving validation requires explicit owner authorization for its named
   coordinated operations. Before asking, produce the site-resolved TP3 plan
   for all configured nodes and the exact restore profile observed by the window
   owner. The source-only `candidate.json` deliberately has launch disabled.
   Publish one clean deployment commit before node synchronization; use only
   `bin/spark cluster sync/start/stop` with `--cluster-config` and inspect `live:`
   and `shape:` before any authorized `--apply`. Do not use `--change-shape`.

4. Within the authorized window, run candidate boot and repeat boot back to back;
   restore production once at the end. Require equal source/image identity on
   every rank and identical metadata/draft layout for fixed request inputs,
   including padding, cancellations and slot reuse. Exercise greedy and seeded
   probabilistic target verification, variable rejection counts, mixed prefill,
   streaming and disconnect/cancellation. Compare target tokens and sampled
   probability/logprob contracts with retained compatible fixtures. Check every
   rank's logs for stalled collectives, startup JIT, fallbacks and non-finite data.
   Untouched sampler/transport code is not a substitute for these checks.

5. Use the recorded reference's exact workload identity. A lean exploratory
   command, only where compatible saved reference runs exist, is:

   ```sh
   bin/spark --cluster-config experiments/2026-11-05-drafter-metadata-replay/candidate.json bench --suites quality,decode --decode-cases prose,code,prose-nothink,code-nothink --concurrency 1,8 --min-samples 3 --max-samples 3
   ```

   Keep all samples and both boots, memory/swap/thermals, startup, step time,
   acceptance, aggregate/per-user decode, TTFT, TPOT and ITL where recorded.
   Include cold/mixed prefill and first-chunk waits in the subsequent full
   matrix. Compare against saved 1030, C36 and the previous compatible candidate
   with signed deltas and separate repeat margins. Do not rerun released builds,
   mix lean/full protocols, infer missing repeats or treat the old host-gap bound
   as a contemporary baseline. No comparable saved reference means inconclusive.

A repeatable serving gain beyond both repeat margins is required. A component
win alone does not qualify the change. Owner acceptance, configuration/docs and
a new immutable baseline are separate promotion work and are not authorized here.
