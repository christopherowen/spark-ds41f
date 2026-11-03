# AgentX on tp3-switched: SemiAnalysis's agent-trace replay on candidate-37

Owner, 2026-11-02: run AgentX. This is its first window. The harness has only
ever been installed and stub-tested on dgx1 (`experiments/2026-10-25-agentx`).

**Status:** prepared, not run.

- Build: candidate-37 at tp3-switched, the profile sp5–sp7 serve
  (`config/candidate-37-tp3-switched-dev.local.json`), image
  `spark-ds41f:candidate-37` (`sha256:e1ee7b1953b1…`).
  [arm-agentx.json](arm-agentx.json) serves it on port 8001, so no production
  client joins the closed loop. Nothing else changes.
- Workload: agentx-harness 1.0.7 (`requirements/agentx-harness.txt`, hashed),
  dataset `semianalysis_cc_traces_weka_062126` at revision `23f152f6`, seed 42.
- [lab-agentx.json](lab-agentx.json), run `agentx-c37-tp3-switched`: points 4
  and 8 (TP3 serves 8 sequences), each a fresh boot, a 15 s clean check, then
  3,600 s measured. About 2.5 to 3 hours with the in-window prepare.
- The arm sets `host.home` to `/home/chris`. The lab's `agentx` job
  (`agentx_target` in `spark_ds41f/lab.py`) reads the profile without the node map, so
  it misses `read_cluster`'s rewrite of the recipe's `/home/swank` to
  `/home/<ssh_user>`, which `bin/spark` gets. Without it the job looks for the hub cache
  under `/home/swank` on sp5, where it doesn't exist. That's a lab bug, for the lab's
  own fix.

## Setup on sp5, between windows (no hold)

1. `uv`: sp5 has none. dgx1's `~/.local/bin/uv` (uv 0.9.17, aarch64, the one
   the dgx1 install used) is copied to sp5's `~/.local/bin/uv`, and its SHA-256
   is checked against dgx1's.
2. `scripts/agentx.py install`: a Python 3.12 venv at
   `~/spark-lab/agentx/venv-1.0.7`, with the hashed requirements only.
3. `scripts/agentx.py fetch --tokenizer deepseek-ai/DeepSeek-V4.1-Flash-1017
   --tokenizer-revision 1bae0e1088993b9b82de08f59de5be5a34e42170
   --hf-hub-cache ~/.cache/huggingface/hub --avoid-cpus 5-9,15-19`: the
   1.72 GiB traces and the datasets cache.
4. `scripts/lab.py run experiments/2026-11-02-agentx-tp3-switched/lab-agentx.json --dry-run`
   on sp5 reports nothing missing.

The job's first step stops serving and runs `prepare --in-window`. That is the
tokenization pass, about 15 minutes the first time.

## Results

Pending the window.
