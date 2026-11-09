# Release benchmark

How a candidate build is shown to be correct and no slower than the release it replaces,
with the commands to run each step. Run everything from the repository root on the head
node. [testing.md](testing.md) explains what each layer proves, [lab.md](lab.md) how lab
windows and jobs work, and [methodology.md](methodology.md#promotion) what a promotion
commit records.

| Step | What it proves | Where | Time (TP4) |
| --- | --- | --- | --- |
| 1. Release tests | the serving code paths, edge cases included | the candidate image, one GB10 | about 8 min |
| 2. Determinism gate | outputs identical to the previous release | the cluster, recipe shape | about 10 min |
| 3. Release comparison | no metric slower than the previous release | the cluster, recipe shape | about 4 h |

A candidate needs every step. A change that alters the output text (any precision change)
fails step 2 by construction and needs its own correctness evidence against DeepSeek's
reference; its speed is then judged on distinct prompts, because the bench's text changes
with it ([lab.md](lab.md#release-comparison)).

## Candidates

A candidate is the latest release plus a selection of PRs. Several candidates are tested
at once, one per testbed. Every PR that passes is carried into all later candidates, and a
PR that misses goes to rework ([methodology](methodology.md#round-purpose-report-and-drop-rule)).
Each round follows the steps below. Candidates 47-49 are the worked example
([experiments/2026-10-26-spark-ds41f-candidate](../experiments/2026-10-26-spark-ds41f-candidate/README.md)).

Owner rules for these rounds (2026-11-09):
- Testing on the dgx and spaceport nodes needs no further approval, and neither does
  merging the round's own PRs. The hold-file rules in AGENTS.md still apply.
- Builds never run on the owner's laptop.
- Every test result is kept.

### Testbeds

| Testbed | Nodes | Use |
| --- | --- | --- |
| `tp4-ring` | dgx1..dgx4 | Production, so a window takes it down: `scripts/lab_idle_start.sh 10 <log>` on dgx1 waits for 10 idle minutes first. Image builds and release tests run here. |
| `tp4-switched` | sp1..sp4 (head sp1) | A candidate's gate and screen |
| `tp3-switched` | sp5..sp7 (head sp5) | A candidate's gate and screen |
| one GPU | sp8 | Each PR's own benchmark. Several jobs share the GPU through `flock /tmp/spark-lab-gpu.lock`. |

- `tp3-triangle` (dgx1..dgx3) needs the dgx fabric re-cabled. The merged candidate qualifies there too.
- Each spaceport head names its shape once (`bin/spark site shape <shape> --apply`, then `cluster sync`).
- The release serves on each spaceport testbed, so a window has a profile to restore.

### 1. Select

- Fetch each open PR's head (`git fetch origin pull/<N>/head`).
- Apply each PR's patches alone on the base tree. Then, for every pair that touches a common
  file, apply one and check the other.
- A candidate holds no conflicting pair. The candidates of one round overlap at most in test files.
- The base is the release's series plus every PR that has passed so far.

### 2. Assemble

The folder is `experiments/2026-10-26-spark-ds41f-candidate/candidate-<N>/`.

- `vllm/series`: the base series (relative paths), then each PR's patches copied from its
  head and renumbered in application order. `receipts/selection.json` records each patch's
  PR, head and SHA-256; `receipts/stack.log` records the stack.
- **Cluster configurations:** one for every shape of `releases/<release>/cluster.json`, each
  with its `-dev` file. Each extends the release recipe at its shape and sets only the image
  `spark-ds41f:candidate-<N>`, its tree labels and the lock. They let anyone run the
  candidate on their own setup; only the testbed's is measured.
- `image-dgx4.json`: a build-only profile that extends a dgx shape.
- **Controls for pure-Python PRs:** `<shape>-no<PR>.json` mounts the base's copies of the PR's
  files from `no<PR>/overlay/` over the same image. Two conditions hold first: no other patch
  in the candidate touches those files, and the PR's patch applies cleanly to the overlay
  files (`git apply --check`).
- `README.md`: the PRs, the testbed, the runs, and each PR's mark, declared before building.
- **Specs:**
  - `build.json` (sync, image, kernel), or `candidate-<shape>.json`, which adds the gate;
  - `release-<shape>.json`.

### 3. Lock and check (on a dgx node)

Work in a checkout outside the deployment one (`~/spark-lab/agents/candidate-<N>/spark-ds41f`),
at the pushed commit:

```bash
bin/spark --cluster-config <dir>/image-dgx4.json build prepare --only vllm --lock
```

- `--lock` writes `patch_head`, `expected_tree` and `patchset_sha256` into `source.json`. Copy
  them into `upstreams.lock.json`, as 1107's lock carries them.
- Run `scripts/release_tests.py check <prepared tree> <base revision> HEAD <dir>/release-tests.json`,
  starting from the base release's classification, and classify every new or changed test
  file. Then run `gate` and `bundle`.
- Run `bin/spark doctor` on every configuration, the unit suite on Linux, and `git diff --check`.
- Merge the folder to `main`. The lab opens a window only on a published commit.

### 4. Build and release tests (dgx window)

- **Build:** the `image` job builds each candidate on dgx4 and copies it along the ring.
- **Release tests:** ONE `kernel` job runs every candidate's release-test bundle at once, each
  on its own node. A bundle takes about 45-50 minutes.
- **Spaceport copies:** dgx does not reach spaceport, so stream each spaceport candidate's
  image to its testbed head:
  ```bash
  ssh dgx1 docker save spark-ds41f:candidate-<N> | ssh <head> docker load
  ```
  Then run `bin/spark --cluster-config <dir>/<shape>.json build distribute --apply` from the
  head, and check that every node shows the same image ID.

### 5. Gate first

Each testbed runs its candidate's gate before anything else. A startup failure then costs
one boot instead of a release run.

### 6. Screen

`release-<shape>.json` runs the release suite (`release-tp3` at TP3) with `single-stream`, on
two candidate boots.
- **Reference:** the base's recorded runs (`"reference": "release"`). Where no suite record
  exists at that shape, boot the previous release as the middle arm, and keep its run as the
  reference for later candidates.
- **`sampled`:** add it only when a PR's mark depends on temperature > 0 or on drafting.

### 7. Own marks

- For every PR, re-run its own evidence (its README's commands) on the candidate image on sp8,
  two boots each. Use the base image as the baseline where the PR compared against it.
- Receipts go in `receipts/own-marks/pr-<N>/`, with a `result.json`.
- One sub-agent per candidate does this.

### 8. Verdicts, attribution and rework

- **Correctness fix:** passes when its functional evidence passes and no core metric is worse.
- **Performance change:** passes when it improves its own metric and no unrelated metric is
  worse. A slower cell of its own bench, on a shape serving uses, counts as worse.
- **Attribution:** when a core metric is worse in the release table, remove suspects one at a
  time with `no<PR>` controls on the same image. Measure only the step that moved: a suite
  file with that step alone, as in `candidate-48/grid-suite.json`.
- **Rework:** a miss goes to a sub-agent. It writes to `receipts/rework/pr-<N>-r<K>/`: the
  analysis, the corrected patch, the commands, the raw outputs and `result.json`. Then apply
  the drop rule.
- **Receipts:** keep every one, unfavourable runs included. `experiments/**/receipts/**` is
  exempt from whitespace checks, so receipts stay byte-for-byte.

### 9. Merge the winners

- **The next candidate** is the base plus every PR that passed, including every reworked PR that meets its mark.
- **Every shape:** it is qualified on every shape the release serves, `tp3-triangle` included.
- **Every testing layer** of [testing.md](testing.md):
  - the release tests;
  - the determinism gate;
  - cold boots;
  - the full release set (`release`, `single-stream`, `sampled`);
  - the quality evals the previous release recorded (LongBench v2, RULER, EvalPlus HumanEval, Vision Eval, tool-eval-bench, the 960K needle, a 1-hour AgentX run);
  - the live functional checks of its fixes.
- **Then promotion** (section 4).

## Arms

Each build under test is an arm: a cluster configuration that extends the recipe. A release
comparison needs a development arm of each build, because the release suite's engine grid
times the engine through vLLM's dev-mode collective RPC, which production does not expose.
A development arm sets one variable and changes nothing the engine computes:

```json
{
 "notes": "<candidate> with the dev-mode collective RPC, for the release suite's engine grid.",
 "extends": {"config": "<experiment>/<candidate>.json"},
 "set": {"environment": {"VLLM_SERVER_DEV_MODE": "1"}}
}
```

`bin/spark bench --suite` refuses a suite with a grid step on an arm without it. A
candidate arm should extend the production recipe at its shape (`"extends": {"config":
"config/cluster.json", "shape": "tp4-ring"}`) and record its upstream lock: the lab refuses
an arm whose recipe and lock name different baselines. Example:
[experiments/2026-10-18-r6x](../experiments/2026-10-18-r6x/README.md).

## 1. Release tests

Select the tests for the candidate's vLLM patch series and check that every selected file
is classified in [config/release-tests.json](../config/release-tests.json):

```bash
python3 scripts/release_tests.py check <vllm-checkout> <base-revision> HEAD
```

`<vllm-checkout>` is the vLLM tree with the series applied (as `bin/spark build prepare`
leaves it) and `<base-revision>` the series' pinned base. A new or changed test file must be
classified (gate, periodic or excluded, with its reason) before the release goes on. Then
write the gate manifest and the kernel-lab bundle that runs it in the candidate image:

```bash
python3 scripts/release_tests.py gate <vllm-checkout> HEAD <experiment>/release-tests-gate-manifest.json
```

```bash
python3 scripts/release_tests.py bundle <experiment>/release-tests-gate-manifest.json <candidate-image> <experiment>/bundles/release-tests <label>
```

Run the bundle as a lab `kernel` job (the cluster is stopped while it runs):

```json
{"experiment": "<experiment>", "run": "<label>-release-tests",
 "jobs": [{"kind": "sync"},
          {"kind": "kernel", "bundles": [{"node": "dgx2", "bundle": "<experiment>/bundles/release-tests"}]}]}
```

```bash
python3 scripts/lab.py queue add <experiment>/release-tests.json
```

Every test must pass or skip; per-test outcomes land under `/c/release-tests/<label>` on the
node and the verdict in `results/private/lab/<run>/verdicts.json`.

## 2. Determinism gate

Boot the candidate at the recipe shape and compare its outputs with the previous release's:

```json
{"experiment": "<experiment>", "run": "<candidate>-gate",
 "jobs": [{"kind": "sync"},
          {"kind": "measure", "bench": false, "bracket": false,
           "extras": [
             ["experiments/2026-10-12-r6g/token_determinism_text.py", ["--tokens", "192", "--long-lines", "700", "--wide"], "determinism"],
             ["experiments/2026-10-12-r6g/token_determinism_text.py", ["--tokens", "192", "--long-lines", "700", "--wide", "--no-logprobs"], "greedy"],
             ["scripts/distinct_streams.py", ["--streams", "16", "--samples", "4", "--tokens", "256"], "c16-distinct"],
             ["scripts/distinct_streams.py", ["--streams", "8", "--samples", "4", "--tokens", "256"], "c8-distinct"],
             ["scripts/distinct_streams.py", ["--streams", "4", "--samples", "4", "--tokens", "256"], "c4-distinct"]],
           "arms": [{"config": "<candidate>.json", "label": "<candidate>"}]}]}
```

```bash
python3 scripts/lab.py queue add <experiment>/gate-tp4-1017.json
```

Without the lab, the same scripts run against the live service (`http://<head>:8000`):

```bash
python3 experiments/2026-10-12-r6g/token_determinism_text.py http://<head>:8000 --tokens 192 --long-lines 700 --wide
```

```bash
python3 experiments/2026-10-12-r6g/token_determinism_text.py http://<head>:8000 --tokens 192 --long-lines 700 --wide --no-logprobs
```

```bash
python3 scripts/distinct_streams.py http://<head>:8000 --streams 16 --samples 4 --tokens 256
```

Pass: `identical: true` with zero token and log-probability differences in every scenario
(c8, staggered, mixed, cached, again, mixed15), and the four output hashes equal to the
previous release's at the same shape.

- **Recorded hashes are out of date.** The 1017 checkpoint's hashes recorded here
  (`3eab5112` / `db25fde3` with log probabilities, `c9cf9dbe` / `c6bb4f4e` as text) are no
  longer current. Candidates 47 and 48 on spark-ds41f-1107, neither declaring an output
  change, give `8e46794a` / `11ec073c` and `8b549cc4` / `12fd8257` at both tp4-ring and
  tp4-switched.
- **Missing hashes.** Where the previous release's hashes at a shape are not recorded, the
  identity check is the release suite's model step: greedy responses identical (256/256)
  against it.

The distinct-stream throughputs are a first look only; step 3 decides speed.

## 3. Release comparison

The candidate boots twice, back to back, and is compared with the previous release's
recorded runs (owner, 2026-10-29): a released build's numbers never change, so it is not
measured again. The record is the shape's `benchmark_reference` in the recipe
(`manifests/suites/<release>-<shape>`, its companions beside it as `-single-stream` and
`-sampled`). The candidate's repeat bounds its boot-to-boot spread, and a change counts only
beyond its interval plus that spread. The previous release is measured again, as the middle
of three arms, only when its record cannot serve: a suite changed since it was recorded (a
changed suite is a new suite, and the lab refuses the record), or the record predates
servebench's request tables (1025 and earlier), which leaves only the window verdict and not
the paired decode verdict.

### Suites

| Suite | Steps | Time a build |
| --- | --- | --- |
| `release` (TP4) | model behaviour on 256 prompts at 16 streams; the engine grid (20 fixed decode shapes, 100 replays); closed loops at 16 and 8 streams; Poisson arrivals at 0.7 requests/s with goodput under 2 s first token and 100 ms a token; decode stalls during a 131K prefill at 3, 11 and 15 streams (3 and 11 with short probes' first-token wait), three repeats; real-text prefill at 32K, 128K, 256K, 512K and 1,000,000 tokens, three repeats | about 60 min |
| `release-tp3` (TP3) | the same at three nodes: 8 streams, 512K context, grid up to 8 requests, closed loops at 8 and 4, Poisson at 0.42 requests/s, stalls at 3, 6 and 7 streams | about 61 min |
| `single-stream` | one conversation at a time on distinct prompts, 300 s measured | about 6 min |
| `sampled` | closed loops at 16, 8 and 1 stream at the recipe's temperature 1 and top-p 1, fixed per-request seed | about 18 min |

Every release runs its shape's release suite with `single-stream` as a companion on the
same boot; r6x also ran `sampled` (added 2026-10-18), which the example below includes. The release and single-stream suites decode greedily; only
`sampled` measures draft acceptance under sampling. Suite definitions are data in
[servebench/suites/](../servebench/suites/); a changed suite is a new suite.

### With the lab (the usual way)

```json
{"experiment": "<experiment>", "run": "<candidate>-release",
 "jobs": [{"kind": "sync"},
          {"kind": "measure", "suite": ["release", "single-stream", "sampled"], "reference": "release",
           "arms": [{"config": "<candidate>-dev.json", "label": "<candidate>"},
                    {"config": "<candidate>-dev.json", "label": "<candidate>-again"}]}]}
```

`"reference": "release"` follows the first arm's `extends` chain to the recipe shape it is
built on and reads that shape's `benchmark_reference`; `"reference"` may also name a recorded
suite run directly. The plan refuses a record of another suite definition, a missing record,
and a candidate measured only once. Without `"reference"`, list the previous release's arm
between the candidate's two boots, as before.

```bash
python3 scripts/lab.py queue add <experiment>/release-tp4-1017.json
```

Each arm boots, then runs every suite into `results/private/suites/<run>/<label>` (companions
into `results/private/suites/<run>-<suite>/<label>`). At the end the lab writes one table a
suite: `results/private/lab/<run>-table.txt`, `<run>-single-stream-table.txt` and
`<run>-sampled-table.txt`. To print a table again:

```bash
python3 scripts/servebench_suite_table.py <run> <candidate> manifests/suites/<release>-<shape> <candidate>-again
```

A label with a `/` is a recorded suite run (repository-relative); with a measured previous
release, give its label instead.

### By hand

The lab job is this sequence. For each arm in turn (candidate, previous, candidate again),
start it:

```bash
bin/spark --cluster-config <arm>-dev.json cluster start --replace --change-shape --apply
```

then run the three suites against it:

```bash
bin/spark --cluster-config <arm>-dev.json bench --suite release --compare none --label <label> --output results/private/suites/<run>/<label>
```

```bash
bin/spark --cluster-config <arm>-dev.json bench --suite single-stream --compare none --label <label> --output results/private/suites/<run>-single-stream/<label>
```

```bash
bin/spark --cluster-config <arm>-dev.json bench --suite sampled --compare none --label <label> --output results/private/suites/<run>-sampled/<label>
```

For the second and third arms, add `--corpus results/private/suites/<run>/<candidate>/model.json.gz`
to the release suite, so their model behaviour is scored on the first arm's text. Then
print each table with `scripts/servebench_suite_table.py` as above (`<run>`,
`<run>-single-stream`, `<run>-sampled`). `bin/spark bench` cools every node first, runs the
fan performance curve, watches memory and temperature (a breach stops the run) and records
the deployment's identity in the run; it refuses a cluster that differs from its arm.

### Reading the table

The first block gives the verdicts: previous release to candidate, each headline metric
judged against the candidate's own repeat. **better** or **worse** means the change exceeds
its 95% interval plus the repeat's own change; **same** means it does not.

The closed loops' verdict row is **decode tok/s, paired**. Each request is paired with the
same prompt on the other build, and the row gives the median ratio over both candidate
boots. Its margin is the interval over requests and a boot floor in quadrature. The floor
is the recorded A/A bound (`AA_FLOOR_PCT` in the table script: 0.35% at one stream), or the
repeat when larger, shrunk by sqrt(3/4) for two candidate boots. The window row
(output tok/s) stays beside it, with its old interval. Together they resolve changes of
about 0.3% at one stream, where the window row's ±3.7% resolved none
([experiments/2026-10-26-paired-verdict](../experiments/2026-10-26-paired-verdict/README.md)).
A rows-a-step block follows: for each arm and closed loop, tokens and verified rows a
step, row yield, wasted rows, and survival and coverage by draft position. The blocks after
it show every metric of every step for each pair.

- **Engine grid:** step time at fixed shapes with no generated text, draft acceptance or
  scheduling in it: the engine's speed alone. Lower is better.
- **Closed loops:** output tok/s = engine steps/s x output tokens a step. Steps per second is
  the engine; tokens a step moves with draft acceptance, that is with the text.
- **Poisson:** output tok/s equals the offered load whenever the server keeps up, so it reads
  same by construction; its latency and goodput rows are the informative ones.
- **Rows marked "no interval"** (latency percentiles) are judged against a 3% tolerance and
  are direction only.
- **Model behaviour:** greedy responses identical, teacher-forced distributions (KL,
  top-1 agreement, perplexity) and acceptance by draft position.

Pass: no verdict worse than the previous release. Promote on the comparison; do not
re-benchmark after it.

## 4. Promotion

The owner decides. [methodology.md](methodology.md#promotion) lists what the promotion
commit records; for the benchmark, the candidate arm's release-suite run (every step's
output except the model step's text, with the release table) becomes the shape's
`benchmark_reference` under `manifests/suites/<baseline>-<shape>/`, and the companion runs
beside it (`<baseline>-<shape>-single-stream`).
