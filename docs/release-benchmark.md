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

A candidate is the latest release plus a selection of PRs. The work has two phases
(owner, 2026-11-09):
- **PR validation** (steps 1-8) is quick. It shows whether each PR earns its keep.
- **Release qualification** (step 9) measures the final candidate once, completely, and
  produces the benchmark data for the release notes and announcement.
- **Freezing** (between rounds) publishes each verified merged base as a pre-release for
  external users, and makes it the next round's base ([Frozen candidates](#frozen-candidates-pre-releases),
  owner, 2026-11-10).

AgentX, quality evals, cold boots and the full release set belong to release qualification
only. Several candidates are tested at once, one per testbed. Every PR that passes is carried into all later candidates, and a
PR that misses goes to rework ([methodology](methodology.md#round-purpose-report-and-drop-rule)).
Each round follows the steps below. Candidates 47-49 are the worked example
([experiments/2026-10-26-spark-ds41f-candidate](../experiments/2026-10-26-spark-ds41f-candidate/README.md)).
The steps that drive the testbeds (queueing, waiting, image copies, gates, verdicts, receipts,
controls, closing a window) are `bin/spark candidate` commands, run from the owner's laptop
([lab.md](lab.md#candidate-tooling)).

Owner rules for these rounds (2026-11-09):
- Testing on the dgx and spaceport nodes needs no further approval, and neither does
  merging the round's own PRs. The hold-file rules in AGENTS.md still apply.
- Every merge of the round's PRs follows a passing unit-suite run on a dgx node (step 3).
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
- Take only the patches that act on this deployment, DeepSeek V4.1 Flash on GB10. A patch for a
  platform or path it does not serve (ROCm, another model, a feature the recipe leaves off) is left
  out. Split a PR whose patches mix the two, and record each patch left out, with its reason, in
  `receipts/selection.json` (owner, 2026-11-09: candidate-48 carried #1011's ROCm-only patch and
  #1013's prefill lanes, and candidate-50 left both out).
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
  - `screen-<shape>.json` (step 6). `release-<shape>.json` is for release qualification (step 9).

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
- **Unit suite on a dgx node.** Run it in that checkout, at the commit being merged:
  ```bash
  python3 -m unittest discover -s tests -p "test_*.py"
  ```
  - A fresh clone lacks the node's git-ignored site maps. Copy `config/nodes.json`,
    `config/nodes-*.local.json` and `config/site.json` from the node's deployment checkout
    first. Without them, the tests that resolve a configuration (`test_agentx`,
    `test_candidate_15`..`18`) stop on a missing node map.
  - It must pass in full there.
  - A run on the owner's laptop (macOS) does not count. Some tests need Linux: the sandboxed
    model-output parser and the CPU-affinity build sizing.
  - Use a node that is not part of a running measurement. Test load on a measured node
    perturbs the bench. When every dgx node is measuring, wait for a gap between jobs or for
    the window to close.
- Run `bin/spark doctor` on every configuration, and `git diff --check`.
- Merge the folder to `main`. The lab opens a window only on a published commit.

### 4. Build and release tests (dgx window)

- **Build:** the `image` job builds each candidate on dgx4 and copies it along the ring.
- **Release tests:** ONE `kernel` job runs every candidate's release-test bundle at once, each
  on its own node. A bundle takes about 45-50 minutes.
- **Spaceport copies:** dgx does not reach spaceport, so each image crosses through the owner's
  laptop once, to sp1, and fans out over the spaceport fabric:
  ```bash
  ssh dgx1 docker save spark-ds41f:candidate-<N> | ssh sp1 docker load
  ```
  - Stream one image at a time. On 2026-11-09, two parallel streams ran at about 2.5 MB/s each,
    and each 10.5 GB image took 108 minutes.
  - A candidate for `tp3-switched` goes on from sp1 to its head, over the fabric:
    `ssh sp1 'docker save spark-ds41f:candidate-<N> | ssh sp5 docker load'`.
  - On each head, run `bin/spark --cluster-config <dir>/<shape>.json build distribute --apply`, and
    check that every node shows the same image ID. Over the fabric this took 4–6 minutes for
    3–4 ranks.
  - `bin/spark candidate ship spark-ds41f:candidate-<N> --from dgx1 --to tp4-switched tp3-switched --apply`
    does all of this, resumably, and verifies the IDs. With `--queue <testbed>=<spec>` it then
    queues the testbed's specs.
- **Production first:** a spaceport window opens only with its production profile live
  (`doctor --live`). If the testbed was stopped, start 1107 there first:
  `bin/spark --cluster-config config/cluster.json cluster start --replace --apply`, after reading
  the plan.

### 5. Gate first

Each testbed runs its candidate's gate before anything else. A startup failure then costs
one boot instead of a release run.
- Queue it with `bin/spark candidate queue <testbed> <dir>/candidate-<shape>.json --apply`.
- Read it with `bin/spark candidate gate <testbed> candidate-<N>-<shape> --expect <ref>/<long>`.
  `--expect` takes the log-probability run's hashes and `--expect-text` the text hashes.

### 6. Screen

`screen-<shape>.json` runs the `screen` suite on two candidate boots: `screen` at TP4,
`screen-tp3` at TP3, about 18 minutes a boot. It covers model behaviour (greedy identity
against the reference), the engine grid, a closed loop (16 streams at TP4, 8 at TP3) and
prefill at 32K and 256K.
- `bin/spark candidate wait` returns when a queued job finishes.
- `bin/spark candidate verdicts <testbed> <run>` prints each table's verdict block and the
  arms' headline numbers from the window log.
- **Reference:** the base release's `screen` run at the same shape.
  - Record it once per testbed shape: two boots of the release, as the first job of the
    round that first needs it.
  - Keep it in that candidate's `receipts/references/<release>-<shape>-screen/` (companions
    beside it as `-<suite>`), and name it in later screens' `"reference"`.
  - Measuring the release here is allowed because no record of this suite exists; it isn't
    re-measuring a recorded number.
  - The release's `release`-suite records serve release qualification, not screens.
- **Companions:** add a suite only when a PR's mark needs a step the screen lacks:
  - `single-stream` for one-stream decode;
  - `sampled` for temperature > 0 or drafting;
  - `release` (with its own reference) for the stall rows, such as TTFT during a long prefill.

### 7. Own marks

- For every PR, re-run its own evidence (its README's commands) on the candidate image on sp8,
  two boots each. Use the base image as the baseline where the PR compared against it.
- **Run the baseline in the same session, interleaved (ABBA).** Microbenchmark spreads between
  sessions are larger than one session's interval. A candidate compared with baseline numbers
  recorded in another session shows "slower" cells that re-running unchanged code also shows
  (#996 round 2). The baseline arm of a PR's kernel A/B is not a release record, so the rule
  against re-measuring a release does not apply to it. Reworks follow the same rule.
- **Judge on the path serving takes.** Decode kernels run from CUDA graphs, so for them CUDA-graph
  replay or kernel time decides. A loop of eager launches measures the launch cost, not the kernel
  (#991 round 1). Record both.
- Receipts go in `receipts/own-marks/pr-<N>/`, with a `result.json`.
- One sub-agent per candidate does this.

### 8. Verdicts, attribution and rework

- **Correctness fix:** passes when its functional evidence passes and no core metric is worse.
- **Performance change:** passes when it improves its own metric and no unrelated metric is
  worse. A slower cell of its own bench, on a shape serving uses, counts as worse.
- **Attribution:** when a core metric is worse in the release table, remove suspects one at a
  time with `no<PR>` controls on the same image. Measure only the step that moved: a suite
  file with that step alone, as in `candidate-48/grid-suite.json`.
  `bin/spark candidate overlay <dir> no<PR> --tree <host>:<prepared tree> --revert <commit> --apply`
  writes a control's overlay and profiles.
- **Confirm in one session first.** A change near its margin against a recorded reference can be drift between sessions. Before attributing it to a PR, run the base and the candidate ABBA in one session.
  - On 2026-11-09, candidate-53's stall-3 read 2.1% slower against candidate-50's run from earlier the same evening.
    The ABBA in one session showed that this was #984's, not drift: with #984, 2.37% slower; without it, level with
    candidate-50 ([candidate-53/receipts/tp3-switched](../experiments/2026-10-26-spark-ds41f-candidate/candidate-53/receipts/tp3-switched/README.md)).
    The same image measured in different sessions of candidates 50-58 moved no more than its boot-to-boot spread
    (candidate-54's grid: sd 0.23% over 7 boots in 4 sessions at tp3-switched, 0.07% over 5 boots in 3 sessions at
    tp4-switched; its c1 within 0.06% across 3 sessions at tp4-ring).
- **First boots and time to ready.** An image's first boot on a node compiles its changed kernels into the
  host-mounted cache (candidate-58: 273 TileLang kernels on sp1 and 427 on sp5 in the gate's boot, none in the screen's),
  which adds up to about two minutes to its time to ready. The lab marks such boots in the window log, and
  `bin/spark candidate verdicts` leaves them out of its time-to-ready comparison ([lab.md](lab.md#runs)).
- **Rework:** a miss goes to a sub-agent. It writes to `receipts/rework/pr-<N>-r<K>/`: the
  analysis, the corrected patch, the commands, the raw outputs and `result.json`. Then apply
  the drop rule.
- **Receipts:** keep every one, unfavourable runs included. `experiments/**/receipts/**` is
  exempt from whitespace checks, so receipts stay byte-for-byte.
  `bin/spark candidate receipts <testbed> <run> <dir>/receipts/<testbed> --apply` copies a
  run's tables, window-log segment, determinism and suite outputs, and lists the model text it
  leaves on the node in `large-files-on-nodes.tsv`.
- **Closing a window:** `bin/spark candidate close <testbed> --apply` closes it once its queue
  is empty. Never write `~/spark-request.json` to end a queue: the runner reads it between
  steps and stops the running job.

### 9. Release qualification (final benchmarking)

**Between releases, a round's winners merge into the next base.** It runs only when the owner calls a release
(owner, 2026-11-09: "there are dozens more PRs to work through until release").
- **The merged base:** the previous base plus the PRs that passed. It gets its own candidate number
  ([candidate-50](../experiments/2026-10-26-spark-ds41f-candidate/candidate-50/README.md) is the first).
- **Verification only:** the gate, the screen on each testbed (which records the previous release's
  screen reference where none exists), and the live checks of the fixes it carries. No release set,
  quality evals, AgentX or cold boots.
- **Its screens are the next round's reference:** later PRs are judged against the accumulated base.
- **Then it is frozen:** a merged base verified on all three testbeds is frozen and published as a
  pre-release ([Frozen candidates](#frozen-candidates-pre-releases)). The next round's three candidates are
  built on it.

The release candidate is the base plus every PR that passed PR validation, including every
reworked PR that met its mark. It is measured once, completely, on every shape the release
serves, `tp3-triangle` included.

**What runs:**
- every layer of [testing.md](testing.md): the release tests, the determinism gate and cold boots;
- the full release set on two candidate boots, against the previous release's recorded runs:
  `release` (`release-tp3` at TP3), `single-stream` and `sampled`. The promoted runs become
  each shape's `benchmark_reference`;
- the quality evals the previous release recorded: LongBench v2, RULER, EvalPlus HumanEval,
  Vision Eval, tool-eval-bench and the 960K needle;
- a 1-hour AgentX run (the lab's `agentx` job) on every shape the previous release reported;
- the live functional checks of the fixes it carries;
- startup memory floors and boot time.

**What the release notes report**, against the previous release (the format of
[release-notes.md](../experiments/2026-10-26-spark-ds41f-candidate/release-notes.md)):
- paired decode at each concurrency;
- the engine grid;
- prefill from 32K to 1M;
- sampled decode;
- short-request TTFT during a 131K prefill;
- greedy identity;
- the quality scores;
- AgentX throughput and prefix-cache hit rate;
- the startup memory floor.

Then promotion (section 4).

### Frozen candidates (pre-releases)

A merged base is **frozen** once the rework cycle that produced it is done and it has passed verification on all
three testbeds (`tp4-ring`, `tp4-switched`, `tp3-switched`): the gate, the screen, the live checks of the fixes it
carries, and its release tests. It is then supplied to external users as a GitHub pre-release, and the next round's
three candidates, one per testbed, are made with it as their base (owner, 2026-11-10).

Freezing is not promotion. `config/cluster.json` and the release's baseline do not change; promotion still needs
release qualification (step 9) and the owner's acceptance (section 4).

**What a freeze records:**
- **The candidate's folder is frozen.** Its series, lock, profiles and verification receipts never change again. Later
  findings about it go into the next candidate's README, not this one.
- **A freeze manifest:** `manifests/frozen/candidate-<N>.json`, immutable like a baseline; a later freeze adds its own.
  It records:
  - the commit, the tag, the date and the base it was built on (the previous frozen candidate or the release);
  - the lock (`patch_head`, `expected_tree`, `patchset_sha256` for vLLM and every upstream series);
  - the image ID on each testbed and the host that built it;
  - the profile for each shape, and the shapes it was verified on, with each gate's hashes;
  - every PR it carries since its base, with its verdict, and every PR dropped or left out on the way, with the reason;
  - the over/under against the current release, per testbed;
  - known issues: unmeasured metrics, open regressions, and caveats such as single-boot figures.
- **A tag** `spark-ds41f-candidate-<N>` on the commit that adds the manifest. It is never moved or deleted.
- **A GitHub pre-release** on that tag. It is marked pre-release, so the release (for example `spark-ds41f-1107`)
  stays "Latest" and the default. Its notes give:
  - what changed since the base, PR by PR;
  - the over/under table, with the verified shapes;
  - the known issues;
  - how to run it: check out the tag, then `bin/spark build prepare`,
    `bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-<N>/<shape>.json build image --apply`,
    `build distribute --apply` and `cluster start --apply` ([setup](setup.md)).
- **No binary assets:** users build the image from the lock, as for a release. Nothing is pushed to a registry.

**Publishing:**
- The coordinator prepares the manifest and the notes, then asks the owner before creating the tag and the
  pre-release (it is public). Tags, releases and pushes go from the owner's laptop only.
- When the next freeze is published, the previous pre-release's notes gain a line "Superseded by
  `spark-ds41f-candidate-<M>`". Old pre-release pages may be deleted later; their tags stay.

**The next round** selects its PRs against the frozen candidate, builds its three candidates on its series, and
judges them against the frozen candidate's recorded screens.

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
is the recorded A/A bound, or the repeat when larger, shrunk by sqrt(3/4) for two candidate
boots. The window row (output tok/s) stays beside it, with its old interval. Together they
resolve changes of about 0.3% at one stream, where the window row's ±3.7% resolved none
([experiments/2026-10-26-paired-verdict](../experiments/2026-10-26-paired-verdict/README.md)).
The engine grid's **decode geometric mean** is judged the same way, against its own recorded
floor, over both candidate boots; it no longer takes its interval plus one draw of the repeat,
whose margin swung between 0.13% and 0.74% with the luck of that draw.

**Both boots of each.** When the reference has a repeat boot, every headline row is the mean
over all four pairings of the reference's two boots and the candidate's two. The repeat is
found by name: an arm or record called `<reference>-again`, or `<reference>-again-<suite>`
for a companion's record (as in `candidate-54/receipts/references/`); a suite job's
`"reference_repeat"` names it when it is called otherwise. Each side's boot noise is bounded by
the floor or that side's own repeat, whichever is larger, and the boot term is
sqrt(F_candidate^2/2b + F_reference^2/2r) for b and r boots averaged: the floor itself for one
boot of each, x0.866 for two candidate boots, x0.707 for two of each, 18% less than with one
reference boot. Rows without a floor (window throughput, prefill, stall) add the two repeats'
boot term the same way to their interval, the largest of the four pairings'. The verdict line
says which boots were averaged, and each row says "both boots of each" with the reference's
repeat. Without a repeat the reference is one boot and rows are judged as before.

**Floors.** `scripts/servebench_suite_table.py` holds them with their provenance:
`AA_FLOOR_PCT` (closed loops, from the pairs of 2026-10-15..25: 0.35% at c1, 0.27% c4, 0.39% c8,
0.43% c16), `GRID_FLOOR_PCT` (0.43% for an engine grid's geometric mean, from 31 same-build
pairs of candidates 50-58) and `FLOORS` by testbed and suite, from the same-build pairs of
candidates 50-58 (2026-11-09..10). A testbed's floor replaces the default with at least 4
pairs and goes below it only with at least 6; otherwise the default stays, and the table's
`floors (...)` line says what the pairs showed.

| Testbed, suite | Floors | Kept at the default |
| --- | --- | --- |
| tp4-ring, screen | grid 0.37%, c16 0.39% (6 pairs each) | |
| tp4-switched, screen | grid 0.33%, c16 0.34% (7 pairs each) | |
| tp4-switched, grid | grid 0.41% (6 pairs) | |
| tp3-switched, grid-tp3 | grid 0.60% (7 pairs) | |
| tp3-switched, sampled | c16 0.96% (4 pairs) | c8 (3 pairs, 0.85%), c1 (2 pairs, 1.22%) |
| tp3-switched, screen-tp3 | | grid (5 pairs, 0.32%), c8 (3 pairs, 0.52%) |
| tp4-ring, single-stream | | c1 (4 pairs, 0.25%) |
| tp3-switched, single-stream | | c1 (4 pairs, 0.32%) |

The testbed is read from the candidate run's recorded identity (node count and transport).
Re-reading receipts: `--suites-dir <receipts>/suites` reads the arms from a copied run;
`--no-reference-repeat` judges against one reference boot.
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
