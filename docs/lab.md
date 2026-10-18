# Experiment windows and the kernel lab

`scripts/lab.py` runs experiments on the Spark cluster without restoring the promoted
service after every run. It replaces the copy-and-edit `runNN.sh` pattern.

## Windows

```sh
scripts/lab.py window open --note "mHC screen"
scripts/lab.py window status
scripts/lab.py window close
```

`open` refuses unless:

- the deployment commit is on `origin/main` and every node's checkout matches;
- `doctor --live` passes for the production profile, so the profile the window will
  restore is the one serving;
- the cluster has served no requests for 30 s;
- no one else holds `~/spark-hold.json`.

It then writes the hold file and starts a watchdog. The hold file records the production
profile (`production_config`), a heartbeat that the runner refreshes every minute, and
`expected_end`: null, because a window has no time cap unless `--minutes M` asks for one. The watchdog restores the production profile and removes the hold if the
heartbeat goes stale for 15 minutes.

`close` boots the production profile (unless it is already live), checks
`doctor --live` against it, and removes the hold.

A window closes itself:

- when its run completes, unless the run uses `--keep-open`;
- when someone writes `~/spark-request.json`. The runner finishes the current job first;
- at a time cap, only if one was given (`--minutes M`).

No clock cuts a job short. A queue runner keeps the window held while its queue is
empty, ready for the next spec, until `window close` or a request (`--idle-minutes M`
closes it after M idle minutes instead). A failed job costs only itself; the queue
continues.

### Production profile

The production profile is the cluster configuration that a window restores. Every
cluster action in the window uses its node map: stop, sync, `doctor --live`, the publish
guard and the node-map check before each boot.

- `--production-config PATH` selects it on `window open`, `window close`, `run` and
  `queue run`. The default is the recipe `config/cluster.json` at the site's shape
  (`bin/spark site shape`), which is right whatever the cabling. Name another file only
  to restore an archived release, such as a shape's previous production record while
  the recipe's image is not yet qualified on it.
- `open` writes it to the hold file as `production_config`. The commands that follow
  use that record: `close`, the watchdog, runs inside the window and a queue runner
  restarted by a sync job.
- A `--production-config` that differs from the open window's record is refused.
- A window opened before the field existed has no record; it uses the option or the
  default.

The other hold fields are unchanged ([AGENTS.md](../AGENTS.md), Shared cluster windows).
Holds written by other agents are never read for a profile.

## Runs

```sh
scripts/lab.py run scripts/lab_specs/lab1-first-window.json --dry-run
scripts/lab.py run scripts/lab_specs/lab1-first-window.json
```

**Arms are overlays.** An arm configuration names what it extends and holds only what it
changes, so it cannot drift from what it is compared with:

```json
{
  "extends": {"config": "config/cluster.json", "shape": "tp4-ring"},
  "set": {"container": {"image": "vllm-ds41f-kkref:...-r6m"}, "environment": {"NAME": "value"}},
  "unset": ["environment.OLD_NAME"]
}
```

`extends` may name another arm (a dev arm extends its release arm); an arm of the recipe
must name its shape. `set` merges into the base, `unset` removes dotted paths (an unset
the base does not have is an error). An arm never carries the recipe's shape or
qualifying benchmark. Older experiments keep their complete files.

A window never opens while production serves requests (`queue run` refuses a busy cluster).
To queue work for when the production agent finishes, start
`scripts/lab_idle_start.sh [MINUTES] [LOG_NAME]` detached on the head node: it starts the
queue runner only after MINUTES (default 10) consecutive idle minutes (no running or
waiting request and no request finished), so a window never takes the cluster in a pause
between an agent's requests. An arm that is not production's model (another checkpoint,
speculation off) should serve on another port (`serve_args` and `deployment.ready_url`),
so an agent returning mid-window cannot reach it.

Over ssh, detach a long `run` or `queue run` from the session: use `setsid -f` with every
stream redirected, or run it inside tmux. `lab.py` logs to stdout. If stdout is the session's
pipe and the session ends, the next log line fails with a broken pipe and the runner exits.
The heartbeat then stops and the watchdog closes the window 15 minutes later.

```sh
ssh dgx1 'cd ~/projects/spark-ds41f && setsid -f scripts/lab.py run SPEC.json \
  > results/private/lab/SPEC.log 2>&1 < /dev/null'
```

A spec names the experiment directory, a run name and a list of jobs. A run opens a
window if none of ours is open.

Specs are checked before anything boots:
- every arm configuration and script must exist;
- the arguments of servebench extras (`scripts/servebench_run.py`, `_engine.py`,
  `_model.py`) must parse with servebench's own parser.

`queue add` refuses a spec that fails. One whose jobs start with a `sync` is only warned
about, because its files may arrive with the sync; the run checks again after the sync,
before the first boot. `run --dry-run` prints the problems.

- **`measure`:** boots each arm with `cluster start --replace --change-shape`, with no
  stop in between, and measures it. The arm's node map must equal production's (below),
  so an arm can change context or KV size but never the node set.
  - The bench checks the live engine against the arm before it measures. A bench that
    exits non-zero (a cluster that differs from the arm, a failed quality gate, failed
    requests, an aborted run) ends the job before its scripts run.
  - Profile `lean`, about 2.5 min per arm:
    - bench prose and JSON with 3 samples each;
    - 1K and 16K prefill, 2 repeats each;
    - 12 distinct single-stream prompts;
    - 8 streams with 2 samples;
    - short prompts with 3 reps;
    - 3 rounds of mixed traffic.
  - Profile `full`: the round-16 matrix.
  - `"bracket": true` measures the first arm again at the end, labelled `<label>-end`, to show drift.
  - `"extras"` replaces the profile's workloads: `[script, [arguments], output stem]` each. A
    bare script name is the experiment's own; a path such as `scripts/distinct_streams.py` is
    repository-relative.
  - Results land where `tables_arms.py` reads them: `results/private/bench/<run>-<label>/` and
    `results/private/determinism/<run>/<kind>-<label>.jsonl`.
- **Sweeps:** `scripts/lab_sweep.py` writes a `measure` job for one environment variable:
  an arm config per value under the experiment directory (the base profile with only that
  variable changed) and a run spec with the base profile first and bracketed. Each arm runs
  the profile's bench and `scripts/distinct_streams.py` at the stream counts given
  (distinct prompts, so concurrent streams do not share routing). Commit the experiment and
  sync before running it.

  ```sh
  scripts/lab_sweep.py --base config/cluster.json --shape tp4-ring --variable VLLM_L2_PREFETCH_GRID \
    --values 1,2,3,6 --experiment experiments/<dir> --run <name> --streams 1,4,16
  ```
- **`validate`:** a trace arm.
  - Sequence: boot A, `c8_trace.py`, `scenario_trace.py --chunk <threshold>`, then
    `trace_mixes.py` (tier `full` only), then a restart (boot B) with the scenarios
    repeated under `-bootB`.
  - Then a stop and `analyze_trace4.py` for each node; the three nodes run concurrently, capped at 16 GB each.
  - Tier `quick`:
    - boot A runs the mixed, chunked, chunked_end, cache_long and distinct scenarios;
    - the restart runs cache_long and distinct only.
  - `analysis-summary.json` reports rows, differences, cross-restart pairs and a pass flag.
- **`kernel`:** runs kernel-lab bundles on the named nodes concurrently, with the cluster
  stopped. It writes `results/private/lab/<run>/verdicts.json` and checks that the
  nodes agree.
- **`measure` with `"suite": "<name>"`** (servebench's `screen`, `release`, `screen-tp3`
  or `release-tp3`, or a suite file) replaces `extras`, `bench` and the profile. Each arm
  boots and runs `bin/spark bench --suite <name> --compare none` into
  `results/private/suites/<run>/<label>`. That brings every bench's conditions: the nodes
  cooled, the fan performance curve running, the memory floor and temperature ceilings
  watched (a breach stops the run), and the arm's identity in the run's `suite.json`
  (`conditions.json` beside it records cooling, fan curve, thermal and memory). A failed
  step lets the remaining arms run; a refused bench (exit 2: a cluster that differs from
  its arm, no fan curve) ends the job. A suite with an engine grid step needs an arm with
  `VLLM_SERVER_DEV_MODE=1`, which the pre-boot check enforces. Arms after
  the first score the first arm's text in model steps. The table
  (`scripts/servebench_suite_table.py`) compares the second arm, the reference, with
  every other arm, and the first arm with the last as the run's own repeat. Order arms
  candidate, reference, candidate again.
- **Companion suites:** `"suite": ["release-tp3", "single-stream"]` runs the release suite,
  then each further suite on the same boot of each arm, into
  `results/private/suites/<run>-<suite>/<label>` with its own table
  (`results/private/lab/<run>-<suite>-table.txt`). A changed suite is a new suite (servebench
  compares only runs of the same definition), so measurements added to every release go in a
  companion suite and the release suite and its recorded references stay unchanged. Every
  release runs `single-stream` (one conversation at a time, concurrency 1 on distinct prompts):
  the speed one user sees, which draft acceptance on the generated text moves, beside the
  release suite's engine grid, the text-independent verdict on a single stream's step cost.
- **Telemetry:** every bench and script step of a `measure` job samples, on each node:
  - GPU SM and memory clocks, power, temperature and clock-event reasons, once a second
    (`nvidia-smi` over SSH);
  - the serving container's threads every 2 s: CPU use, and how much of it ran on slow
    cores (`cpu_capacity` below the midpoint, as sparknet's "big").
  The summary goes beside the step's output as `<output>.telemetry.json`: percentiles per
  field, event-reason counts, the busy threads with their core share and slow-core share,
  and the clock lock in force. Telemetry never fails a step.
- **`"telemetry": false`** on a `measure` job turns the per-step sampling off (A/B checks
  of its own effect).
- **Failed steps keep their evidence:** when a script step exits non-zero, every node's
  serving-container log tail (4,000 lines) is saved beside its output as
  `<output>.engine-<node>.log`, before a later boot replaces the container.
- **Arm options `memory_snapshot` and `compact_memory`:** before that arm's boot, serving
  stops and every node's free memory by block size (`/proc/buddyinfo`) is saved to
  `memory-<label>.json`. `compact_memory` then runs
  `sudo -n sh -c 'sync; echo 1 > /proc/sys/vm/compact_memory'` and records the result.
- **Placeholders:** `{label}` and `{run}` in extras arguments become each arm's label
  and the run's name, so arms that share extras write their own outputs.
- **Reading results:** `scripts/lab.py report RUN` prints the run's table and each
  measured step's node telemetry: SM clock p10/p50/p90 and minimum, power, peak
  temperature, clock-event reasons and the lock in force.
- **`gpu_clocks`:** `{"kind": "gpu_clocks", "lock_mhz": [min, max]}` locks every node's
  GPU SM clock (`sudo -n nvidia-smi -lgc`) for the jobs that follow, and
  `{"kind": "gpu_clocks", "reset": true}` resets it (`-rgc`).
  - The lock survives boots.
  - A lock that fails on any node is reset on all of them and ends the run.
  - While a lock is in place, `~/spark-lab/gpu-clocks-locked.json` records it.
    Restoring production (closing the window, including the watchdog's close) resets
    the clocks first, so production never serves locked.

Jobs that stop the cluster (`validate`, `profile`, `kernel`) run
`bin/spark --cluster-config <production profile> cluster stop --apply --parallel`. The
stopped containers are kept; the next boot's `cluster start --replace` replaces them.
Set `"remove_stopped": true` on a job to remove them as well (`cluster stop --remove`).
`run --dry-run` prints the production profile and each stop command.

Every arm must use the production profile's node map. `boot` refuses any other
node set before launch, because a window cannot recable the fabric.

Before an arm's first boot, every mounted B12X file must carry at least as many
`fence_proxy` calls as the image's own copy. Overlays built from an older B12X tree
fail this check. The overlay hashes of each run go to
`results/private/lab/overlays-<run>.txt`.

## Kernel-lab bundles

A bundle is a directory holding `candidate.json` and `overlay/{vllm,b12x}/<package path>`.
Overlay files are mounted over `/opt/spark3/candidate/<package>/<package>/...`.

`candidate.json` holds:
- `image`, `env`, `argv` (the harness command inside the container);
- `mounts`: `[source, destination]` pairs. A source is resolved from the bundle first, then
  from `~/spark-lab/inputs`, then from the node's checkout;
- `sync`: inputs that are not in Git, copied to `~/spark-lab/inputs` on the other nodes;
- `variant_configs`: configurations allowed more than one row group, such as production;
- `workdir`: the container working directory (default `/opt/spark3/candidate/b12x`);
- `verdict`: `"exit"` for a test-suite bundle, such as pytest over the image's own
  `/opt/spark3/candidate/vllm/tests`. It passes on exit status 0 with tests passed and none
  failed, and records the failed test ids, which the nodes must agree on.

Before replaying, a kernel job brings its nodes below the bench's cooling threshold
(55 °C hottest zone). If any is hotter, all of them cool together at the maximum fan
floor until the last is below it; then their usual fan control returns. This is the
same thermal baseline every bench starts from ([methodology](methodology.md)).

Bundles, synced inputs and outputs stay under `~/spark-lab/` on each node, outside
the deployment checkout.

```sh
scripts/lab.py kernel-local experiments/2026-09-29-determinism/bundles/mhc-smoke --dry-run
```

`kernel-local` runs on the node itself. It:
- refuses while the serving container runs there;
- takes `/tmp/spark-lab-gpu.lock`;
- writes `output.txt` and `verdict.json`: row groups, bit-equality sets, timings, errors,
  input hashes and pass/fail.

A verdict passes when:
- every configuration not listed in `variant_configs` has one row group;
- every compared pair is bit-equal;
- nothing crashed.

A test-suite verdict passes when the suite exits 0 and reports passed tests and no
failures or errors.

Compile caches persist per node in `~/.cache/spark-lab/compile`.

## Release comparison

Measure a candidate against the previous release the same way every time, so a
change in the numbers means a change in the release, not in the method:

```json
{"kind": "measure", "profile": "release", "bracket": false,
 "arms": [{"config": "<candidate>.json", "label": "<candidate>"},
          {"config": "<previous release>.json", "label": "<previous>"},
          {"config": "<candidate>.json", "label": "<candidate>-again"}]}
```

- **One session, candidate first.** The arms run back to back on the same boot
  conditions. The candidate goes first, so heat soak never counts against it, and runs
  again last to bound the session's drift. Read the verdicts against the previous
  release: `scripts/release_table.py RUN <previous> <candidate> <candidate>-again`
  (the table compares each listed arm with the first listed, in argv order).
- **The `release` profile (`scripts/lab.py`):**
  - quality;
  - decode on prose and code at 1, 2, 4, 8 and 16 streams, 5-10 samples until each
    interval is within 2%;
  - prefill on source text at 1K, 32K, 128K, 256K, 512K and 1,000,000 tokens, three
    repeats;
  - the sixteen-prompt acceptance sweep;
  - twelve prompts in turn at one stream, and different prompts at once at 4, 8 and 16 streams;
  - short-prompt TTFT, mixed traffic, and `token_determinism.py --wide`.

  About 35-40 minutes an arm on TP4.
- **The report (`scripts/release_table.py`, the profile's table step)** gives one
  verdict per metric. faster or slower means beyond the two arms' combined 95%
  intervals; same means within them. Sections run from the most trustworthy down.

What the rules guard against (each found the hard way, 2026-10-11):
- **The bench's decode points follow the text.** Each is one prompt repeated on N
  streams. A release that changes the output text (any precision change) changes the
  workload at every point: acceptance, verification length, step time. Code c16 read
  -8% on r6f, and sixteen distinct prompts read level. Judge throughput on distinct
  prompts and on the paired sweep; read the bench's points as direction only.
- **The size limit.** The bench runs only prefill sizes below max-model-len - 64, so
  1048576 on a 1M profile is skipped silently. The profile uses 1,000,000.
- **Repeats.** With two repeats a t interval spans tens of percent. The report uses
  1.96 sd / sqrt(n) for prefill, with three repeats.
- **Cost curves.** Arms run on pinned DSpark cost curves (`SPARK3_DSPARK_COST_DIR`).
  To compare step costs, boot each arm with an empty cost directory and
  `SPARK3_DSPARK_PROFILE_TOKENS=random`. The boot pins its real-row curves there
  (experiments/2026-10-11-c16-cost).
