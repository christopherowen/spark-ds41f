# Commands, attempts, and resource record

The deployment checkout is `/home/chris/spark-lab/checkouts/expert-route-reuse`
on `sp8`. Commands below ran there unless marked local. Source checkout:
`.work/vllm-source`. No GPU or serving action was run.

## Preparation

The isolated deployment checkout was created from the existing remote repository
and checked out at `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74` on
`feat/expert-route-reuse`. The source checkout was cloned with `--shared` from
`/home/chris/projects/spark-ds41f/.work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-a039bacb/src/vllm`.
It has `upstream=https://github.com/local-inference-lab/vllm.git`,
`origin=https://github.com/christopherowen/vllm.git`, and `local-cache` for that
remote-local prepared tree. It started at the candidate-38 lock's `revision`, on
`feat/expert-prefill-swizzle`. The full ordered series named by that lock was
applied with:

```sh
git -c user.name='Christopher Owen' \
  -c user.email=3221756+christopherowen@users.noreply.github.com \
  am --quiet --committer-date-is-author-date <every patch in series order>
git diff --check 19f2c20ed4d6 HEAD
git rev-parse HEAD 'HEAD^{tree}'
```

`runs/remote-prepare.txt` records all 572 patches, the exact reproduced head/tree,
and a historical patch whitespace warning. Preparation ran in a user scope with
`MemoryMax=8G`, `CPUQuota=200%`, `TasksMax=64`, `nice -n 10`, and
`flock -n /tmp/spark-source-prepare.lock`.

The initial local attempt (before the recovery instruction) was:

```sh
bin/spark --cluster-config \
  experiments/2026-10-26-spark-ds41f-candidate/candidate-38/tp3.json \
  build prepare --only vllm
```

It failed with a full disk at patch 0162. The failed log is preserved as
`runs/local-preparation-failed.txt`. No local preparation resumed after recovery.
An initial remote cache path ending `9b160482` did not contain the source commit;
inspection selected the existing `a039bacb` tree instead. No source was changed
in either cache.

## CPU code generation

Each job acquired `/home/chris/spark-lab/offline-cpu.lock` followed by
`/home/chris/spark-cpu-work.lock`, in a user unit with `MemoryMax=8G`,
`CPUQuota=200%`, and `TasksMax=128`. Docker itself was additionally limited:

```sh
docker run --rm --name expert-route-codegen --network none \
  --cpus 2 --memory 8g --memory-swap 8g \
  -e CUDA_VISIBLE_DEVICES= -e MAX_JOBS=2 -e OMP_NUM_THREADS=2 \
  -v "$PWD/experiments/2026-11-05-expert-route-reuse:/experiment:ro" \
  -v "$SOURCE/vllm/models/deepseek_v4_1/tilelang/moe.py:/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/tilelang/moe.py:ro" \
  --entrypoint /bin/sh \
  sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b \
  -c 'uv venv --system-site-packages --python /usr/bin/python3 /tmp/route-venv >/dev/null && /tmp/route-venv/bin/python /experiment/check_codegen.py --expect STATE'
```

For `STATE=before`, SOURCE is `.work/control`, containing the original file from
`git show 1f331025:vllm/models/deepseek_v4_1/tilelang/moe.py`. For `STATE=after`,
SOURCE is `.work/vllm-source`. Both six-case runs passed, with source hashes in
`runs/codegen-before.txt` and `runs/codegen-after.txt`. All panel-zero hashes
match across runs. Panel 3 equals panel 0 before the fix and emits rasterization
afterward. The image logs missing native `_C`/CUDA runtime warnings because this
is CPU lowering; no device compilation or GPU execution occurs.

Exploratory compiler-API failures are retained: `cpu-lower-probe.txt` rejected an
unsupported `pass_configs` argument; `-v2` rejected the obsolete target-string
syntax; `-v3` lacked an active target context. `-v4` succeeded with a dictionary
CUDA target and nested target/PassContext. These are setup failures, not kernel
correctness or performance failures. `compiler-interface.txt` records the API.

## Accounting and static validation

```sh
python3 -m unittest discover \
  -s experiments/2026-11-05-expert-route-reuse -p 'test_*.py' -v
bin/spark doctor
ruff check vllm/models/deepseek_v4_1/tilelang/moe.py \
  tests/kernels/moe/test_deepseek_v41_tilelang_moe.py
ruff format --check vllm/models/deepseek_v4_1/tilelang/moe.py \
  tests/kernels/moe/test_deepseek_v41_tilelang_moe.py
pre-commit run --files vllm/models/deepseek_v4_1/tilelang/moe.py \
  tests/kernels/moe/test_deepseek_v41_tilelang_moe.py
git diff --check
```

The first local unit run failed because the all-dead case had no zero-valued
payload keys (8/9 passed). The Counter was initialized with those keys, then 9/9
passed locally and remotely. This failure description is reconstructed from the
console; its raw first log was not saved. The final suite adds unsupported
geometry and empty-capture checks. No synthetic result is labeled real traffic.

The first remote doctor run used Spaceport's ignored switched node maps against
the checkout's direct-cable recipe and failed. `runs/doctor-site-map-failed.txt`
is retained. Only this task's ignored maps were replaced with the already-valid
DGX maps from its isolated local checkout, for static validation of that recipe;
no site or production configuration was edited. `runs/doctor.txt` is the retry.

Ruff reuses the installed hook executable at
`/home/chris/.cache/pre-commit/repoqyjfirf7/py_env-python3.12/bin/ruff`.
Pre-commit reuses the existing environment under
`/home/chris/spark-lab/checkouts/vllm-drafter-orchestration-20261105/.venv`;
it does not install into or change that environment. Logs retain hook results.
The task checkout lacks complete build inputs, so a full `bin/spark build check`
is not represented as passed. CPU/memory limits are ceilings; no GPU-memory
observations, boot repetitions, confidence intervals, or timings exist yet.

## Publication access

A remote HTTPS `git push --dry-run origin HEAD:refs/heads/feat/expert-prefill-swizzle`
failed because sp8 had no GitHub credentials. An SSH test with GitHub's published
host key also failed public-key authentication. No credentials were copied.

Final accounting: 10/10 tests passed. Static doctor: configuration OK. Both
candidate and untouched-base pre-commit runs failed on the same three E501 lines,
existing formatting, 122 typing errors, and three pre-existing TileLang import
policy violations. The hook's unrelated automatic formatting was removed;
the final source diff contains only two kernel lines and 21 regression-test lines.
`runs/source-precommit-baseline.txt` records the control. The source commit is
`5cc36f7662f77330efe7d40c141e1d79d38e2c37`, tree
`7acbc93b4e42ecb74d9a2f621ed13b4985708493`. Final `git diff --check` passed.

The remote host prepared a 2,418,257-byte Git bundle with two packing threads
and 32 MiB packing-window memory. This small bundle was imported into the existing
local inspection checkout for publication only; no local clone, source
preparation, tests, or build ran after recovery. The local import took 0.25 s.
Publication uses the existing Mac credentials, without copying them to a node.
The push caps packing at one thread with a zero search window, 16 MiB window
memory and 32 MiB packed-object mapping limit. The review base branch is the
exact locked candidate-38 head; neither main nor a deployment profile is changed.

Source publication succeeded as draft
<https://github.com/christopherowen/vllm/pull/5>. The review base branch
`feat/expert-reuse-candidate-38` points to the exact locked head; the source branch
`feat/expert-prefill-swizzle` points to the candidate commit above. The exported
patch was applied to a clean temporary index at the base and reproduced the
candidate tree exactly (`runs/exported-patch-check.txt`). No ignored files or
credentials are deployment inputs.


## Stronger CPU audit and native-screen preparation

The same image, two-CPU/eight-GiB limits and two shared CPU locks were used for:

```sh
python3 -m unittest discover -s experiments/2026-11-05-expert-route-reuse -p 'test_*.py' -v
python3 experiments/2026-11-05-expert-route-reuse/analyze_ordering.py --output experiments/2026-11-05-expert-route-reuse/runs/ordering-bounds.json
# Inside the existing CPU-only image/venv, with the candidate file mounted:
/tmp/route-venv/bin/python /experiment/check_audit.py
# With the untouched candidate-38 file mounted instead:
/tmp/route-venv/bin/python /experiment/check_default_baseline.py
```

`audit-cpu-matrix-setup-failed.txt` records the failed `inspect.getfile(JITFunc)`
probe. Using the module's `__file__` fixed it. The next complete matrix used an
accidental topk=7; its JSON/log are retained with `-topk7` names. The final native
matrix at topk=6 passes all 72 compiler cases, 48 aligned workspace cases and
four plan-default cases. `audit-default-comparison.txt` confirms four old/final
default source hashes match. These checks allocate no GPU memory.

`audit-source-precommit.txt` retains the existing lint/type/import failures after
the expanded source tests. Unrelated autoformat output was removed. Source
commit `f3bfdf8b8c08870754cc3c756b75faa9156595e0` was exported as patch 0002 and
published to the existing PR; its tree is in `base.json`.

Hardware bundle `bundles/swizzle-screen/` binds source, test and benchmark hashes.
The wrapper retains `kernel-local`'s serving refusal and shared GPU lock, makes
lock acquisition nonblocking, caps Docker at two CPUs/16 GiB/no swap, and records
five-second host thermal/memory readings. It aborts its own container at 85 C,
below 12 GiB available memory, lost ownership, or 45 minutes. It starts below
55 C with at least 24 GiB available. Performance modes invoke the repository cooling and performance fan helpers
and require their success, restoring the prior fan-service state afterward. No fan software is installed by this experiment.
The first host inspection found no fan service on sp8; performance there remains
blocked unless the established fan policy is clarified. Correctness alone does
not establish a speedup. The controller reads the dgx1 hold immediately before
each invocation; the wrapper independently requires the exact node hold/window.

```sh
python3 experiments/2026-11-05-expert-route-reuse/run_screen.py \
  --mode gate --process-id gate-a --holder expert-route-reuse \
  --window WINDOW --out /node-local/new-receipt-directory --dry-run
```

Remove `--dry-run` only after publication, exclusive handoff and the documented
hold/resource checks. Run gate, smoke, full A and full B as separate processes;
these are process repeats, not serving boots. Full serving qualification remains
separate. The paired screen uses eight input blocks, six timing replays per arm
and three whole-plan calls per CUDA graph. Arm order rotates/reverses; all arms
share weights, activations, routes, workspace and output address. It compares
panels 0/1/2/3/4/8, TP3/TP4 shard geometry on one GPU, low-overlap synthetic
routes, skew and M-tier boundaries. The 95% intervals use paired input-block
means with t(7); raw values remain in every receipt. It checks output bits,
unchanged IDs/weights and the same first 17 rows in a separate invocation.


Screen preparation validation completed remotely under the same two CPU locks
and 2-CPU/8-GiB user-unit caps: Python syntax compilation, 14 tests, static doctor,
and the wrapper dry-run pass. `audit-exported-series.txt` reproduces source tree
`8d20314f29ed7cc26b8c4e15f85046b36faaa6df` by applying both exported patches to a
fresh index at the candidate-38 base. The earlier full 572-patch preparation is
retained rather than repeated. `screen-dry-run.txt` records the exact bounded
Docker plan. Complete task build inputs are still absent, so a full build check
is deferred until preparation of a serving image.


## Correctness preservation comparison, before hardware execution

The acceptance criterion is now a demonstrated correctness repair with no
material performance loss; a speedup is optional. `default` loads the untouched
candidate-38 module and the fixed module in one process, with one shared weight
copy. This avoids the previously recorded memory-placement confound. All 16
CPU grouped-GEMM default sources match (`default-pair-cpu.txt`). The first probe
also attempted the unchanged SwiGLU helper, which queries the real GPU's vector
width; `default-pair-cpu-no-driver-failed.txt` retains that setup failure. No GPU
access was enabled to evade the shared hold. Native mode checks every plan
kernel, including that helper, on the actual hardware.

The wrapper uses the existing bench fan/cooling functions through a local
transport for the selected node. DGX4 has the required fan service installed
(read-only inspection); sp8 does not. An independently launched `watch_screen.py`
user unit stops only the exact named/container-window-labelled job if the main
unit ends or its hold heartbeat ages beyond 120 seconds. It is separate from
the global lab watchdog, which still owns production restoration. Watchdog
output and resource receipts remain beside each job. Start the watchdog as a
separate user unit immediately after the bounded main unit, before leaving the
job unattended. Both need the same holder/window/process identity.

```sh
python3 experiments/2026-11-05-expert-route-reuse/run_screen.py \
  --mode contract-before --process-id contract-before --holder HOLDER \
  --window WINDOW --out /node-local/contract-before --dry-run
# After exclusive handoff: contract-before, gate-a, default-a, gate-b, default-b.
# Run each under a remote user unit with CPUQuota=200%, MemoryMax=8G,
# TasksMax=128; Docker additionally has 2 CPUs/16 GiB/no swap.
python3 experiments/2026-11-05-expert-route-reuse/watch_screen.py \
  --unit MAIN_UNIT.service --process-id PROCESS --holder HOLDER \
  --window WINDOW --out /node-local/PROCESS
```

The mode/order specification above is a plan, not a receipt of GPU execution.
All `--apply` serving operations remain separate coordinated commands with the
named committed profile; this isolated kernel phase requires none.


## Test-report interface audit (CPU only)

Before any native execution, review identified that pytest's JUnit report does
not guarantee a `failure.type` attribute. The expected-old-failure checker now
requires the exact test identities, exit code 1, four failures, the swizzle
marker in each failure message and AssertionError in its text. It rejects
compiler/setup errors and skipped tests. A real CPU pytest fixture intentionally
emits four matching failures solely to verify the report format; this is not
an execution of the native source regression. Its XML and output are retained
as `pytest-format-probe.*`. Four parser tests include corrupted compiler/error,
skip, count and identity cases. The full preparation suite passes 31 tests.

The report probe ran without GPU/network access, with one CPU/2 GiB inside the
existing image and a system-site-packages uv environment. The first attempt
could not use the image's root-owned uv cache; the second could not write the
report because a guessed container UID differed from the host UID. Both failed
logs, and the resulting four missing-report unit errors, are retained. The
successful attempt uses a container-local `UV_CACHE_DIR` and the actual remote
UID/GID (1002/1002). This changes no serving environment or credentials.

Native runs now retain their JUnit files under each receipt directory. The
fixed native gate requires all 24 selected tests to pass without skips. The
kernel source remains f3bfdf8; only the experiment's report handling changed.
Pinned checkpoint metadata was also verified remotely: revision 1bae0e10,
config SHA-256 8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879;
text geometry and SwiGLU limit match the screen. `checkpoint-geometry.json`
retains the observed values without weights or prompt data.

Hardware coordination now preserves the explicit Engram, collective-pdl and
verification-budget reservations in that order. This task has no acquisition
waiter and no GPU hold/request file. After those reservations, its ready native
sequence is contract-before, gate-a, default-a, gate-b, default-b, estimated
30–45 minutes on idle DGX4 with standard cooling/fan control. Each individual
process has a 45-minute safety cap. An optional non-timing SP8 correctness subjob
is permitted only when an actual holder owns and executes it. The coordination
record never overrides a live hold. No GPU results exist at this checkpoint.


## DGX4 execution receipts (2026-11-06)

Executed on `dgx4` under scoped hold `dgx4-expert-route-reuse-1793950700` (`nodes: ["dgx4"]`):
- `runs/contract-before-attempt1-conftest-tblib/` (exit 1: `tests/conftest.py` missing `tblib`; fixed with `--noconftest`)
- `runs/contract-before/` (exit 0: 4 expected `AssertionError`s reproduced in `14.3 s`)
- `runs/gate-a/` (`82.7 s`, `24 passed`) and `runs/gate-b/` (`18.5 s`, `24 passed`)
- `runs/default-a-attempt1-run-ssh-binding/` (exit 1: package-global `run_ssh` binding; fixed in `run_screen.py`)
- `runs/default-a/` (`170.2 s`, 30 cases) and `runs/default-b/` (`150.9 s`, 30 cases)
- `runs/full-a/` (`521.6 s`, 38 cases across panels `0, 2, 3, 4, 8, 16`) and `runs/full-b/` (`508.4 s`, 38 cases)
- `runs/gate-panel2-a/` (`28.0 s`, `24 passed`) and `runs/gate-panel2-b/` (`19.5 s`, `24 passed`)
Fan state restored to `automatic` (`dgx-fan-control.service` `inactive`) and `dgx4:~/spark-hold.json` released.
