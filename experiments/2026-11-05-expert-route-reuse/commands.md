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
