# TileLang main refresh

## Candidate and scope

Evaluate TileLang main `2123e412ce2217b324c50c4ec967213906456ac5`
(2026-11-03, “Reuse partial barriers across storage scopes”, #3761), the
latest main checked on 2026-11-04. It still reports version 0.1.15; the full
commit, patch fingerprint, tree and recursive submodule commits identify it.
`base.json` records the deployment repository commit and the immutable 1030
reference. The only dependency advanced is TileLang.

The candidate retains the production instruction and JIT features and includes
the staged IKET extensions, as requested. The original patches, promoted lock,
release recipe, source manifest and baseline remain reproducible.

| Previous patch | Candidate entry | Reason and upstream status |
| --- | --- | --- |
| `patches/tilelang/0001-sm121-block-scaled-mma.patch` | omitted | Already upstream through [#3430](https://github.com/tile-ai/tilelang/pull/3430), merged 2026-10-10. |
| `patches/tilelang/0002-sm120-mxf8f6f4-block-scaled-mma.patch` | `tilelang/0001` | MXFP8 and FP8 × E2M1, expanding FP4 loads and per-K32 UE8M0 scales; [#3495](https://github.com/tile-ai/tilelang/pull/3495) remains open. |
| `patches/tilelang/0003-sm120-mxfp4-block-scaled-mma.patch` | `tilelang/0002` | MXFP4 UE8M0 `scale_vec::2X`; also covered by #3495. |
| `experiments/2026-10-17-r6u/tilelang/0004-jit-prepare-and-frozen-kernel-cache.patch` | `tilelang/0003` | `tilelang.prepare`, `JITImpl.prepare`, cache freeze/thaw, and a named error for an unprepared call; still local. |
| `patches/tilelang/0004-iket-extended-ranges-and-payloads.patch` | `tilelang/0004` | Staged START_END/MULTI_EVENT ranges, cross-warp scopes, all 15 payload dtypes, ExtendedNativeDump and profiling CLI options; no upstream PR. |

The instruction and JIT patches needed three-way context updates; none had a
merge conflict. The IKET patch applied directly. Each exported patch records
its upstream base, rationale, quality implications, tests and upstream status.
Historical GPU results in their messages describe the old build only.

`tilelang/0005` handles an interaction with upstream #3286: its newer compact
shared-scale path hardcodes both byte selectors to zero. Keep that optimization
for NVFP4 (`scale_vec::4X`), and retain the existing per-MMA selectors for MXFP8
(`1X`) and MXFP4 (`2X`). Four new GPU regression cases use unit operands and
scales 1, 2, 4 and 8 on either A or B; every result must be exactly 480. These
cases are retained for target-hardware execution, not counted as local passes.

## Profiles and build identity

`candidate-tp4.json` and `candidate-tp3.json` overlay the frozen 1030 recipe at
its ring and triangle shapes. Both select one candidate lock, source manifest,
TileLang series and immutable proposed image tag. Launching is disabled while
qualification is pending. All other upstream sources and serving arguments
come from 1030. The profiles carry no inherited benchmark qualification.

Fresh preparation records and verifies the complete TileLang source identity:

```sh
scripts/example_site.py
bin/spark --cluster-config experiments/2026-11-04-tilelang-refresh/candidate-tp4.json build prepare --only tilelang --lock
bin/spark doctor
bin/spark --cluster-config experiments/2026-11-04-tilelang-refresh/candidate-tp4.json doctor
bin/spark --cluster-config experiments/2026-11-04-tilelang-refresh/candidate-tp3.json doctor
python3 experiments/2026-11-04-tilelang-refresh/check_sources.py
python3 -m unittest discover -s tests -p test_source_locks.py
python3 -m unittest discover -s tests -p test_build.py
python3 -m unittest discover -s tests -p test_kernel_backend.py
python3 -m unittest discover -s tests -p test_releases.py
```

The `tilelang-refresh` CI job also checks both candidate profiles, freshly
applies the series without `--lock`, and runs the source identity/syntax check.
It follows the repository's existing trigger policy: same-repository PRs
use the local test gate; external PRs, scheduled and manual runs execute CI.

The full local gate ran with Python 3.12.13 and the hashed bench-client
dependencies: **626 tests passed** in 40.213 s. Both candidate doctors and the
promoted doctor pass. Fresh preparation verifies all five patches and the
recursive submodule pins; all 18 changed Python files parse as Python 3.12.
The full suite's servebench output comes from its test fixtures, not the cluster.

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --require-hashes -r requirements/bench-client.txt
.venv/bin/python -m unittest discover -s tests
git diff --cached --check
```

The example node maps above are ignored, local-only inputs for offline checks;
their head address is aligned with the frozen recipe because overlay profiles
do not perform the recipe's site-address substitution. Use the site's actual
maps for a hardware build. Native command receipts,
including errors, are under `runs/local/`. `results.json` lists the completed
checks and their limits. Source syntax checks do not execute the compiler.

## Quality, workload and acceptance gates

No feature loss is the acceptance condition: retain SM121 enablement, the MX
instruction formats and exact varying-scale products, prepare/cache-key reuse,
freeze rejection and thaw, and every staged IKET range, payload and dump mode.
A clean patch application establishes source reproducibility, not numerical or
performance equivalence after a compiler update.

On an idle authorized build host, prepare the complete candidate inputs, run
`build check`, then build once with the candidate profile and distribute that
same OCI digest. The target-hardware tests must include:

- TileLang's NVFP4, MXFP8/mixed FP8 × FP4, MXFP4 and NVFP4 scale-layout suites
  on SM121, including their varying-scale exact-product tests;
- prepare/freeze/thaw behavior and cache reuse for the serving call forms,
  including a cold cache and a single cold rank;
- `testing/python/tools/test_tilelang_tools_cuda_iket.py`, including the GPU
  range/payload cases, plus the serving IKET debug mode;
- the deployment's release tests, output digests, sampled controls, mixed
  traffic and memory guards at the frozen recipe's workload limits;
- two candidate boots (a boot and a repeat boot), comparing each with the
  recorded 1030 suite for its shape. Do not remeasure the released build.

Keep every run, error, startup memory minimum and GPU memory observation.
Stop qualification on a numerical regression, lost API, failed request,
unprepared kernel or memory-guard stop. Compiler changes can alter codegen,
synchronization and rounding even when each carried patch is unchanged.
No runtime performance claim or promotion follows from the local checks.

## Decision

See `decision.md`. This change prepares a reviewable candidate. A native CUDA
image build, GPU suites, two serving boots and owner acceptance are still
required before promotion. No cluster operation is part of this preparation.
