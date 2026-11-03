# Candidate-38 consolidation checks

Base deployment commit: `e9e0f09b15891ba4ab77c74e7e749e9f9be898b5`.
The consolidation carries the fault-investigation records and corrects candidate-only
profile metadata and documentation. It changes no vLLM patch, image, serving argument
or rendered rank environment. No GPU job or serving operation was run for these checks.

## Source reproduction

The lock, series and all 572 patch files were read with `git show <base>:<path>`;
their exact bytes were extracted into `.work/copy-drafter-shared-budget-integration/`.
A fresh shared clone of the local pinned vLLM object store was checked out on
`feat/copy-drafter-shared-budget-integration` at upstream base
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354`. Its `origin` is the contribution fork
`https://github.com/christopherowen/vllm.git`; `upstream` is
`https://github.com/local-inference-lab/vllm.git`. Each patch was applied in series order:

```sh
git -C .work/copy-drafter-shared-budget-integration/vllm \
  -c user.name='Christopher Owen' \
  -c user.email=3221756+christopherowen@users.noreply.github.com \
  am --quiet --committer-date-is-author-date <extracted-patch>
```

The patch-set digest uses the repository's `patchset_sha256` definition (series
bytes, then each listed patch name, NUL and patch bytes). Patch 561 reproduces
candidate-37's head `5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575`; patch 572 reproduces
the final candidate's locked head and tree. [Receipt](source-reproduction.json).

Verification from the fresh vLLM directory:

```sh
git diff --check 19f2c20ed4d64ae22bd009ceacadd85e2a81e354..HEAD
rg -n 'slot_draft_limits|_limits_staging|_limits_event|_pending_limit_resets|slot_limits' vllm tests
git status --porcelain
```

The diff check passes, the search returns no matches (exit 1), and the tree is clean.
All 23 Python files changed since candidate-37 pass Python's `compile(..., 'exec')`
without imports or bytecode writes. These are source checks, not GPU test execution.

## Profile, repository and build checks

```sh
bin/spark doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-38/tp4.json doctor
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-38/image-dgx4.json doctor
python3 -m unittest discover -s tests -t .
python3 scripts/render_docs.py --check
git diff --check
```

All pass. The repository suite runs 626 tests with 5 skipped in 38.740 s
([receipt](repository-tests.json)). The [profile comparison](profile-check.json)
records both inherited doctor errors before correction and verifies identical
rendered environments for all four ranks afterward. The baseline reference now
agrees with the lock, and the redundant HCA input is removed; the node map still
renders the same HCA environment variable.

The existing final image's complete build inputs on dgx4 also pass a fresh
read-only `build check` from its clean detached build checkout `df50aa94`:

```sh
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-38/image-dgx4.json build check
```

The [build receipt](build-check.json) records the exact checkout, input directory
and hashes of the private raw logs. The candidate-only baseline/HCA metadata
correction does not alter these build inputs.

## Qualification boundary

These checks retain the final image's already-recorded release gate (8,069 passed,
360 skipped), targeted tests (395 passed, 1 skipped), and PDL suite (234 passed,
zero violations). The gate's `tblib_stub` and `offline_skip` support and the PDL
suite's `--noconftest` scope remain explicit in their original bundles and receipts.
Earlier failed image results remain recorded too. No new speed or memory
measurement was made during consolidation. The combined candidate still requires
distributed boot/repeat, full-model correctness and performance qualification;
these unit and source checks do not promote it or establish no performance regression.
