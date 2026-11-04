# TileLang consumer-fences

## Change and source

Generic-proxy stage reads must finish before a consumer releases the stage for the next async-proxy TMA write. Emit the required proxy fence before consumer release.

This is an independent follow-up to [#875](https://github.com/christopherowen/spark-ds41f/pull/875)
at deployment `e33d1e0223b7346b821a9201d5c67c3745c0151a`. The six-patch series references its five
unchanged patches, then adds only [upstream #3537](https://github.com/tile-ai/tilelang/pull/3537) at
`d044264967190554349ad5ddebf478f4432478f6`. It retains MXFP8/MXFP4, prepare/freeze, IKET extensions and
the MX shared-scale selector repair. Other sources and serving arguments are
unchanged from the parent candidate.

The reproducible patch head is `a500cfb06c620516243f9d71a1a814aa115f6d69` and tree is `93d54ec9d049abe77cb7d0de3b9ebc1f98ca9951`.
The tree exactly matches the prior standalone compiler snapshot. Different
commit metadata does not re-identify or replace the native run receipts.

## Recorded standalone evidence

The parent fails both generated-code fence assertions in each process. The port passes both assertions and the fast-math numerical reproducer twice. The numerical reproducer also passes on the parent. Portable generic-read and mma.sync cases ran on SM121; Hopper-only WGMMA cases were excluded.

`results.json` preserves every selected run, including failed controls and
invalid setup attempts. `evidence.tar.gz` contains their original command
arguments, output, JUnit/CUDA files, temperatures, memory samples and library
identities, plus the native build receipt. The full audit and harness are pinned
at [ad40ec23](https://github.com/christopherowen/spark-ds41f/tree/ad40ec2379053fe132c3167ce19ade22877224c9/experiments/2026-11-04-tilelang-kernel-review).
The original compiler was built in toolchain image
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
No released serving build was remeasured. No new GPU run is claimed here.

## Reproduction

The disabled TP3/TP4 profiles extend the parent profiles and select this
candidate's lock, manifest, six-patch series and proposed immutable image tag.
Prepare and validate locally:

```sh
scripts/example_site.py
bin/spark doctor
bin/spark --cluster-config experiments/2026-11-04-tilelang-consumer-fences/candidate-tp4.json doctor
bin/spark --cluster-config experiments/2026-11-04-tilelang-consumer-fences/candidate-tp3.json doctor
bin/spark --cluster-config experiments/2026-11-04-tilelang-consumer-fences/candidate-tp4.json build prepare --only tilelang
python3 experiments/2026-11-04-tilelang-consumer-fences/check_sources.py
```

The dedicated CI workflow reproduces the same source and profiles, following
the repository policy of skipping same-repository PR runs in favor of the
local unit gate. The exact local validation commands and output are under
`runs/local/`; this PR does not qualify a full runtime image.

The native regression selection in the recorded toolchain is:

Run `python3 experiments/2026-11-04-tilelang-consumer-fences/check_gpu.py` with `PYTHONPATH` selecting the built compiler source tree. The three portable functions are called directly because the original file is marked SM90-only; no WGMMA-only function is called.

The pytest selections are relative to the built compiler source tree; import
that selected compiler before pytest to prevent a control test directory from
overriding it. Use a fresh process and cache for each repeat, as the preserved
command receipts do. Any future GPU run needs the applicable reservation and
idle/memory checks; none of these steps starts vLLM.

## Quality and acceptance

Changes synchronization only. Restrict the fence to generic-proxy reads; preserve the upstream treatment of async-only consumers. No numerical failure or serving speedup is claimed from the sampled workload.

The recorded tests validate this isolated change. Before promotion, run the
complete retained-feature suite on the resulting candidate, build and identify
one runtime image, perform the release quality/performance gates and two
candidate serving boots against recorded release runs, and obtain owner
acceptance. Do not infer combined-stack qualification from separate PRs.
Launch remains disabled, and production configuration and immutable baseline
manifests are unchanged.

## Local validation for this PR

Promoted and TP3/TP4 candidate doctors pass. Standard fresh preparation applies
all six patches and verifies the deterministic head/tree and all recursive
submodule pins. The tree equals the recorded sp8 variant, and 19 changed
Python files parse for Python 3.12. The complete local repository suite passes
**626 tests**; generated documentation and workflow YAML syntax checks pass.
`build check` returns 1 with its explicit incomplete-input warning because only
the TileLang source was prepared. No complete runtime image check is claimed.
Exact commands and outputs remain under `runs/local/`; `validation.json` records
the check boundaries. No GPU or serving operation was performed for PR creation.

Local stdout/stderr files are bundled byte-for-byte in
`runs/local/validation-output.tar.gz`; the JSON command receipts identify each
archive member. Original `.log` files remain ignored and local.
