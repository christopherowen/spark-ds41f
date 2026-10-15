# TileLang main rebase

Candidate compiler source for the r6h TileLang backend. The only changed
build input is TileLang: move from `b95ee4ff` to main at `fd2ddda1` and
apply the remaining MXFP8/MXFP4 changes from
[tile-ai/tilelang#3495](https://github.com/tile-ai/tilelang/pull/3495).
TileLang's version is still 0.1.15; the new pin includes 38 upstream commits.

The SM121 enablement in
[#3430](https://github.com/tile-ai/tilelang/pull/3430) is now upstream as
`5f803828`, so its local patch is omitted. The candidate series retains the
MXFP8 and MXFP4 changes and the CUDA source-test portability update. It
replays to exactly the tree of #3495 at `d833b788`:
`b079bea60e9b17d656d20ed709c03fd2237934ad`.

## Inputs

- [base.json](base.json) records the r6h deployment and source pins.
- [upstreams.lock.json](upstreams.lock.json) changes only the TileLang source
  entry and selects [source.json](source.json).
- [tilelang/series](tilelang/series) is applied after the new upstream pin.
  Each patch records its base, rationale, quality implications, validation,
  and submitted upstream PR.
- [candidate.json](candidate.json) describes the change.
- [tp3.json](tp3.json) and [tp4.json](tp4.json) preserve the corresponding r6h
  model/kernel runtime settings and select the candidate source lock, distinct
  image tag, and TileLang tree label. A separate TileLang JIT cache namespace
  ensures the qualification compiles the rebased compiler outputs, even though
  the package version is still 0.1.15. Launch is disabled until qualification.

The promoted pins, profiles, and immutable baseline records are unchanged.

## Validation

The native CUDA compiler was rebuilt from the same tree during the upstream
PR submission. The TVM source revision and recursively checked dependencies
are unchanged; the prebuilt TVM libraries and unchanged Python bindings were
reused. Validation used an isolated container on dgx4 with the GPU hidden,
CPU and memory limits, and a private temporary compiler cache.

- New MXFP8/MXFP4 suites: 20 passed, 7 runtime tests skipped.
- MXFP8, FP8 × FP4, and MXFP4 device compilation passed for both `sm_120a`
  and `sm_121a`.
- Lowered scale selectors were `ki` for MXFP8 and `ki * 2` for MXFP4 in the
  block_K=128 kernels.
- Repository formatter/pre-commit and whitespace validation passed on #3495.
- Candidate TP3/TP4 configuration checks and fresh `build prepare --only tilelang`
  passed. Prepared source, submodule commits, profile controls, and the
  exported TileLang context fingerprint were verified.
- The first combined validation harness exited after the passing source
  suite because its standalone compile calls omitted a `Target` context.
  Adding that context fixed the harness; all six device compilations passed.
  Both receipts are retained under `runs/`.

A fresh deployment-style replay uses the fixed human committer and author
commit dates required by the build tooling. Its commit ID differs from the
upstream PR because the patch messages include deployment provenance; the
resulting source tree is identical. The candidate manifest records this
reproducible patch head, tree, patch fingerprint, and submodule commits.

```sh
bin/spark --cluster-config experiments/2026-10-12-tilelang-main/tp4.json doctor
bin/spark --cluster-config experiments/2026-10-12-tilelang-main/tp4.json build prepare --only tilelang
```

## Qualification before promotion

Retain as a candidate. In a coordinated GPU window, build the distinct image,
run the block-scaled runtime suites and r6h kernel tests, then compare the
current TP4 quality, batch invariance, decode, and prefill workloads against
r6h. Record the image digest, every run, memory observations, and the decision.

The register-A optimization
[#3286](https://github.com/tile-ai/tilelang/pull/3286) is still unmerged and is
not applied here. If it enters the selected upstream base, include the scale
selector guard from
[Horizon-Tech-doo/tilelang#1](https://github.com/Horizon-Tech-doo/tilelang/pull/1)
and repeat the varying-scale runtime coverage.
