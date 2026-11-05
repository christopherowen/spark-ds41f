# Candidate 36: direct switched exchange

Base and original source/image identities: `base.json`. One variable: add #846
as vLLM patch `0675` after candidate-36's `0001..0673`. The patch preserves its
upstream base, rationale, quality implications, tests and upstream status.

The current candidate-36 profiles select `spark-ds41f:candidate-36`.
The original `results.json` and `verdict.json` are retained byte for byte under
`original/`; their passing verdicts apply to the original image only. The exact
original source manifest, upstream lock and patch series are retained in Git at
`base.json`'s `base_commit`, alongside the source identifiers recorded there.
No serving cluster operation is part of this source update.

## Existing hardware evidence

[The original experiment](../2026-11-01-direct-exchange-switched/README.md)
already measured this exact candidate-36-plus-#846 source tree with a boot and
a repeat boot per arm on four GB10 behind a MikroTik CRS804-4DDQ: prefill
+3.2–3.4%, decode unchanged, identical greedy outputs and top-20 distributions,
and no host RDMA error growth. It retains every run and the identical disk-Engram
test failure in both arms. Switch drop counters were not captured. Those are
existing measurements, not a new qualification of the proposed image tag.

No new GPU run, memory observation or performance comparison is made here.

## Local verification

Commands and outputs are retained in `checks.json`:

```sh
bin/spark doctor
python3 -m unittest tests.test_source_locks
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp4.json build prepare --lock --only vllm
python3 -m unittest tests.test_source_locks tests.test_build
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp4.json doctor
git diff --check
```

The promoted configuration's doctor passed before the change; all 17 source-lock
and build tests passed afterwards. Candidate-36's TP4 doctor reports two inherited
errors: its frozen base supplies `NCCL_IB_HCA`, which the current renderer owns,
and it names baseline `2026-10-25-spark-ds41f-1025` while its lock names
`2026-10-30-spark-ds41f-1030`. All three values match the pre-change profile and
lock. They are not changed as part of the #846 backport.

The new vLLM identity is patch head `74e6e1920b3d`, tree `bc7c9cd5e022`,
559 patches. The tree matches the original #846 hardware measurements exactly.
The full stack was applied in a fresh checkout while locking these identifiers;
the helper's second preparation pass ran out of local disk space at patch
`0650` and removed its temporary checkout. One inherited intermediate patch
emitted a blank-line-at-EOF warning; the final upstream diff check in the
locking pass passed. The remaining preparation and verification moved to DGX4
at the owner's request; the failed local attempt is retained in `checks.json`.

## DGX4 verification

Published source commit: `27769d29dee779663e0e9495f07b4ea3d90a9e92`, checked out
cleanly and detached in `/home/swank/projects/spark-ds41f-candidate-36-switched`.
The main DGX checkout remained at `df50aa94efb5f7e897f9e1281a9e00c207dec762`.
The isolated checkout uses DGX4's existing build cache and a node-local copy of
its site map. Three setup attempts (missing map, unsupported command option,
then a map symlink outside the checkout) are retained under `runs/`.

[The successful run](runs/dgx4-verification.json) records:

- all 559 patches prepared in 32.836 seconds, reproducing patch head
  `74e6e1920b3d` and tree `bc7c9cd5e022`;
- exactly #846's two serving files and two test modules differ from original
  candidate-36; all four parse successfully;
- all 17 source-lock and build tests pass (1.454 seconds);
- the candidate doctor reports exactly the same two inherited errors;
- `git diff --check` passes.

The candidate has source inputs only; complete image build contexts were not
prepared, so full image `build check` and GPU tests are not run. No image is
built or deployed, and no serving operation is performed.

## Local disk cleanup

Removed 196 abandoned temporary Git pack files, reclaiming 71.21 GiB and
leaving about 97 GiB free. Indexed temporary packs were removed only when all
their objects were present in retained packs or loose objects. The cleanup
preserved the working files, index, refs, worktrees and canonical packs;
`git fsck --connectivity-only --no-dangling` passed. Exact inventory and hashes
are in the local `.work/disk-cleanup-2026-11-05.json` receipt; `checks.json`
retains the outcome summary.

Disabled this local repository's automatic Git maintenance (`gc.auto=0`,
`maintenance.auto=false`) to reduce the duplicate repack buildup. Previous
values are recorded in the cleanup receipt.

## Conclusion

#846 is added to candidate-36 under the existing name. Source validation passed
on DGX4; the updated image still needs building and hardware qualification.
