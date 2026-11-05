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

Full build contexts and wheel inputs are absent locally, so complete image
`build check` and GPU tests are not run. No image is built or deployed.

## Conclusion

#846 is added to candidate-36 under the existing name. Source validation is
recorded here; the new image still needs building and hardware qualification.
