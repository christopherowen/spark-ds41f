# Candidate 36 topology profiles

Base and exact rename map: `base.json`. Candidate-36 keeps the image name
`spark-ds41f:candidate-36`, its source lock and its 559-patch #846 stack.

The base switchless profiles are `tp3-triangle.json` and `tp4-ring.json`.
Their derived profiles and prospective lab job names identify the triangle too;
all active references are updated. Earlier run receipts retain their original
commands and names at their recorded commits.

`tp3-switched.json` through `tp8-switched.json` are thin overlays on the frozen
spark-ds41f-1030 recipe at each switched shape. Only candidate image/source
identity is overridden. The recipe owns sizing, node-map selection and switched
NCCL settings; no runtime recipe is copied per shape.

The ring profile explicitly uses baseline `2026-10-30-spark-ds41f-1030`, matching
its lock, and removes the inherited manual `NCCL_IB_HCA` so the pinned renderer
selects adapters from the site map. Historical parent recipes are unchanged.

## Validation

[The successful DGX4 run](runs/dgx4-verification.json) records:

- all 13 profiles pass the doctor's core configuration checks;
- 56 per-rank environments render successfully; switched ranks use direct
  topology, traffic class 106, both device paths, the shared switched NCCL values
  and no inherited ring controls;
- image, source lock, expected tree, baseline and tensor-parallel size agree;
- the resolved triangle is identical to its previous profile, and the resolved
  ring differs only by the two intended validation corrections;
- all three renamed lab jobs produce plans with existing triangle profiles;
- all 64 recipe, switched-topology, source-lock and build tests pass;
- the ring profile's CLI `doctor` and the staged whitespace check pass.

The offline fixtures use the committed example maps with the head's management
address bound to the profile's rendezvous address. Actual site overrides must
keep `distributed.master_addr` equal to the head's address in the node map.
These checks validate configurations; they do not probe a physical network.

The first verifier setup attempt missed the repository import path; the second
used an example head address that differed from the profile's rendezvous address.
The setup failure and the second attempt's findings are preserved under `runs/`.

Commands:

```sh
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp4-ring.json doctor
python3 -m unittest tests.test_recipe tests.test_switched_topology tests.test_source_locks tests.test_build
git diff --cached --check
```

The source patch series is unchanged from
[the verified #846 backport](../2026-11-05-candidate-36-direct-exchange/README.md).
No image build, GPU experiment, cluster sync or serving operation is part of
this change. New profiles require site node maps and hardware qualification
before any promotion.
