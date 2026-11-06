# Offline commands

All CPU-heavy jobs ran on DGX4 in `/home/swank/spark-lab/collective-pdl`, with
`CPUQuota=200%`, `MemoryMax=8G`, `TasksMax=64`, `Nice=10`. Compile libraries were
limited to two jobs/threads; BLAS to one. None of these commands runs a GPU
workload or performs a serving operation. The Mac only coordinated and relayed
the small published Git commit using its existing authentication.

The executed scripts, including dependency installation, source checkout,
validation and logging, are archived under [runs/commands](runs/commands/).
The vLLM reproduction script is [reproduce-source.py](reproduce-source.py);
it was executed from the identical file at the path below. These are execution
records, not idempotent setup helpers; preserve existing source preparations.

The following invocations reproduce the bounded execution of those scripts.
Unit names are task-local; the final invocation is recorded verbatim.

```sh
systemd-run --user --unit=collective-pdl-cpu --property=CPUQuota=200% --property=MemoryMax=8G --property=TasksMax=64 --property=Nice=10 /bin/bash /home/swank/spark-lab/collective-pdl/cpu-checks.sh
systemd-run --user --unit=collective-pdl-reproduce --property=CPUQuota=200% --property=MemoryMax=8G --property=TasksMax=64 --property=Nice=10 /usr/bin/python3 /home/swank/spark-lab/collective-pdl/reproduce.py
systemd-run --user --unit=collective-pdl-validate --property=CPUQuota=200% --property=MemoryMax=8G --property=TasksMax=64 --property=Nice=10 /bin/bash /home/swank/spark-lab/collective-pdl/validate-experiment.sh
systemd-run --user --unit=collective-pdl-deployment-checks --property=CPUQuota=200% --property=MemoryMax=8G --property=TasksMax=64 --property=Nice=10 /bin/bash /home/swank/spark-lab/collective-pdl/deployment-checks.sh
```

The primary source check is `make check` at sparknet commit `1a1205e47e928a80c7015dca3ce87d70f8753707`.
The fresh deployment source check is `bin/spark --cluster-config
experiments/2026-11-05-collective-pdl/tp3-triangle.json build prepare --only sparknet`.
Doctor ran against the base and each experimental profile; exact commands and
exit statuses accompany their logs. The base `build check` warning records
incomplete image inputs; it does not qualify a build. Interrupted Mac work is
recorded in `results.json` and is not counted as a passing run.

After staging the complete experiment, `git diff --cached --check` also passed.


## Extended offline checks

Current source `fb027b4` passes the same bounded `make check`. The archived
`prepare-alternatives.py` creates the two original tested commits; their
published Git Data API commits have exactly the same trees (see
`alternatives.json`). Both source alternatives pass 98 CPU checks and CI.

`compile-sources.sh`, `compile-alternatives.sh` and `compile-benchmark.sh` run
inside CPU-only containers (`--runtime=runc`, no GPU, no network, two CPUs,
8 GiB memory and swap limit, 256 process limit). The source compile helpers
assert generated-code ordering and produce native CUDA and hashed receipts.
All 84 source cases and both benchmark copy kernels compile. The first failed
inspection attempt is retained as `runs/cpu-compile.log`.

`prepare-current.py` reuses verified clean prepared inputs and runs canonical
`build prepare` and `build check` for the current source pin. The final check
passes; no candidate image has been built. Heavy preparation remains remote
and bounded by the same CPU/memory scope, using `/tmp/spark-source-preparation.lock`.
Compilation uses `/tmp/spark-cpu-analysis.lock`.

The new controller is plan-only by default. Its CPU controls run as:

```sh
COLLECTIVE_PDL_MAP=config/nodes.json /home/swank/spark-lab/collective-pdl/venv/bin/python experiments/2026-11-05-collective-pdl/test_source_gate_controls.py
```

Every applied source gate needs the exact owned window and clean published
checkouts on all ranks. Benchmark mode additionally needs that variant's
passing gate receipt and verifies each log hash. Applied hardware commands
will be recorded in their own per-run receipt; none has run yet.

### Native entry attempt and rendezvous repair (2026-11-06)

The exact three-rank gate plan is `runs/native-entry-20261106/gate-plan.json`;
`controller.log` retains the guarded C36 restoration. Its immutable gate receipt
names harness `e0630878` and source `fb027b4`; the repair does not rewrite them.
The CPU-only old-fails/new-passes Gloo fixture runs in the exact C38 image with
`--runtime=runc --network=none`, empty CUDA visibility, CPU 2, memory 8 GiB,
no extra swap and 256 PIDs. The fixture replaces only the GPU fault scenario
with a CPU collective and asserts CUDA remains uninitialized; it exercises
the real probe rendezvous. Its code and the bounded check scripts are archived
in `runs/commands/`, with hashes in `receipt-native-rendezvous.json`.
All three source variants passed `make check` (98 cases each); deployment
checks passed 755 cases and five source-controller controls. The revised source
was freshly prepared and both profile doctors passed. Full image preparation
is deferred until selection; the current build check correctly warns that the
new context is incomplete. No candidate image exists.
The controller now keeps its owned-window heartbeat alive through restoration.
