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
