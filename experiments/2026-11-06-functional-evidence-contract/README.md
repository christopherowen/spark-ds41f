# Explicit functional evidence before timing

Base: `2c4b23afcd834ba2adf6a15ca434e59fa4f49f74`.
This is a CPU-only validation change; no runtime configuration, image or upstream
patch changes. The intended variable is whether declared collector requirements
are checked before a functional phase can advance to timing.

The shared lab previously used collector process exits without interpreting
functional verdicts. A collector can validly exit zero while reporting an
observational difference. A consumer requiring a particular verdict must declare
and enforce it. The new contract distinguishes requirements from observations,
rejects incomplete evidence and preserves the runner's existing restore behavior.

C43 adopts explicit contracts for its three greedy gates and sampled isolated
repeat, plus complete private sampled recordings. Its sampled batch differences
remain observations. C40's historical audit already recorded these differences
as a limitation and required only the isolated repeat; it did not claim sampled
batch invariance. C40/C41/C42 specs and all published evidence remain unchanged.
Other collectors still use their existing exit/audit contracts; this does not
claim that every functional property is covered by the new declarations.

The separate verification-budget attempt in [PR 917](https://github.com/christopherowen/spark-ds41f/pull/917)
failed its then-declared sampled batch requirement and restored C36 before release.
It remains failed, with no timed result or candidate qualification. This change
neither diagnoses a runtime sampling bug nor retroactively passes that attempt.

## CPU proof and validation

`run_cpu_proof.py` runs real short-lived collector subprocesses against the base
lab and the new lab, with cluster/window/timing actions mocked. Identical inputs
exercise false, missing and malformed required evidence, a failed required
isolated repeat, observed batch differences and a valid required verdict. It
records whether the timing step was reached and whether failure restoration was
requested. A request/response fixture also exercises all 24 sampled records,
separate token/logprob differences and manifest tamper detection in the unit tests.

Validation passed on Spaceport8 (`spark-0016`) at clean published implementation
commit `2176e3d60283ae6a2f3165be5f84dcd6939c74c9`, on 2026-11-06 at 01:53 UTC.
All 146 tests passed: 15 evidence/recorder, 114 lab, 9 C43 controller and 8 C40
compatibility tests. `bin/spark doctor` returned `configuration OK` and
`git diff --check` passed. No upstream series changed. The owned checkout has no
complete `inputs.json` build preparation, so a full build check was not applicable.

The [counterfactual receipt](runs/2176e3d6-attempt-1/before-after.json) preserves
all six old/new cases, collector output, runner logs and new evidence receipts:

| Input from a zero-exit collector | Base reaches timing | New reaches timing | New requests restore |
| --- | --- | --- | --- |
| Required verdict false | Yes | No | Yes |
| Required verdict missing | Yes | No | Yes |
| Required verdict malformed | Yes | No | Yes |
| Required isolated repeat differs | Yes | No | Yes |
| Observed batch difference, isolated repeat matches | Yes | Yes | No |
| Required verdict passes | Yes | Yes | No |

These are CPU fixtures, not serving measurements. The old runner did not consume
the new declarations; this demonstrates the missing enforcement mechanism, not
a historical audit falsely claiming sampled batch invariance. No validation
command failed. An initial read-only inventory used the unavailable SSH alias
`spaceport8` (name resolution failed before connection); `sp8` is the valid alias.

Every command, exit, log and `/usr/bin/time -v` resource observation is retained in
[runs/2176e3d6-attempt-1](runs/2176e3d6-attempt-1). The highest recorded maximum RSS
was 41,708 KiB (lab tests); every command recorded zero swaps. The counterfactual
used 37,456 KiB and took 0.176 seconds internally. There was no GPU operation.
The commands ran serially under:

```sh
systemd-run --user --scope --quiet -p MemoryMax=2G -p CPUQuota=100% \
  taskset -c 0 flock -w 60 /home/chris/spark-lab/offline-cpu.lock \
  flock -w 60 /home/chris/spark-cpu-work.lock bash -s
```

Working directory: `/home/chris/spark-lab/verification-budget-replay/checkout`.
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MAX_JOBS=1`.
The Python executable is `/home/chris/spark-lab/verification-budget-replay/env/bin/python`
(3.12). The `.command.txt` files contain each exact command; for example:

```sh
/home/chris/spark-lab/verification-budget-replay/env/bin/python \
  experiments/2026-11-06-functional-evidence-contract/run_cpu_proof.py \
  --output /home/chris/spark-lab/verification-budget-replay/logs/functional-evidence/2176e3d6-attempt-1/before-after.json
```

Conclusion: declared failures now stop advancement and request the existing
restoration path, while declared observations retain their meaning. Request
recordings are checked for complete case coverage, IDs, generation controls,
token/logprob lengths, hashes and agreement with the collector's aggregate
digests. This is a validation/diagnostic improvement with no serving-speed claim.

## Future row-cost qualification contract

Before another hardware turn, a new plan must declare the intended requirements
and workload. Greedy token/text equality and native logprob equality remain
required. Sampled isolated repeats require identical request controls; batch
trajectory differences are observations unless the full native RNG inputs and
step grouping can be shown identical. HTTP request seeds alone cannot establish
that condition. Private per-case recordings preserve diagnosis material but do
not expose native logit positions or speculative steps.

Sampling quality still needs the existing release/quality gates and appropriate
distributional or teacher-forced checks, alongside the source/math tests. An
observed batch difference is neither a sampling-quality failure by itself nor a
quality pass. Timed work and a repeat boot can begin only after the predeclared
requirements pass. Compare against recorded reference runs; no reference rerun
or new hardware window is authorized by this CPU change.
