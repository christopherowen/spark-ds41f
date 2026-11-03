# Candidate-38 qualification on TP3 triangle

Base deployment: `fb89f24ad50fc3c79bb7b532e796c4f179d08bca`.
Intended variable: the complete selected candidate-38 image/recipe versus the
recorded spark-ds41f-1030 TP3 release. This qualifies the combined candidate;
it does not attribute any difference to an individual patch. The standard and
CopyDrafter recipes are reported separately, each with a boot and repeat boot.

The owner authorized building on dgx4 and testing on dgx1–3 using the normal
regression workflow. The current serving profile to restore is
`experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json` on the
same TP3 triangle. No topology change is intended.

## Image and inputs

The final candidate was already built on dgx4 at deployment `df50aa94`:
`sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b`.
Its dgx4 build receipt records 466 seconds, eight compile jobs and 78.1 GiB
minimum available memory. Current `build check` passes for the complete prepared
inputs. Reuse this immutable build and its completed image/release-test gates;
copy it unchanged through `bin/spark build distribute`, then independently check
all three image IDs, source labels, layer lists and Config hashes.

Source inputs and profiles remain those of
[the selected candidate](../2026-10-26-spark-ds41f-candidate/candidate-38/README.md).
No source overlays, effort ceiling or new tuning variable is added.

## Workloads and commands

Publish one commit, open a lab window against the verified serving profile, then
synchronize all three clean node checkouts to that exact published commit using
`bin/spark cluster sync --revision COMMIT --apply` with the restore profile named.
The specs deliberately contain no sync job that could move to a later commit.
Inspect start plans' live and shape lines before the first boot.

Run `functional-default.json` and `functional-copy.json` first. It retains the existing candidate's functional
matrix and CopyDrafter probe, repeats each profile on a fresh boot, and collects
all-rank image/state/log receipts after each boot. Review every actual script
exit, request/digest result and rank log before proceeding; an overall lab exit
is insufficient.

```sh
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/functional-default.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/functional-copy.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/release-default.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json --keep-open
python3 scripts/lab.py run experiments/2026-11-03-candidate-38-tp3-qualification/release-copy.json --production-config experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json
```

The performance specs run `release-tp3`, `single-stream` and `sampled` on the
same boot of each arm, then repeat that arm on a fresh boot. Each profile has
its own table and paired verdict; different profiles are not pooled as repeats.
Use the immutable TP3 suite references under
`manifests/suites/2026-10-30-spark-ds41f-1030-tp3{,-single-stream,-sampled}`.
The released image is never remeasured. Retain suite conditions, boot/startup
memory, telemetry, all failures and the final serving restoration checks.

Judge fixed-row engine cost, paired one-/four-/eight-stream throughput, sampled
throughput, prefill through one million tokens, mixed-prefill stalls and first-token
latency against the recorded release and its A/A floor. Inspect confidence intervals
and the candidate's own repeat before calling a regression. A failure or unresolved
regression keeps the candidate unqualified; record it before deciding any follow-up.

## Preflight observations

All three nodes initially have clean checkouts at `3adababa` and the same running
candidate-36 image `sha256:b0522146640a7d852c017ec8c99423224492b7c2c4951eb3a64005ab4555ab11`.
Live doctor matches the restore profile with two existing recommendations:
management ports off wireless and model-weight defragmentation on dgx2/dgx3.
Do not change host settings within the comparison. No hold was present on dgx1
or dgx4 during initial read-only inspection. The site's ignored node map is
preserved and is not replaced with this workstation's same-named file.

Status: prepared; runs and errors will be recorded when executed.
