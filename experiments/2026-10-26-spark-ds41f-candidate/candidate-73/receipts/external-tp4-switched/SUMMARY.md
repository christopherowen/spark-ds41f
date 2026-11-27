# Candidate-73 TP4 switched qualification — NO PROMOTION

## Result

The full two-boot protocol was executed on four DGX Sparks, 64 KiB kernel, switched dual-rail fabric. **Correctness passed; performance promotion clearance did not.** The candidate was confined to port 8889 throughout. Known-good 1120 was restored to production 8888 and verified externally with real completions on both aliases. No steering or dev mode in production. Gateway stayed alive, health reports 0.23.4.

## Identity and exact configuration

- Upstream candidate-73: `24950b3ecaab9eeb2946116651a81380a3892cc6`.
- Site checkout during final qualification: `353f249c5200768712f3860b5a0b543687cca7ed`.
- CFG: `experiments/2026-10-26-spark-ds41f-candidate/candidate-73/tp4-switched-dev.json`.
- Image on build rank: `sha256:03bf46ab96da9e4ab7343d4c7ad1abbb63e6e03e2879e196217ec2f9f8467ee2`. All ranks' ordered RootFS.Layers and full Config matched; loaded image IDs differ on two ranks, equivalence was checked rather than assumed from labels.
- vLLM tree `0aa091fa2b3dc5ab239816682a4b0f84e209cc58`, sparknet tree `1739407706277142da7a12c95fca3c503c8cb11c`.
- Kernel `7.0.0-1019-nvidia-64k`, page size 65536 on every rank; stock UVM.
- Candidate source locks/sizing retained: TP4 switched, 1M context, 32 sequences, 16.5 GiB KV. Site delta: home/paths, port+ready URL 8889, compatibility alias, root-owned compaction helper.
- GLP-27 pinned revision/checksum retained, alpha2.0. An unauthenticated download was denied; the existing authorized fleet token successfully downloaded the exact vector. No new license acceptance. The vector itself is not published.

## Verified protocol

`verify_receipts.py` asserted all 20 ordered phase receipts. Only `bench-b1` and `bench-b2` returned nonzero (both rc1, comparator regressions). Driver rc0 does NOT mean all gates passed.

| Gate | Boot1 | Boot2 |
|---|---|---|
| Live doctor | rc0, recommendations retained | rc0, recommendations retained |
| Wide determinism, 192 tokens/700 lines, with logprobs | identical, zero token/logprob differences | same |
| Wide determinism without logprobs | identical, zero differences | same |
| Tool gate, stock+weightless at0.0/0.3 | strict80/80,0 derailments | strict80/80,0 derailments |
| Full benchmark quality |5/5|5/5|
| Full benchmark source-text matrix |rc1, regression comparator|rc1, regression comparator|
| Release suite including grid and all1M prefill repeats |rc0|rc0|
| Single-stream suite |rc0|rc0|
| Sampled suite |rc0|rc0|

Two-boot release model comparison: **256/256 greedy responses identical, 65,536 teacher-forced positions, 100% identical distributions, KL mean/p99/max0**. `table-release.txt` carries the comparison.

Full boot1 ready94.8s; full boot2 ready91.6s. All suite thermal slowdown counters0; no thermal halts. Host fan service absent on all ranks, firmware automatic; recommendations are kept in doctor/conditions receipts. Release boot1 reported0.03GiB swap growth on rank0; other recorded release/suite swap growth0. Full matrix minimum memory stayed above10GiB. Final stop generated TCPStore/NCCL monitor warnings after coordinated shutdown began; raw warnings are retained, not called runtime correctness failures. Rollback running logs had no matches in the targeted fatal-error scan.

## Performance: matching topology/harness, different site

The baseline is upstream's frozen1120 `tp4-switched` reference, NOT our prior pico-harness chat80.4/code147.5/distinct381.9 numbers. Those unrelated measurements were not mixed into this comparison.

Two candidate boots versus frozen1120, using upstream's own paired table:

- Release c16 paired decode **−6.01%**, output **−6.06%**; measured263.52/263.82tok/s versus recorded280.51.
- Release c8 paired decode **−1.99%**, output **−2.41%**.
- Grid decode time geometric mean **+2.14%** (slower).
- Release source prefill about **−5% to−7%** across32K–1M; all repeats completed.
- Sampled c16 paired decode **−5.93%**, output **−6.14%**.
- Single-stream paired decode **−1.53%**; output **−2.09%** is within its uncertainty and judged same.

Both source-text full matrices independently flagged substantial decode/pre-fill differences. These results do **not establish an engine-only regression**: the frozen reference is from upstream hardware, and no same-site1120 control was run. Site/fabric/host differences remain possible. That uncertainty is a reason to withhold unattended promotion, not to describe the candidate as proven faulty.

## Exploratory runs, retained without cherry-picking

- Initial doctor failed because site base had removed GLP while an intermediate candidate overlay supplied only GLP generation defaults. Added complete candidate GLP metadata in base, then final stock-only unset overlay where appropriate.
- First non-dev stock boot completed a full matrix and failed comparison. Its filler-vs-source prefill rows are **not comparable** and are not used as a gain/regression claim. `initial-nondev/` includes the matrix, doctor, gates, exit receipt.
- A stock-dev exploratory boot was interrupted during its matrix once existing authorized GLP access was located. `exploratory-stock-dev/INTERRUPTED.json` gives the reason. It is not counted as a completed qualification boot.
- The frozen switched how-to omits the dev overlay, but the release grid explicitly requires `VLLM_SERVER_DEV_MODE=1`; final full boots use the upstream dev profile on8889 only.

## Restoration

Exact `site-1120` commit `cc3fe02cda60bf7e2ca96b170bcc4226fd8a8f4e`, image `spark-ds41f:1120` restored on every rank. RootFS/Config of rollback images had been verified before use. Startup rc0, ready125.6s. `verify_production.py` verified clean identical checkouts, exact per-rank image IDs, running containers with zero restarts, active guards and >5GiB available memory. Both production aliases returned exactlyOK, steering alias absent, qualification port closed. `production-restored.json` is the readback. Installed watcher constants and production runbook were not changed because no promotion occurred.

## Evidence

`full/` contains both boots' complete benchmark and suite artifacts, including model distributions; `full-node-logs/` holds engine logs. Build/preparation/distribution, per-rank image/environment receipts, site diff, decisions and comparisons accompany them. Original evidence remains local. The public receipt copy redacts site IPs/hostnames/paths, scans for secret-shaped tokens and excludes credentials, vectors, git bundles and private node maps.
