# C42 stopped at quality gate

C42 is rejected for a retained IKET feature failure. Performance was not measured.

| Build | C1 output tok/s | C4 output tok/s | C8 output tok/s | Raw C8 mean vs1030 | Verdict |
|---|---:|---:|---:|---:|---|
|1030|62.63|121.21|158.60|reference|Saved immutable release|
|C36|65.46|125.67|163.03|+2.79%|One compatible saved boot|
|Validated base C40|65.38 /65.37|126.10 /126.17|163.41 /163.59|+3.09%|Correctness retained; no proven serving gain vsC39|
|Last measured C41|65.36 /65.34|126.08 /126.10|163.69 /163.53|+3.16%|Rejected core costs; serving neutral vsC40|
|Current C42|not measured|not measured|not measured|not measured|Failed required IKET gate|

Slashes separate boot/repeat. Raw output-window means and their changes are
context, not incremental performance proof. Prompt-paired per-user decode
estimates are separate statistics. C36 has no saved compatible repeat; no
interval is invented. Existing C40/C41 findings, both floors and adverse rows
remain in their immutable native publications and derived reports.

C42's complete136-file gate has8073 passed,360 skipped,1 failed,0 errors; the
one failure is an explicit prepared-kernel lookup under IKET cache-disabled
mode. The local source witness reproduces the exact native refusal without
importing TileLang/TVM/Torch or running a GPU. Metadata-only C44 cannot fix it
and is withdrawn unmeasured. C42 actual wait exit1, ended2026-11-05
21:33:57UTC. The source/control commit is b6896b389c5bb0004e13fe9f4f941a31a60f87ca;
image sha256:70e9abba12d83b79a268574a2057b096af2b9fc5fcbc1b07364d6eb580b1a35a.
No seeding, functional boot, metric boot or top-k/feature gate followed the
failed release gate. All29 builder/gate and21 head artifact hashes verify.
Independent named C36 doctor/ranks passed with the same preexisting containers,
imageb052... and profile, zero restarts, both native processes absent. The
closure request and its fulfillment remain separate original receipts; the
actual stop was a test failure, not scheduled sharing.

See failed-gate-audit.json, failed-native-publication.json and [cache review](../../preflight/cache-provenance-review.json) and
[IKET review](../../preflight/iket-retained-feature-review.json). Original native failure/control JSON
is published byte-for-byte in failed-native/. Raw logs/XML remain private in
failed-export/ with every artifact hash retained. Build/gate conditions are
not serving thermals or performance. Do not rerun this rejected image.
