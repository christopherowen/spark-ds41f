# Cooperative summaries with wide coarse selection

Base `e607aea8ef32fb55fae665ddaf95dc0e0445cbd0`. One variable: integrate the byte-preserving512-thread cooperative summary kernel into the raw-wide source1024/dense64/native-generation recipe. All selection/scoring code after the summary definition is byte-identical to the parent; the kernel AST equals the previously screened512-thread implementation. No new approximation, token policy, query mask, centroid layout or capacity is introduced.

Fresh thirteen-patch application at the C36 prepared upstream head matches every affected overlay. Exact commands and hashes are in `runs/local/source-check.json`. Preparation is remote, one CPU/4GiB, one worker; no complete image inputs. Runtime checks must include full-pool byte equality against the previous serial kernel, actual/synthetic scales and skip/recycle behavior under init/race checking, frozen declarations, retained score bits, strict suffixes and graph lifecycle. Prior reference code is mounted only in out-of-band test bundles. Two component timing processes and repeated final-answer/full-request gates remain separate requirements; no gain or model quality is claimed here.

Untimed standalone GPU correctness gates (`summary-initcheck`, `summary-racecheck`, `registration`, `actual`, `initcheck`, `graph`) now pass on `sp8`; see `results.md` and `runs/runtime-gates-01/`.
