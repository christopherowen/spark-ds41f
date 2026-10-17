# Decision

Status: **retain for more evidence** (external site receipt)

The contributing site made this configuration its default after these runs: shorter decode steps (36.7-39.1 vs
43.4-46.9 ms), +4..+25% decode throughput, +20..+57% salted cold prefill and a passing 993K-token needle, at the cost of
one extra qeval miss (`json_count`, reproducible) and a few percent at C16 on coding / math / format.

For this repository it is evidence that `tp4-ring` runs on a stock 4 KiB kernel at the recipe's sizing with 18-21 GiB of
headroom per node, and that r6u needs r6v's 0180 to boot on a ring. It is one boot per engine with the site's own battery
rather than `bin/spark bench`, and has no soak or agent workload. Receipts: `runs/`.
