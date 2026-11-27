# Decision

Status: **retain for more evidence** (external site receipt)

The contributing site now serves spark-ds41f-1126 on `tp3-switched` (three GX10 on a switch) and keeps
`tp4-switched` ready to switch on.

Against 1107 on the same three nodes, the same day:
- prefill is 6-13% faster cold, and 31% and 57% faster on the ~545K and ~990K needles;
- C16 aggregate is 57% higher (430 tok/s, TTFT 0.7 s against 2.7 s). That gain comes mostly from tp3-switched's 16
  sequences against 1107's 8; C4 is 5.5% lower;
- quality is equal or better: qeval 72/75 against 70/75, vision and tools 7/7, and the DSML fixtures reproduce the
  site's earlier token IDs.

tp4-switched is a further 22-34% faster at every concurrency, with 16-19 GiB MemAvailable per node. One DSML
fixture produces different tokens there, so the site keeps tp3 as the default for now.

For this repository the receipt is evidence of four things:
- 1126 runs on both switched shapes with GX10 boards;
- 1126 refuses a threshold that worked on 1107 at the same budget;
- `engram_on_disk` needs up to 115 GB per node for both shapes, and a full root disk breaks stop and rollback;
- `0768` reproduces the DSML token IDs of the site's earlier overlay on tp3.

It is one boot per configuration, with the site's own battery rather than `bin/spark bench`, and the site's C1
overlay in every run. There was no soak or agent replay. Receipts: `runs/`.
