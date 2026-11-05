# Decision

Status: **retain for more evidence** (external site receipt)

The contributing site now serves spark-ds41f-1030 on the recipe's 7.0 64 KiB host with the memory saver, at 8,192
batched tokens. Against r6u on the same day:
- decode steps 8-10% shorter;
- aggregate throughput +4.7 to +10.2%;
- cold prefill +10-18%;
- qeval text unchanged.

At the same budget the 64 KiB host is speed-neutral, with about 2.4 GiB more MemAvailable per node. The site measured
the recipe's 16,384 too: 4-6% more prefill from 11K tokens and about 1% more aggregate throughput, against 8,192 on
the same host. It did not keep it, because a short request beside a
545K cold prefill then streamed at 2.3-2.9 s per chunk, against 1.3-1.5 s at 8,192.

For this repository the receipt is evidence of three things:
- 1030 runs on a stock 4 KiB host;
- tp4-ring holds on the 64 KiB host on GX10 boards with Secure Boot, after a site-key enrollment;
- what the 16,384 budget costs mixed traffic.

It is one boot per configuration, with the site's own battery rather than `bin/spark bench`, and no soak or agent
replay. Receipts: `runs/`.
