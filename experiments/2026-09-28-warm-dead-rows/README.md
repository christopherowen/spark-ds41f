# Dead-row kernel warmup

Base: the promoted configuration (`2026-09-28-karmic-kraken-r5i`).

## Question

On a freshly booted r5i, the first request took 568 ms to first token against
206-251 ms for the next three, and every rank logged two Triton kernels as
compiled during inference: `_combine_sampled_and_draft_tokens_kernel` and
`_get_num_sampled_and_rejected_kernel`. Kernel warmup runs with adaptive
verification off, so their DEAD_ROWS variants were never compiled before
serving. vLLM patch 0020 launches them at the end of warmup. Does the first
request stop compiling?

## Gates

`check.sh`: build r5j (`vllm-ds41f-kkref:04c30fa98e79-r5j`, vLLM tree
`bf8910a6`), start `cluster-candidate.json`, time four short requests
(`first_request.py`), require no "JIT compilation during inference" line on any
rank, and run the LRU gate. The change only moves compilation into startup, so
no decode or prefill bench is run for it.

## Results

2026-09-28, `check.sh` and one restart without a rebuild:

| Boot | First request, first token | Next three | Serving-time compilations |
|---|---|---|---|
| r5i, fresh boot | 568 ms | 206-251 ms | 2 per rank |
| r5j, right after the image build | 1244 ms | 171-218 ms | 0 |
| r5j, restarted | 579 ms | 202-212 ms | 0 |

LRU 5/5; lowest dgx1 MemAvailable 6.36 GiB during the gate.

No kernel compiles after readiness any more. The two compilations were not
what made the first request slow, though: without them the first request still
takes about 0.37 s longer to its first token than the next ones, from another
first-use cost not identified here. The 1244 ms run followed the image build and
copy, which leave the host page cache cold.
