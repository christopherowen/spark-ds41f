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

Pending.
