# Two-request CED padding regression on a 64 KiB TP3 ring

This adds an independent regression for the routing-mask fix already supplied
by Carlos Molina in [PR #15](https://github.com/christopherowen/spark-ds41f/pull/15).
It does not change the runtime patch, promoted configuration, or source pins.

## Observed failure

Two production crashes on r6 each printed 32 TileKernels router assertions,
all at block `[66, 0, 0]`, threads 32–63. A temporary pre-router capture then
reproduced the failure with one decoding request and one prefilling request.
Both the image-containing workload and a workload with **zero images** failed.

At decoder layer 20, all three ranks had the same spare compacted row:

- CPU plan: six scheduled decode rows plus 128 retained prefill rows = 134.
- Actual GPU decode range: five rows, so only 133 compacted rows were valid.
- CED index for row 133: `-1`; the old router mask still marked it active.
- Router projection input at that row: 5,120 NaNs, already present before
  projection, scoring, or route scaling.

The observed query offsets were `[0, 5, 374]` and `[0, 5, 1636]`, corresponding
to 369 and 1,631 actual prefill rows. The capture localizes the mask error;
it does not identify the first operation that made unused padding non-finite.
No prior NaN/Inf warnings appeared between the last successful POST and the
first assertion in either original log. Minimum observed available host memory
in the instrumented reproductions exceeded 5.8 GiB.

The four-image allowance is not necessary to trigger this bug. A 13-image
history completed alone, while the text-only overlap also reproduced it.

## Regression

`tests/test_two_request_routing.py` constructs the real CED index map for both
observed prefill lengths and actual decode counts 1, 4, 5, and 6. It checks:

1. CED's scheduled upper bound leaves the expected `-1` tail.
2. The installed decoder router uses the gathered padding mask, including
   interior dead verification rows.
3. An encoder with the same row count still uses the original layout.
4. The real TileKernels router ignores NaNs in masked rows, returning expert
   `-1` and weight zero, with bitwise unchanged outputs on live rows.

The mask assertion deliberately precedes the NaN kernel call: unpatched r6
fails cleanly without poisoning the CUDA context. No model checkpoint or
private conversation is needed for this test.

Run inside the recipe image on an idle GPU before loading model weights:

```sh
python3 -m pytest --noconftest -q tests/test_two_request_routing.py
```

## Controlled variable and provenance

The deployed recipe was `dbae4e1b80d10aa4f18bca4d1eaf72410489e4bd`, image
`vllm-ds41f-kkref:04c30fa98e79-r6`, vLLM tree
`20c7c7586ba62e237a4157fd741974ca5b7624be`. The fix was the unchanged
`0045-tilelang-ced-decoder-routing-mask.patch` from PR #15, SHA-256
`9fa2192cb1baf0d3dccde326c098f881dd000354117de25b172539c35bdaf2a0`.
The four changed runtime Python files were read-only mounts in a separate
three-rank canary container. No diagnostic hooks were present in the fixed run.

All serving features remained enabled: 524,288 context, eight sequences,
4,096 batched tokens, prefix caching, full/piecewise CUDA graphs up to 48 rows,
async scheduling, DSpark five-token adaptive verification, and the expanded
image allowance. Request sampling stayed at temperature 1 and top-p 0.95.
The host used the 64 KiB NVIDIA kernel and the recipe's memory-saver module.

The three pre-existing image IDs were not identical, although their pinned
source labels agreed. Each node kept its own original image throughout;
this was a patch-isolation experiment, not a new image qualification.

Private request bodies, tensor dumps, and transcripts are retained locally
and deliberately excluded from this contribution. Aggregate receipts and
limitations are recorded in `results.json`.
