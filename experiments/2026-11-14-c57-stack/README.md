# Candidate-57 Patch Stack Screen

Four vLLM commits cherry-picked onto `candidate-55` Build 9 (`44db6a403`):

```
3793961db perf(dsv41): branchless sparse MLA stage-2 online softmax over inactive segments
0ac8a53bb perf(spec_decode,sample): skip dead verification rows in sampler
2171c005e perf(dsv41): SP ratio-2 compressor projection and CED encoder-only chunks
dc712a66f perf(dsv41): fuse DSpark confidence calibration and keep active num_reqs in dummy runs
```
