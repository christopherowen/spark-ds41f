# Small metadata placement screen

Purpose: establish the attributable component gain before requesting an expensive
serving round. This is a new candidate-operation experiment, not a remeasurement
of a released build. The eager control and all alternatives run the exact same
candidate source, integer kernel and input fixtures. No weights are loaded.

Compare four valid placements: the existing full eager host builder followed by
a query graph; a direct metadata-kernel call followed by the query graph; metadata
in a separate graph; and metadata inside the query graph (the implementation).
A fifth, static-consumer-only graph is explicitly an invalid implementation for
changing requests and only bounds the residual metadata-refresh cost. It helps
decide whether fusion into input preparation could still matter.

Use 1, 7 padded to 8, 8 and 16 requests, five query rows, and prefix lengths 256,
8192, 131072 and near 1M. Keep one metadata group so extra groups cannot inflate
the result. Each case checks every readable integer buffer and its consumer after
length changes, padding and slot reuse, poisoning outputs before every valid arm.
Compile the two builder specializations before freezing the kernel cache; capture
must succeed with that cache frozen. The existing three CUDA regressions also run.

Each of two fresh process boots takes 24 paired trials of 128 iterations, balances
arm order, preserves every raw sample, and reports paired bootstrap intervals.
Report enqueue time, completion time and stream elapsed separately; the latter
includes gaps while the host is enqueuing. Save driver and tensor memory and all
failures. A host-dispatch win is not yet a model latency or token-throughput win:
GPU work already queued ahead can hide it, and full affected-path validation is
still required. Do not extrapolate these synthetic consumer timings to serving.

Resources: Spaceport only, one GPU after its present job finishes, two CPUs,
8 GiB memory, two build workers, isolated per-boot caches and output paths. The
expected short screen is about five minutes; a cold compiler may take longer.
The active dgx1 hold currently forbids GPU experiments. Execute only after that
hold is released or the owner explicitly exempts this isolated Spaceport screen.
No serving-cluster operation is part of this screen.

Invocation inside the immutable candidate-38 image with the three committed
runtime files overlaid at `/opt/spark-ds41f/candidate/vllm/vllm`, source tests at
`/source`, and this committed deployment checkout at `/experiment`:

```sh
uv venv --python /usr/bin/python3 --system-site-packages /tmp/drafter-venv
/tmp/drafter-venv/bin/python -m pytest --noconftest --import-mode=importlib -q /source/tests/v1/attention/test_b12x_v41_ced_metadata.py -k draft_metadata_graph
/tmp/drafter-venv/bin/python /experiment/experiments/2026-11-05-drafter-metadata-replay/measure_metadata.py --boot 1 --output /out/measurements.json
```

Boot 2 uses a new container, new cache and output path, and `--boot 2`.
No result is available at preparation time.
