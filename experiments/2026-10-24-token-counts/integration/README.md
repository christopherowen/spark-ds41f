# Qualified runner integration

The token-count port is now qualified with the runner-input port. All **539 tests
pass** across the runner, block tables, KV zeroing, dead rows, InputBatch, sink
recovery, DFlash inputs and kernel preparation. This includes the 19 pre-existing
InputBatch tests deselected in the standalone qualification.

All 32 focused count cases also pass under CUDA 13.0 compute-sanitizer memcheck
with **zero errors**. All ten paired count-performance cases pass the unchanged
no-slowdown gate, on a fresh seed and allocations. Serving-sized calls take
0.957–0.969 microseconds versus Triton's 1.067–1.072, about 9–11% less time.
The benchmark measures the count kernel, not other runner kernels or model TPS.

[results.json](results.json) keeps the timings, intervals, image ID and every
bundle hash. [run/](run/) contains the native receipts, and the exact runnable
[bundle](bundle/) repeats the focused memcheck, paired benchmark and full suite.
Every source file mounted in the bundle matches the committed source.

## Source and reproducibility

The runner's warmup API is now `register_input_batch_warmups(dead_rows=...)`.
The port preserves that API, declares both index types for each enabled dead-row
variant, and updates the owner-declaration test. The last Triton dispatcher
imports are removed; the original kernel remains the comparison reference.

The runner branch forked before candidate 4's final GLP change, which is already
in the image used for all the kernel windows. The integrated source merges that
base so it represents the image plus overlays exactly. All 15 files changed from
candidate 4 are mounted in the bundle and match the integrated tree; no unmounted
runtime difference is hidden by a source label.

- Integrated source: `7a7f09dcdab7eba530f64351018703986d6d8faf`.
- Runner component: `1bc9f13f620c12ea07b5fd0d8542d1758fb81825`.
- Token-count component: `78c4ae1283cc8b9836cfaa81169d19a63bbc82f7`.
- Qualified tree: `31cc6045d59e7c931c01fa0fb27d62de1cfe12d8`.
- Fresh replay head: `8329a68e108cff211c199b30afea04e532aa104d`.

The complete [230-patch series](vllm/series) replays from pinned upstream
`19f2c20ed4d64ae22bd009ceacadd85e2a81e354` into exactly that tree. The additional
patches carry the existing runner port and fixes before the count migration.
[Source metadata](source.json), the [source lock](upstreams.lock.json) and a
[launch-disabled build profile](candidate-tp4.json) describe the final source.
Candidate configuration, Ruff and whitespace checks pass. [Test classification](release-tests.json)
reports 129 selected files, 72 gates and zero problems; the complete release gate
and full-model serving comparison remain outstanding.

The [original adapter patch](after-runner-port.patch) is retained as an alternative
when applying only the count change after the runner component. Its
[component identities](adapter-source.json) describe its original base, before
the candidate-4 reconciliation. Do not apply it as well as patch 0273 from the
complete series. The full series and source manifest above are the build inputs.

## Service state

The coordinated window opened at 13:11:19 UTC on 24 October 2026. Production was
restored and passed `doctor --live` at **13:14:31 UTC**; the hold was removed.
The promoted recipe and source lock are unchanged. No release image was built
or promoted. These results qualify kernel integration, not production release.
