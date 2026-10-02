# Decision: retain R580 and the original r5o image

Do not promote either candidate. The completed baseline/corrected-driver/CUDA/
baseline sequence shows no consistent end-to-end benefit large enough to justify
the additional driver maintenance and incomplete CUDA build alignment. This is
not a claim that every workload is unchanged or that a fully rebuilt newer stack
could not improve performance.

The second R580 control matters. Corrected R610's eight-stream prose gain falls
from 5.1% against the first control to 0.1% against the second. Eight-stream code
answers change from 2.2% faster to 4.5% slower. Reasoning code and prose answers
still favor corrected R610 by about 3% and 4–5%, respectively, but three samples
and one candidate boot do not resolve a general driver benefit. Source-text
prefill is approximately level. Full values and intervals remain in the reports.

CUDA exp2 changes eight-stream throughput by -2.9% to +3.0% relative to corrected
R610 with the original image. Its single-stream estimated step times range from
0.6% lower to 2.0% higher. It also has an unresolved long-prefill slowdown in one
sample. The experiment updates selected shared libraries and Triton's assembler;
it does not rebuild PyTorch, B12X/CuTe or native extensions against CUDA 13.3.

Stock R610 is unsuitable for this tested 64 KiB configuration: large-copy faults
reproduced on all three machines, and on dgx3 with stock UVM. A separate signed
RM alignment correction was necessary. Default system memory pools also retained
freed pages and blocked vLLM's startup free-memory check; the successful arm
disabled them. Those dependencies must accompany any future R610 qualification.
The memory-saver R610 branch remains unpromoted and explicitly documents that it
does not fix the RM defect itself.

The restoration uses the exact cached R580 packages and signed memory-saver
0.2.0, removes the RM override and pool setting, and retains the 64 KiB kernel,
3.5 GiB KV per rank and 524,288-token limit. Production configuration, image,
model weights, upstream pins and source patch series are unchanged. Experiment
images and cached packages are retained for reproducibility.

Restoration completed on all three nodes. `doctor --live` reports configuration
OK and a matching live cluster, the final LRU screen passed 5/5, and every rank
runs the original image ID `sha256:288fc5bd909eb7e5fa51bd5abd94286140a0f07c40b92d763f8be4970309e5ab`.
The experiment's pinned-cost, assembler-override and batch-invariant environment
variables are absent. Receipts are `runs/restored-production-quality.json`,
`runs/restored-production-doctor.txt` and the three restored host inventories.

The unchanged upstream build inputs passed `bin/spark3 build check`; CLI/host
policy tests passed 127/127. After moving the new tests before the direct test
entry point, that file also passed 23/23 when invoked directly.

Revisit when the packaged driver includes the alignment correction or a specific
newer compiler/library change offers a measured benefit for our serving kernels.
Use repeated boots and matched workloads to resolve any small gain before a full
quality/capacity qualification. No additional benchmark campaign is scheduled.
