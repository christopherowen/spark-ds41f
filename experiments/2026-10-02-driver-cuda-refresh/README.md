# Driver and CUDA refresh qualification

**Decision: retain R580 and the original r5o image.** The completed four-arm
screen found no consistent upgrade benefit. See [results](results.md) and
[decision](decision.md); compatibility failures and successful controls are
preserved alongside the candidate build and restoration procedures.

User-authorized evaluation on all three Sparks, from the exact base in
`base.json`. Production uses R580, 64 KiB pages, memory-saver 0.2.0, r5o,
3.5 GiB KV per rank and 524,288-token context. Keep those capacities and the
5/3 GiB startup/steady guards throughout.

Compare the current container on R580 and R610 first, then a separate CUDA
13.3 container on R610. Freeze DSpark costs across the arms. The R610 allocator
port changes surrounding source context only; its hardware qualification is
part of the driver arm. The candidate kernel policy belongs to this experiment;
`host.kernel_policy` lets doctor and start validate it without changing defaults.

Before mutation, cache the exact replaced R580 packages, simulate the R610
transaction, and check the enrolled signing key. Use Ubuntu precompiled module
metapackages for both kernels; do not introduce NVIDIA DKMS beside memory-saver.
Stop the entire cluster before changing drivers. Build/install the pinned port,
reboot all nodes, check loaded identities, and run allocation/readback, copy,
matmul and graph smoke on each node before serving. Record failures as well as
successes. A failed management path requires diagnosis before another start.

Screen quality, one/eight-stream prose and code decode, and real-text prefill
at 4K, 16K and 64K. Compare step time, draft acceptance and TPS separately.
For small effects repeat the control. Promotion additionally needs long-context
and admission checks, matching node inventories, boot-log review, docs and a
new immutable baseline. Until then, restore the current R580 production stack
and close the hold. Native reports and compatibility receipts are in `runs/`.

## CUDA library/compiler screen

The derivative image changes cuBLAS to 13.5.1.27, CUDA runtime to 13.3.29,
NVRTC to 13.3.33, preserves the existing nvJitLink 13.4.52, and explicitly selects the 13.3.33 assembler
for Triton. Hashes and architecture-specific wheel URLs are recorded. PyTorch,
vLLM, B12X, CuTe DSL and native extensions remain exactly r5o. The existing
13.4 nvcc wheel is preserved; the Triton assembler is installed separately.
This is an ABI-compatibility experiment, not a fully rebuilt CUDA 13.3 stack.
The original cuda-toolkit metapackage pins older libraries, so promotion of this
arm would also need aligned package requirements/build inputs. Record this
metadata mismatch explicitly; do not describe a successful import as full
support. Verify actual loaded library paths/versions and freshly compiled
Triton artifacts before interpreting timings. B12X's CuTe compiler is unchanged.

## R610 compatibility failure and separate RM correction

All three nodes passed 3,584 small allocations/readback after reboot, but failed
the 4.5 GiB copy with Xid 31 / FAULT_PTE. Repeating with unlimited memlock
also failed. dgx3 reproduced with the Ubuntu stock UVM after removing the saver
and rebooting (stock source version `625DCD62A2DB1AC8DCCA0FF`). No model was
started on these builds.

This matches [NVIDIA issue 1269](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1269),
reported by Max Spevack: the newer RM requires 2 MiB-aligned DMA submaps, but
its 64 KiB branch selects 4 GiB minus 64 KiB. The separate experimental patch
extends its existing ARM64 alignment rule to all ARM64 page sizes. The 4 KiB
value remains unchanged; 64 KiB uses 65,504 pages, exactly 4 GiB minus 2 MiB.
The local source hash and exact patch are pinned. This changes RM, so it is
outside memory-saver's UVM-only scope and is installed as a separate temporary
signed module override. The restoration script removes it before R580 returns.
Testing this corrected driver is a new arm, not evidence that stock R610 works.

The corrected RM passes 3,584 allocation/readbacks, the 4.5 GiB transfer,
BF16 matmul/graph replay and 60 probes around 15 separate DMA boundaries in
a 64 GiB allocation on all nodes. Its ELF build ID is pinned; NVIDIA's module
source-version string does not distinguish this header-only correction.

The first corrected-driver model start failed before weight loading: CUDA
reported only 50–51 GiB free. R610's `NVreg_EnableSystemMemoryPools=529`
default retained the 64 GiB test allocation after process exit. Its source
`nvidia/nv-reg.h` documents that caching behavior. The experimental host config
sets this option to zero, restores returned pages to ordinary host accounting,
and checks the loaded setting before serving. This setting is removed during
restoration. The original failed-start logs are retained.

The first derivative image (`exp1`, `sha256:f81aca445450fafe595751f9c60b8f39eb42998583855ebeaf7a70e71b4b9d38`)
passed the bounded CUDA tests but was discarded before serving: the package audit
showed r5o already has nvJitLink 13.4.52, whereas that image installed 13.3.33.
The corrected `exp2` keeps 13.4.52. Shared runtime packages move forward only;
this is why the audit records actual libraries instead of assuming the image's
CUDA label describes every installed component.

## Protocol and reproducibility

Each arm uses this screen from the head node, with its configuration and a unique
output directory:

```sh
bin/spark3 --cluster-config "$arm" bench \
  --suites quality,decode,prefill \
  --decode-cases prose,code,prose-nothink,code-nothink \
  --concurrency 1,8 --min-samples 3 --max-samples 3 \
  --prefill-text source --prefill-sizes 4096,16384,65536 \
  --prefill-repeats 2 --compare none --output "$output"
```

The sequence is R580/original image, corrected R610/original image, corrected
R610/CUDA exp2, then a second R580/original-image boot. Configuration files are
`baseline-pinned.json`, `driver610.json` and `cuda133.json`. The R610 arm includes
the UVM port, RM alignment correction and disabled system pools: a successful
stock-R610 performance comparison was impossible, so do not call it driver-only
without this qualification. The CUDA comparison additionally changes runtime,
cuBLAS, NVRTC and Triton's assembler together; it does not isolate each component.

All screens use benchmark-ordering seed zero, temperature-zero requests with
seed 42 and a 256-token decode limit, a discarded decode warmup round,
and prompt hash
`cc7f29400eff69775f0d9fb016be00924f46a81be7b42dc5c15a4eb9d82da779`.
The two source-text samples at each nominal prefill size contain 3,726/3,615,
14,905/14,984 and 59,653/56,289 tokens, respectively. The native reports retain
each actual length and TTFT. No measurement sample was excluded as an outlier.
Client commits differ as experiment files and host checks were published; the
benchmark implementation and promoted serving sources are unchanged across arms.

The pinned cost file is `dspark-costs-e9eb8edaf99252b3.json`, SHA-256
`5dc8961a53b95db8020b32c710e8bfbcaa0150ec6324289a198481aad4081b37`,
copied into `runs/`. Only rank zero reads it and broadcasts the curves, as the
shipped patch specifies; workers need no duplicate file. Pinning this table does
not make generated text, draft acceptance or scheduling deterministic. Use the
reported accepted/verified drafts and step times alongside TPS.
The reported single-stream step time is an estimate, `(1 + accepted drafts) /
decode tokens per second`, rather than a device trace; verification work still
varies. The five quality checks inspect the generated LRU class structure. They
are a compatibility smoke screen, not proof of numerical or general model-quality
equivalence. This screen does not establish determinism.

Exp2's image ID is
`sha256:8e9612387b0e01f5223ac6690971e17bf4340b1c91740f782e3c973b346b8013`
on every node. The loaded-library audits are retained.
Both library audits were run on R610; "original" identifies the original
container libraries, not the R580 host driver. Fresh Triton/Inductor/vLLM
cache directories selected the new assembler; startup compiled 132 warmup keys
in 40 seconds. B12X/CuTe and native extension builds were unchanged. Startup
times are not a fair performance comparison because these cache states differ.
After image export/load and before the repeat R580 control, stopped hosts were
synced and their file page caches dropped; per-node precondition logs remain in
`.work/driver-cuda-refresh/`. Serving-time memory guards stayed at 5/3 GiB.

## Restoration procedure exercised

Inside the exclusive window, stop the three-rank service with
`bin/spark3 --cluster-config "$arm" cluster stop --apply`. On all nodes run
`bash experiments/2026-10-02-driver-cuda-refresh/host-transition.sh old`.
It removes the temporary RM override and system-pool setting, removes saver
0.3.0, verifies cached package hashes, restores the exact R580 packages, and
builds/signs/installs saver 0.2.0. After all three succeed, reboot all nodes and
verify loaded identities before any serving start. The two newly introduced,
unused packages (`nvidia-prime` and `nvidia-firmware-610-610.57.04`) were also
removed after an apt simulation showed that only those packages would be removed.
Their cached DEBs remain available; no general autoremove was used.

The restored inventory and bounded CUDA checks are retained for each node.
Expected loaded UVM source identity is `34683B82C2D3339BBD2EEC9`, packing `Y`,
Secure Boot enabled, and the same 64 KiB boot default. The final serving restore
uses `config/cluster.json`, removing the experiment's pinned-cost environment.
Close the hold only after `doctor --live` reports the production cluster matches.
