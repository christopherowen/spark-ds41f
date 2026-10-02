# 64 KiB boot and serving trial — 2026-10-02

Status: compatibility checks and the fresh 4 KiB rollback control passed.
The subsequent [allocation profile and UVM recovery trial](../uvm-pool/README.md)
identified the overhead and recovered 1.78–1.86 GiB/node over 4 KiB, without a
substantial measured speed gain. This requires an experimental driver patch;
the normal default remains stock 4 KiB pending driver promotion. Production
image remains r5o. The observations below describe the original stock-driver
boot screen, before that investigation.

## Authority and identity

The owner requested: “boot test, probably set 64kb as the default kernel.”
This authorized the coordinated stop, trial reboots and rollback, with a default
change conditional on the result. The cluster hold was `codex-64k-boot-test`.
No other GPU experiments or cluster changes ran during the window.

The 4 KiB baseline ran before reboot. The 64 KiB serving screen and the fresh
4 KiB control used deployment commit `0ad64e14027078f08dd467842f483da0e032d4b5`,
unchanged `config/cluster.json` and the same image on all three ranks:
`vllm-ds41f-kkref:04c30fa98e79-r5o`,
`sha256:288fc5bd909eb7e5fa51bd5abd94286140a0f07c40b92d763f8be4970309e5ab`.
The kernel pair is `7.0.0-1019-nvidia` / `7.0.0-1019-nvidia-64k`, with
NVIDIA driver `580.178.04`. Secure Boot remained enabled.

The production KV budget, model weights, precision, compile caches, TP=3 and
speculative settings were unchanged. Production profiles its draft-verification
costs at each startup; this trial did not pin them. It therefore tests the
host profile as production starts it, not a fixed-work kernel speed change.

## Boot procedure and compatibility

1. Captured the original 4 KiB host state and a short quality/decode/prefill
   baseline. Stopped serving with `bin/spark3 cluster stop --apply`.
2. Installed the [page-aware swap and memory policy](../../../host/kernel-memory/README.md).
   A separate 16 GiB `/swap-64k.img` has a 64 KiB swap header. The original
   `/swap.img` is retained. The boot service selects the matching file.
   On 64 KiB only, THP is `never` and `vm.min_free_kbytes=45166`; the existing
   `vm.watermark_boost_factor=0` remains on both kernels. These are part of
   the tested profile, so results cannot be attributed to base page size alone.
3. Booted dgx3 with a GRUB one-shot, retaining the explicit 4 KiB normal default.
   Its NVIDIA, RoCE and signed fan-control modules loaded. `nv-cpu-governor`
   initially failed because the matching `linux-tools` package was missing.
   Installed `linux-tools-7.0.0-1019-nvidia-64k=7.0.0-1019.19~24.04.2` on all
   nodes and restarted the governor service on dgx3. Doctor now checks the
   exact package, executable and active performance governors. No failed units
   remained. The failed boot-service attempt is retained in the dgx3 receipt.
4. Ran [cuda_smoke.py](cuda_smoke.py) in a bounded 20 GiB container. It checks a
   single 4.5 GiB pinned-memory/GPU transfer across the 4 GiB boundary, the
   reverse transfer, BF16 matrix multiplication and CUDA graph replay.
   Booted dgx1 and dgx2 the same way and repeated the smoke check on each.
   All passed; [results](cuda-smoke.json).
5. Started the production configuration with `bin/spark3 cluster start --replace
   --apply`. Startup guards (5 GiB) and steady guards (3 GiB) remained in force.
   The model loader, three-rank communication, CUDA graphs and API started;
   readiness took about 121 seconds. Ran the serving screen below.
6. Stopped all ranks and rebooted every node normally into the retained 4 KiB
   default. All three restored the original swap and had no failed services.
   Started the same image and configuration (API readiness about 125 seconds)
   and ran the identical serving screen as a fresh-boot control.

No guard was weakened, no image was rebuilt, and no default was changed to the
candidate. Both swap files and kernels remain installed. Rollback is verified,
not just described. No upstream source or patch series changed.

## Serving screen

The same command was used for `serving-64k` and `serving-fresh-4k`:

```sh
TMPDIR=/tmp bin/spark3 bench \
  --suites quality,decode,prefill,prefix,admission \
  --min-samples 2 --max-samples 2 \
  --concurrency 1,2,4,8 --decode-cases prose,code \
  --prefill-sizes 1024,32768,65536 --prefill-repeats 1 \
  --prefill-text source --prefix-tokens 32768 --admission-tokens 65536 \
  --compare none --output results/private/kernel64k-boot/<arm>
```

Decode has one discarded warm-up at each point. Source-prefill targets produce
1,146, 29,447 and 59,982 actual prompt tokens; admission requests contain
63,595 tokens each. The harness records actual counts in the native reports.
Five LRU functional checks are not a numerical-equivalence or determinism test.
Production's existing nondeterministic arithmetic is unchanged.

Decode has only two measured samples per point, and prefill only one. Confidence
intervals are wide; do not infer a small speedup from point estimates. The
long-lived initial 4 KiB screen is retained as historical context, not silently
substituted for the fresh control. The screens cover only the listed prompt
lengths; they do not qualify 262K context or vision behavior.

## Memory interpretation

Linux `MemTotal` rises from about 121.69 to 123.79 GiB per node, a gain of
about 2.10 GiB. That is not the same as additional space available to serving.
At API readiness the fresh 4 KiB control had 7.09/7.95/8.04 GiB available on
nodes 1/2/3. The 64 KiB loaded-model state and workload minima were lower.

The captures also show less secondary page-table/slab memory but more anonymous
memory on 64 KiB. These categories do not fully explain the difference; the
remaining driver allocation/padding and host accounting costs have not been
isolated. The 64 KiB boot also ran the standalone CUDA smoke first, whereas the
fresh 4 KiB control went directly to model startup. A clean 64 KiB repeat without
that smoke would be needed before claiming a precise causal memory penalty.
This is an unexplained observation, not a final promotion decision or a claim
that all 64 KiB workloads use more memory.

Further investigation should first reproduce loaded-model headroom with matched
fresh boots and startup work, then attribute host/driver allocations. There is
no reason to change model memory budgets or production defaults based solely
on the larger `MemTotal` number.

## Evidence and checks

- `baseline-4k/bench.json`: original, long-lived 4 KiB screen.
- `serving-64k/bench.json`: full candidate serving screen.
- `serving-fresh-4k/bench.json`: full fresh rollback control.
- `dgx*-before-4k.txt`: original pretrial host captures.
- `dgx*-after-64k.txt`, `dgx*-trial-inventory.json`: successful candidate boots.
- `dgx*-serving-64k.txt`: candidate loaded-model memory and kernel journal.
- `dgx*-fresh-4k-*.txt`: rollback, loaded-model and final host captures.
- `results.json`: compact comparison and decision; native reports are authoritative.

Host-independent checks: 16 kernel-readiness tests and 49 existing doctor tests
pass. No prepared build inputs exist in this local worktree, and the serving
image/upstream series did not change; no rebuild or upstream patch replay was
needed for this host test.
