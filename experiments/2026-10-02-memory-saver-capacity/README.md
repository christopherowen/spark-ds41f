# Memory-saver capacity deployment

Owner-authorized deployment of DKMS memory-saver on all three nodes, with
explicit 4 KiB and 64 KiB serving profiles. The owner requested installation,
doctor checks and a running service with more KV memory and a larger context.
Source identity and intended deltas are in `base.json`.

The r5o container image, weights and model arithmetic are unchanged. This
validates the combined host-memory and capacity change; it does not attribute
speed differences to a single setting. The 5 GiB startup and 3 GiB steady
memory guards remain enabled.

## Configuration

| Profile | Kernel pages | KV per rank | Request limit |
| --- | ---: | ---: | ---: |
| `config/cluster-4k.json` | 4 KiB | 2.2 GiB | 262,144 tokens |
| `config/cluster-64k.json` | 64 KiB + memory-saver | 3.5 GiB | 524,288 tokens |

The request limit includes prompt and output tokens. Eight sequences are still
allowed; the cache is not sized for eight simultaneous full-length requests.
The checkpoint's native maximum is 1,048,576; RoPE is unchanged.

Startup of the 64 KiB profile reported **2,845,543 KV tokens**, **5.43x full
524,288-token contexts**, and **0.98 GiB** for CUDA graph capture. KV token
capacity includes the model's mixed cache layout; it should not be inferred
by dividing the allocation by a single per-token constant.

## Host and doctor validation

Preparation was published as `f2089ed`, then synchronized to all three clean
node checkouts. Each ran `bash experiments/2026-10-02-memory-saver-capacity/install-dkms.sh`.
The pinned memory-saver source is `2dd6ef1`, DKMS version 0.2.0, driver 580.178.04.
The existing enrolled fleet key signs the module. Secure Boot remains enabled.

- Before installation, actual `doctor --live` output warned about missing
  memory-saver DKMS on all nodes and printed separate corrective instructions.
- After installation, those warnings cleared. The service was stopped, so
  doctor correctly continued reporting absent containers.
- An actual start using the 64 KiB profile while the hosts still ran 4 KiB
  was rejected on every node before any serving container was created.
- All three booted `7.0.0-1019-nvidia-64k` and loaded UVM source version
  `34683B82C2D3339BBD2EEC9`, with packing `Y` and no status issues.
- All three passed 3,584 small CUDA allocations across four threads with
  full readback, followed by a 4.5 GiB transfer, BF16 matmul and CUDA graph test.
- The 64 KiB swap file and memory service were active, THP was `never`, the
  reserve was 45,166 KiB, and every CPU governor was `performance`.
- The coordinated start completed and live doctor passed on the larger profile.

The original observations are in [runs/fleet-validation.json](runs/fleet-validation.json)
and the individual install, CUDA, boot and doctor logs. Text logs normalize
line endings and trailing whitespace; native JSON reports are preserved. This inventory was
captured after the one-shot boots, before changing the normal GRUB default.

## Serving screens

First run, `f2089ed`, all nodes on the same published commit:

```sh
bin/spark3 --cluster-config config/cluster-64k.json bench \
  --suites quality,decode,prefill,prefix,admission \
  --decode-cases prose,code --concurrency 1,2,4,8 \
  --min-samples 3 --max-samples 3 \
  --prefill-text source --prefill-sizes 1024,32768,65536,262144 \
  --prefill-repeats 2 --prefix-tokens 32768 \
  --admission-tokens 500000 --compare none \
  --output results/private/bench/memory-saver-capacity-64k
```

The [native first report](runs/bench-initial.json) is retained, including its
failed admission verdict. Quality was 5/5, every request completed, and there
were zero preemptions. However, the 256-token replies finished before all
four long prompts could finish prefilling. Peak running requests was three,
so the check did not establish four simultaneous full contexts. This was a
workload-duration limitation, not a request or memory failure. Head-node
MemAvailable stayed at or above 5.94 GiB and swap growth was zero on every node.

Commit `5b9e49b` added explicit admission output-budget and forced-length options,
with payload tests. Defaults remain 256 tokens and natural stopping. All three
nodes were synchronized to that published commit without restarting serving.
The same complete screen was repeated with:

```sh
# Append these options to the command above and use a new output directory:
--admission-output-tokens 4096 --admission-force-length \
--output results/private/bench/memory-saver-capacity-64k-sustained
```

The forced replies keep early requests alive while later prompts prefill. This
is a capacity stress workload; its end-to-end throughput includes the long
serialized prefills and should not be compared with short-prompt decode TPS.

Near-limit retrieval uses `long-context.py`: three source-text prompts, each
actually tokenized to 518,976–520,000 tokens, with a unique phrase at 10%, 50%
and 90% of the text. The requested answer is checked against that phrase.

The sustained screen passed all requested suites. All four actual prompts
(485,025–485,026 tokens each) produced exactly 4,096 tokens, with zero
preemptions. All four fully prefilled contexts coexisted for **58.85 seconds**;
peak KV use was **52.51%**. Minimum MemAvailable was **5.83 / 7.59 / 7.60 GiB**
on dgx1/dgx2/dgx3, with zero swap growth and zero thermal slowdown.

The [native passing report](../../manifests/benchmarks/2026-10-02-karmic-kraken-r5o-64k.json)
is the new benchmark reference. Its workload identity still names r5o as the
then-promoted baseline: it was measured on the explicit 64 KiB profile before
promotion. The promotion changes that metadata and selects this same profile;
it does not change the measured launch arguments or image.

This is a three-sample decode screen and a two-sample source-prefill screen,
with no matched 4 KiB rerun. It establishes a fresh reference without claiming
a precise speed improvement. The node image identities and start times are
recorded in both reports; no restart occurred between the screens.

Near-limit retrieval passed **3/3 at 519,142 tokens**, returning the exact
phrase at all three depths. Prefix caching remained enabled, as in production;
the workload counter reports 21% cached tokens across the three requests and
zero preemptions. Actual request sizes and answers are in
[runs/long-context.log](runs/long-context.log); histogram-derived token quantiles
in the workload summary are approximate, so use the API usage counts for size.

```sh
bin/spark3 --cluster-config config/cluster-64k.json workload \
  --label memory-saver-near-520k \
  --json results/private/memory-saver-capacity/long-context-workload.json \
  -- python3 experiments/2026-10-02-memory-saver-capacity/long-context.py
```

## Deployment decision

Promote the tested 64 KiB profile under the owner's request: select it in
`config/cluster.json`, pin the normal GRUB default to the already-running
64 KiB kernel on every node, and retain the 4 KiB profile and boot entry.
The exact applied default-selection commands are in
[runs/set-64k-default.sh](runs/set-64k-default.sh). This changes future boot
selection and leaves the tested serving processes running.

The new immutable baseline is
[`2026-10-02-karmic-kraken-r5o-64k`](../../manifests/baselines/2026-10-02-karmic-kraken-r5o-64k.json).
Both profiles passed local doctor; 123 unit tests passed. Their only runtime
differences are the kernel requirements, KV allocation and context limit.
Final live doctor output is retained in `runs/doctor-selected-64k.log`.


## Setup issues retained

The first local synchronization hit macOS's SSH control-socket path limit;
rerunning with `TMPDIR=/tmp` succeeded. Worker result directories initially
rejected creation; only the experiment subdirectory was created with the
correct owner before those workers' installation. Neither issue changed a
kernel or serving state before it was resolved.

Concurrent commits `320aee4` (CI file-type dispatch) and `3f25820` (watchlist)
were preserved before publishing the admission harness. No inference image
was rebuilt. `bin/spark3 build check` passed against the pinned source stacks.

Operating instructions are in [memory-profiles.md](../../docs/memory-profiles.md).
