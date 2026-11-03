# External site receipt: spark-ds41f-1030 at tp4-ring, on 4 KiB and on the 64 KiB host (DeepSeek-V4.1-Flash-1017)

The site of [2026-10-17-site-tp4-4k-1017](../2026-10-17-site-tp4-4k-1017/README.md) moved its ring from r6u to
release **spark-ds41f-1030** in two stages, on one day, with the same client battery:

1. the engine first, on the stock 4 KiB host;
2. then the host, to the recipe's 7.0 64 KiB kernel with the memory saver, at the recipe's 16,384 batched tokens and
   at 8,192.

It is a receipt from another site, not a promotion request. Both shapes stay unqualified here (`--unqualified-shape`).

## Site

| | |
|---|---|
| Nodes | 4 x ASUS Ascent GX10 (GB10), DGX OS 24.04, Secure Boot on, `host.fan_control: false` (not NVIDIA's P4242 board) |
| Ring | switchless, like-port cabling, both PCIe functions per cable, one /24 per stripe (as in the earlier receipt) |
| Weights | DeepSeek-V4.1-Flash-1017 `1bae0e10`, on the head's disk; workers read the snapshot over read-only NFS, except shards 47/48 (the Engram tables, 94.6 GiB each), local on every node |
| Image | `spark-ds41f:1030`, `sha256:0cb7fc77...`, built once with `bin/spark build image --apply` (34 min), tree labels equal 1030's `expected_labels`, copied to the other nodes |
| 4 KiB host (stage 1) | kernel `6.17.0-1032-nvidia`, driver 580.173.02, no memory saver; arm [`candidate-arm-4k.json`](candidate-arm-4k.json) |
| 64 KiB host (stage 2) | kernel `7.0.0-1019-nvidia-64k`, driver 580.178.04, `kho=off`, memory saver 0.4.1 (`ec1b17a8`, DKMS) signed with a site MOK key; arm [`candidate-arm-64k.json`](candidate-arm-64k.json) |
| Serving | one name (`glp` unset), 1,048,576 context, 16 sequences, KV 10.5 GiB, DSpark 5, 64 images per prompt; batch budget 8,192 (threshold 8,096) unless stated |

The 64 KiB host was prepared with [`host/prepare-64k.sh`](host/prepare-64k.sh), one node first and alone (a one-shot
`grub-reboot`, [`host/reboot-64k-once.sh`](host/reboot-64k-once.sh), checked with
[`host/check-boot.sh`](host/check-boot.sh)), then the other three, the head last. The same change also applied the
low-RAM headless items: `multi-user.target`, DGX Dashboard off, `vm.watermark_boost_factor=0`, NVMe interrupt
coalescing masked, APST 2000/60000 ms, `splash` off.

## Findings worth knowing upstream

1. **1030 runs on the stock 4 KiB host too, and beats r6u there.** On the same day, at the same budget: decode steps
   8-10% shorter, aggregate +10.2/+9.5/+5.6/+4.7% at C1/C4/C8/C16, cold prefill +10-18%. qeval text is identical to
   the token (71/75, 9,440 completion tokens), with 4-5 GiB more MemAvailable per node at rest (head 23.8, workers
   24.6-25.5 GiB).
2. **At the same batch budget, the 64 KiB host is speed-neutral and gives about 2.4 GiB per node.**
   - Throughput -0.6/-0.6/-0.6/+1.4%; prefill and needles within ±1%; outputs identical.
   - MemAvailable at rest: head 26.4, workers 26.8-27.7 GiB, against 23.8 and 24.6-25.5 on 4 KiB. The lowest through
     the whole battery, the ~1M needle included, was 26.1-27.6 GiB per rank.
   - The memory figure covers the whole host change, not the memory saver alone. `watermark_boost_factor=0` alone
     changes MemAvailable's accounting, and the driver moved too.
3. **The 16,384 budget's cost under mixed traffic, measured.** On the same 64 KiB host, against 8,192:
   - aggregate throughput +1.5/+1.1/+1.6/+0.9% at C1/C4/C8/C16;
   - prefill +4-6% from 11K tokens (2-4% slower at 3K);
   - the ~1M needle 5,590 against 5,410 tok/s.

   A short streamed request ([`runs/stall_probe.py`](runs/stall_probe.py)), started 10 s into a ~545K-token salted
   cold prefill:

   | | 16,384 | 8,192 |
   |---|---:|---:|
   | alone: TTFT / chunk gap p50 | 0.13 s / 32 ms | 0.13 s / 33 ms |
   | beside the prefill: TTFT | 3.27 s | 1.53 s |
   | beside the prefill: chunk gap p50 / max | 2,342 / 2,877 ms | 1,272 / 1,462 ms |
   | reply time (1.7 s alone) | 73.8 s | 64.3 s |
   | the long prefill itself | 6,540 tok/s | 6,278 tok/s |

   `--prefill-yield short` brings the first token early. After that, each decode step of the short request rides one
   prefill chunk until the long prompt is done. This agrees with tp4-ring's note that steps beside a long prompt are
   about twice as long at 16,384. With agent and chat clients sharing the engine, this site serves 8,192.
4. **Host install notes for the 64 KiB kernel**:
   - Driver 580.173.02 is no longer published; `apt-cache madison` shows 580.178.04 and 580.196.07. Installing
     `nvidia-driver-580-open` 580.178.04 moves the HWE meta-packages to `7.0.0-1024-nvidia` (4 KiB), the newest
     kernel. Without a `GRUB_DEFAULT` pin (written before apt, as `experiments/2026-10-02-kernel-64k/prepare.sh`
     does), the next boot is that kernel.
   - Nodes that still have the older `6.17.0-1029-nvidia` kernel must remove its 580.173 module package. The
     transaction needs one removal there (`--no-remove` refuses it).
   - `linux-modules-nvidia-580-open-6.17.0-1032-nvidia` 6.17.0-1032.32+1 keeps 6.17 bootable as a fallback with
     580.178.04. This site checked it with `modinfo` only and did not boot it.
   - `nvidia-cdi-refresh.service` fails during the driver upgrade while the old module is loaded. It is clean after
     the reboot.
   - DKMS signs the memory saver with an enrolled site key through
     `/etc/dkms/framework.conf.d/memory-saver-mok.conf` (`mok_signing_key`, `mok_certificate`). One key pair serves
     all four nodes, enrolled once at each console. After boot the loaded `nvidia_uvm` srcversion is
     `37A4D127AE60F1F3AE60A02` (= `config/kernel-trial.json`), `uvm_pack_sysmem_leaf_tables` is Y, and all eight
     rails read 109.23-109.30 Gb/s (`ib_write_bw`, 1 MiB, [`host/rails.sh`](host/rails.sh)).
5. **Reboots on GX10**:
   - Four warm reboots for MOK enrollment, with a monitor and keyboard attached, left all four GPUs clock-latched
     (507-734 MHz, about 9 W under load; decode steps 58-77 ms). Only a cold drain cleared it (power off, hold the
     power button). The four warm reboots into 64 KiB the next day, with nothing attached, did not latch (2,288-2,340
     MHz, 94 W). [`docs/recovery.md`](../../docs/recovery.md) describes the latch; attaching peripherals looks like a
     trigger here.
   - One node logged Xid 119 (GSP timeout) about a minute after its enrollment boot. `nvidia-smi` still listed the
     GPU, while `cuInit` returned 100; a reboot cleared it.
6. **Smaller things**:
   - The build's base tag (`vllm/vllm-openai:nightly-af1c0149...`) is gone from Docker Hub ("manifest unknown"). The
     Dockerfile's digest still pulls, and buildx pulls by digest.
   - With `host.fan_control: false`, `doctor --live` on the 64 KiB kernel still warns about the `dgx_ec_fan_control`
     kernel inventory and its signature. It also warns about the missing 4 KiB `7.0.0-1019-nvidia` fallback files.
     All are warnings only.
   - The weight-fragmentation check resolves the snapshot's relative links on the host (`../../blobs`, i.e.
     `$HOME/blobs`), not the profile's `blobs` path. So it cannot check a snapshot+blobs layout.
   - `doctor --live` reports `nvidia-drm modeset is N` as an error on tp4-ring. The site left the modeset package in
     place, and starts are unaffected.

## Results

The same head-side client on every run, thinking off unless a task enables it: the battery of
[`../2026-10-17-site-tp4-4k-1017/runs/battery.sh`](../2026-10-17-site-tp4-4k-1017/runs/battery.sh), plus a salted
sweep and a ~1M needle ([`runs/tests.sh`](runs/tests.sh)). It is the site's own battery, not `bin/spark bench`.

| Measure | r6u, 4 KiB | 1030, 4 KiB, 8,192 | 1030, 64 KiB, 16,384 | 1030, 64 KiB, 8,192 |
|---|---:|---:|---:|---:|
| idle step, counting / code (ms) | 35.9-36.2 / 37.9-38.1 | 32.6-33.1 / 34.9 | 32.4-32.5 / 34.5-35.0 | 32.9-33.2 / 34.9 |
| v41bench aggregate tok/s, C1 / C4 / C8 / C16 | 88.8 / 219.3 / 340.7 / 478.7 | 97.9 / 240.2 / 359.9 / 500.9 | 98.8 / 241.6 / 363.7 / 512.5 | 97.3 / 238.9 / 357.8 / 507.7 |
| v41bench cold prefill tok/s, 2.9K / 11.6K / 46.8K / 93.3K | 4,446 / 5,753 / 6,140 / 6,051 | 4,899 / 6,457 / 7,223 / 7,167 | 4,821 / 6,776 / 7,604 / 7,540 | 4,902 / 6,374 / 7,250 / 7,185 |
| salted cold prefill tok/s, ~3.4K / 13K / 54K / 109K (mean of 2) | - | 4,657 / 6,441 / 7,200 / 7,190 | 4,577 / 6,781 / 7,482 / 7,544 | 4,743 / 6,438 / 7,224 / 7,218 |
| salted needle, ~545K | PASS, 5,240 | PASS, 6,266 | PASS, 6,553 | PASS, 6,298 |
| salted needle, ~990K | - | PASS, 5,388 | PASS, 5,590 | PASS, 5,410 |
| qeval, 75 tasks, greedy | 71/75 | 71/75 | 71/75 | 71/75 |
| vision + tools | 7/7 | 7/7 | 7/7 | 7/7 |
| KV startup line | - | 12,040,153 (11.48x) | 11,554,122 (11.02x) | 12,040,153 (11.48x) |
| MemAvailable at rest, head / workers (GiB) | 18.7 / 20.5-21.5 | 23.8 / 24.6-25.5 | 24.1 / 23.9-25.1 | 26.4 / 26.8-27.7 |
| warm ready (s) | - | 194.1 | 172.2 | 180.3 |

- qeval misses the same four tasks in every run (`code_interval_intersect`, `json_count`, `json_escape`, `math_m9`).
  The four 1030 runs produce identical text.
- Per-category C16 numbers move a few percent both ways between runs at the same budget, so single C16 categories are
  not used here.
- The r6u run is the same-day baseline. It was stopped before its salted sweep and ~1M needle, for a console
  window.
- The memory sampler ([`runs/memwatch.sh`](runs/memwatch.sh), `memwatch.tsv`) ran on both 64 KiB runs only.

Limits:
- one boot per configuration, no confidence intervals, no soak;
- the stall probe is one request per budget;
- no agent-workload replay;
- fan control is not installed, so `bin/spark bench` was not used.

## Files

- `base.json`, `candidate.json`, `candidate-arm-4k.json`, `candidate-arm-64k.json`, `results.json`, `decision.md`
- `runs/r6u-same-day`, `runs/1030-4k-8192`, `runs/1030-64k-16384`, `runs/1030-64k-8192`, each with:
  - idletest, v41bench, the ~545K needle, qeval (text + JSON), vision/tools;
  - for the 1030 runs, the salted sweep, the ~990K needle and MemAvailable before and after it;
  - for the 64 KiB runs, `stall.jsonl` and `memwatch.tsv`.
- `runs/tests.sh`, `runs/stall_probe.py`, `runs/memwatch.sh`: the client side added since the earlier receipt. Its
  `battery.sh`, `compare.py` and `needle_salted.py` are unchanged.
- `host/`: the 64 KiB preparation as run (`prepare-64k.sh`), the one-shot reboot and boot checks, the persistent
  GRUB pin (`pin-default-64k.sh`) and the rail test.
