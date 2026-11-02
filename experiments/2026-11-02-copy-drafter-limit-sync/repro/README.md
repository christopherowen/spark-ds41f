# Repro: CopyDrafter stale slot draft limits (c37 direct ring)

This is the reproducibility package for the crash in #852 and the host-sync fix in
PR #855. It is written so the same sequence can be re-run on another ring.

## Base and image

- vLLM base: candidate-37 patch head `5b7cfa37f` (tree `af0fb152`), from
  `experiments/2026-10-26-spark-ds41f-candidate/candidate-37/tp3.json`.
- Profile: `profiles/site-cand37-ring2-copy8-sync.local.json` (ring2, TP3,
  envelope 6155 / capture 59 / long-prefill 6096).
- Image: `spark-ds41f:candidate-37-af0fb152`, local id
  `sha256:2241b8f1ae65c0230f4c75a64f5cfc2a202216ccf4906448002ccd8cc88ec476`.
  The upstream image id (`e1ee7b1953b1…`) differs despite the same vLLM tree,
  so the image layers/build environment differ.
- Hardware: 3-node direct ring TP3, head `<ring-ip>`. Kernel
  `7.0.0-1019-nvidia-64k`, `PAGESIZE=65536` (64 KiB), driver `580.178.04`, P0
  SM ~2184 MHz (not explicitly `nvidia-smi -lgc` locked), no throttle reasons.
- Suite: single-stream, sha256 `7c59b27e8503`.

## Arms

| arm | change |
|---|---|
| base/copy16 | plain c37, `min_match=16` |
| copy8 | plain c37, `min_match=8` |
| race8 | copy8 + patch 0002 (stale-limit / ring-reuse counters) |
| pad0 | copy8 + patch 0003 (`-1` → `0` draft padding control) |
| dbg8 | copy8 + bounds/debug overlay |
| sync8 / sync8full | copy8 + patch 0001 (host-sync fix) |
| sync16full | `min_match=16` + patch 0001 |

The profile mounts the fix file over
`vllm/v1/worker/gpu/spec_decode/copy_drafts.py`; the instrumentation and pad0
patches replace the same file / `.../spec_decode/tilelang_copy.py`.

```sh
# boot the arm (ring2; --apply --replace)
./bin/spark --cluster-config config/site-cand37-ring2-copy8-sync.local.json \
  cluster start --apply --replace

# copy-dense probe (3 prompts a category)
python3 experiments/2026-10-23-next-release/acceptance/copy_probe.py \
  http://127.0.0.1:8000 --prompts 3 > /tmp/$tag-probe.jsonl

# deterministic single-stream suite
python3 -m servebench suite http://127.0.0.1:8000 \
  --suite servebench/suites/single-stream.json \
  --out /tmp/$tag --label $tag > /tmp/$tag-suite.log 2>&1
```

The crash arm is the same profile with the `mounts` entry removed. The
original 240-second-cut run is `sync8`; `sync8full` uses no client timeout.

## Results

| arm | probe tokens/step | suite |
|---|---|---|
| copy16 (m=16, unfixed) | copy 16.014, edit 15.315, tool 6.083, prose 2.651 | pass, 64.683 tok/s |
| copy8 (m=8, unfixed) | copy 16.360, edit 15.792, tool 6.636, prose 2.651 | **failed** (12,650,645); engine assert |
| race8 (m=8 + counter) | — | **failed**; 1,209 stale-limit logs, 0 ring reuses |
| pad0 (m=8 + `-1`→`0`) | — | **failed**; same assert |
| dbg8 (m=8 + debug) | — | **failed** |
| sync8 (m=8 + fix, 240 s) | — | client cut at 240 s; engine **alive**, no assert |
| sync8full (m=8 + fix) | copy 16.360, edit 15.792, tool 6.905, prose 2.651 | **pass**, 63.867 tok/s, 2.2589 tok/step |
| sync16full (m=16 + fix) | copy 16.014, edit 15.315, tool 6.232, prose 2.651 | **pass**, 63.58 tok/s (−1.7% vs 64.683, CI95 2.78) |

`sync8full` suite: `{"tps_mean": 63.8667, "ci95_pct": 2.773, "tpot_p50_ms":
15.551, "failed": 0}`. `sync16full`: `{"tps_mean": 63.58, "ci95_pct": 2.783,
"tpot_p50_ms": 15.559, "failed": 0}`. The live m=8 arm ran 64 minutes / 280
requests with 0 asserts and 0 inference failures; the m=16 arm ran 28 minutes
/ 150 requests clean before the m=8 soak.

## Crash signature

- `copy8-suite.log`: suite failed, 12,650,645 failures; probe completed first.
- `race8-engine.log` line ~2591: `Device assert failed: route_partials: gate
  logits contain nan or inf`; 1,209 `COPY_DEBUG_STALE_LIMIT` lines; 0
  `COPY_DEBUG_RING_REUSE` lines.
- `pad0-engine.log` line ~1337: same assert, despite the `-1`→`0` padding.
- `soak16-crash-head.log`: the earlier m=16 live-traffic crash at 15:47:41Z;
  the full crashed container log was lost when the arm was replaced.

## Instrumentation

Patch `0002-instrument-stale-limits.patch` prints
`COPY_DEBUG_STALE_LIMIT` whenever `sync_slot_draft_limits` sees the queued D2H
event incomplete, and `COPY_DEBUG_RING_REUSE` if the alloc ring reuses a slot.
The m=8 run produced exactly the 1,209 / 0 counts above.

The `-1`→`0` control is `0003-pad0-control.patch`; the crash persisted, which
places the bad rows in the scheduler placeholders rather than the copy-drafts
tensor.

## Tests and the pinned invariant

`tests/v1/spec_decode/test_copy_drafts.py::test_stale_wide_allocation_rejects_placeholder_tail_and_doubles_cap_without_sync`
pins the non-blocking design; the host-sync patch fails it. The review asks for
a device-side clamp so the invariant stays green; this package is the
diagnosis/host-sync record, not the final merge form.

Running the pytest suite inside the serving container is not safe: on
2026-11-02 the memguard killed the engine (exit 137, `OOMKilled=false`). Use a
stopped/dedicated container. The full suite result is therefore pending a
maintenance window.

## Memory observations

After the old engine stopped during the boot: MemAvailable 114.9 GiB (head),
117.3 GiB (rank 1), 117.5 GiB (rank 2); steady memguard 3 GiB. The m=8 crash
occurs in the suite warm-up, not at load time.

## Receipts

- `receipts/copy8-suite.log`, `copy8-engine.log` — unfixed m=8 crash.
- `receipts/race8-suite.log`, `race8-engine.log` — stale counter.
- `receipts/pad0-suite.log`, `pad0-engine.log` — padding control.
- `receipts/dbg8-suite.log` — debug arm.
- `receipts/sync8-suite.log`, `sync8-engine.log` — 240 s cut, engine alive.
- `receipts/sync8full-suite.log`, `sync8full-engine.log`,
  `sync8full-probe.jsonl` — fixed m=8 pass.
- `receipts/sync16full-suite.log`, `sync16full-engine.log`,
  `sync16full-probe.jsonl` — fixed m=16 pass.
- `receipts/copy16-*`, `receipts/base2-*` — unfixed m=16/baseline.
- `receipts/soak16-crash-head.log` — m=16 live-traffic crash head.
- `receipts/dbg8-engine.log.gz` — compressed debug-arm engine log.

The m=16 15:47:41Z full log was lost; the head excerpt is retained.

Internal hostnames/IPs and the user home path are redacted (`<ring-ip>`,
`<node>`, `$HOME`); the unredacted receipts are available privately.
