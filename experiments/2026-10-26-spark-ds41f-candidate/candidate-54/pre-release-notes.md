# spark-ds41f-candidate-54 (pre-release)

A frozen candidate: the merged base of the second round of PR validation, verified on three testbeds and supplied
for testing ahead of the next release. It is not a release: `spark-ds41f-1107` stays the latest release and the
default, and `config/cluster.json` does not change. The next round of candidates is built on this one.

## What changed since spark-ds41f-1107

**vLLM: 672 patches** on the 1107 base (`0001..0761` of 1107, then `0762..0791` and `0799`); `patch_head`
`15833ed1`, tree `ad2dc11f`, `patchset_sha256` `d990ee5d`. Every other source is 1107's.

**Fix: temperature-0 determinism at `tp4-ring` (`0799`).** The MoE combine kernel read the shared expert's output
before its PDL wait. In a captured CUDA graph, the shared expert (on the aux stream) could still be storing those rows,
so one rank's MoE output could differ, about one boot in three at `tp4-ring`. The same race made the speculative
drafter's accept decisions differ between boots. `0799` reads the shared rows after the wait. 1107 has the same load
order (its `0756`), though its gate showed no failing boot in four.

**Recipe: TP3's L2 prefetch windows capped at 15/12/12 MB** (`VLLM_DS41_L2_PREFETCH_WO_MB=15`, `_FFN_MB=12`,
`_NEXT_MB=12`) on the `tp3-triangle`, `tp3-triangle-4k` and `tp3-switched` shapes. #961's paced prefetch made TP3's
1-request decode steps 5.5-7.2% slower at its default window sizes; the caps remove that. TP4's recipe is unchanged.

**29 PRs, 30 patches:**

| PR | Patch | Title | From | Verdict |
| --- | --- | --- | --- | --- |
| #966 | `0762` | perf(dsv41): default unswizzled routed MoE TMA weight loads to L2 evict_first | candidate-47 | pass |
| #997 | `0763` | perf(dsv41): certified two-pass Cauchy-Schwarz lm_head greedy vocabulary screening | candidate-47 | pass |
| #965 | `0764` | perf(worker): eliminate decode-step redundant H2D transfers, UVA copies, and eager ATen ops | candidate-47 | passed after rework |
| #1012 | `0765` | perf(dsv41): overlap decode KV/indexer-key writes with query/index-query on aux_stream | candidate-47 | passed after rework |
| #1016 | `0766` | perf(dsv41): align Q-B and indexer Q-B decode tiles with 128x128 tile-contiguous weights | candidate-47 | passed after rework |
| #964 | `0767` | perf(dsv41): sparse MLA stage-1 KV gather and staging barrier opt | candidate-47 | pass |
| #955 | `0768` | Require the DSML token for DeepSeek V4.1 tool call markup (#949) | candidate-48 | pass |
| #980 | `0769` | perf(spec_decode): elide TP confidence broadcasts and lockstep TP sampling seeds | candidate-48 | passed after rework |
| #1013 | `0770` | fix(vllm): enforce completions structured-output grammar and add size-classed prefill lanes (in part: its grammar fix (0770) alone, as candidate-50 carried it) | candidate-48 | pass |
| #972 | `0771` | feat(serve): enable periodic SSE keep-alive heartbeats and shield streaming abort cleanup | candidate-48 | pass |
| #1011 | `0772` | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 (in part: candidate-48's 0768 and 0769 alone (here 0772 and 0773), as candidate-50 carried them) | candidate-48 | pass |
| #1011 | `0773` | experiments(vllm): audit Nov 2026 mainline PRs and backport 0762..0764 (in part: candidate-48's 0768 and 0769 alone (here 0772 and 0773), as candidate-50 carried them) | candidate-48 | pass |
| #973 | `0774` | perf(dsv41): fast-path multi-turn chat rendering and segment tokenization | candidate-48 | pass |
| #999 | `0775` | perf(dsv41): widen long-context indexer top-k segments and reuse source blocks | candidate-49 | passed after rework |
| #993 | `0776` | perf(dsv41): vectorize indexer_score_v41 KPool loads and co-schedule dense CTA columns | candidate-49 | pass |
| #991 | `0777` | perf(dsv41): use hardware cvt.rn.satfinite.e2m1x2.f32 in indexer and GEMM MXFP4 quantizers | candidate-49 | meets its mark (the coordinator's verdict) |
| #1009 | `0778` | perf(dsv41): fuse SP prefill local attention front and keep index_weights local | candidate-50 | passed after rework |
| #979 | `0779` | perf(dsv41): route SP prefill gather_many and row exchange through sparknet | candidate-49 | passed after rework |
| #996 | `0780` | perf(sp-prefill): cache SP local routing masks and skip SP tail pad rows | candidate-49 | meets its mark (the coordinator's verdict) |
| #961 | `0781` | perf(dsv41): 64 KiB chunked, %globaltimer-paced side-stream L2 prefetch | candidate-51 | pass |
| #963 | `0782` | perf(dsv41): unconditional pack12 plane loads, byte-perm vocab SWAR, and per-column exception lookup | candidate-51 | passed after rework |
| #987 | `0783` | perf(dsv41): extend WO-B WIDE_DECODE to 128 rows and guard high-concurrency decode cliffs (in part: its WO-B part (0783), as candidate-51 carried it) | candidate-51 | pass |
| #1001 | `0784` | perf(dsv41): match 128-row vocab head tile pipeline stages on GB10 | candidate-51 | passed after rework |
| #1030 | `0785` | fix(dsv41): align moe_swiglu_cast with DeepSeek Expert.forward BF16 rounding | candidate-52 | pass |
| #968 | `0786` | perf(spec_decode): clamp scheduled decode verify rows to remaining max_tokens | candidate-52 | pass |
| #969 | `0787` | perf(dsv41): direct-insert packed Engram cache rows and align split cache arrays | candidate-52 | pass |
| #960 | `0788` | Target FULL CUDA graph metadata/CED replay and MoE post-gate fork | candidate-53 | passed after rework |
| #971 | `0789` | perf(spec-decode): cumulative survival in copy_apply, single-req 16-row tile cap, and wide-step dead-row mask (in part: round 1: the copy_apply survival gate and the single-request tile cap, without the dead-row floor (0789)) | candidate-53 | passed after rework |
| #977 | `0790` | perf(spec_decode): bound CopyDrafter long-context history scan in copy_match | candidate-53 | pass |
| #998 | `0791` | perf(worker): compact packed vocab tables and expand display carve-out weights | candidate-53 | passed after rework |

Left out on the way: #984 missed its mark and is in rework; #986, #1005 and #1010 were dropped after rework (no defensible correction). The patches left out at selection, each with its reason, are in the manifest (`prs.left_out`). Every carried PR is christopherowen's; every patch's author is Christopher Owen except `0768` (#955), koldfrontier's, an owner-approved exception.

## Verification

The image is `spark-ds41f:candidate-54`, ID `d502217e` (built on dgx4 from the lock), the same on every node of the
three testbeds. Its release tests pass: 10,171 passed, 311 skipped.

| Testbed | Determinism gate | Screen against 1107 | Live checks | c1 accept decisions between boots |
| --- | --- | --- | --- | --- |
| `tp4-ring` (dgx1..dgx4) | 12 of 12 boots with `0799` (4 of 12 failed before it), then the final image; TP4 hashes `18d61961`/`d6568649` | grid 4.94% faster; c16 output 1.74% and paired decode 1.35% faster; prefill the same at 32K, 7.35% faster at 256K | run at `tp4-switched` | identical on 4 of 4 boots |
| `tp4-switched` (sp1..sp4) | 4 of 4 boots, then the final image; the same hashes | grid 4.24% faster; c16 output 1.50% and paired decode 1.13% faster; prefill the same at 32K, 7.56% faster at 256K | #955, #1013, #972 pass | identical on 4 of 4 boots |
| `tp3-switched` (sp5..sp7) | 4 of 4 boots, then 2 of 2 with the prefetch caps; TP3 hashes `5c22ce41`/`bce6092e` | grid the same; c8 the same (one boot); prefill not measured | | |

Verified means the gate, the screen and the carried fixes' live checks on each testbed, plus the release tests. It is
not release qualification: no release benchmark set, quality evals, AgentX or cold boots.

## Against spark-ds41f-1107

Two candidate boots against 1107's recorded screen at each shape; a change is reported only past the run's own
repeat. The targets (c1 decode 100 tok/s, prefill 10k tok/s) are measured at `tp4-ring`.

| Testbed | Metric | 1107 | candidate-54 | Change | To target |
| --- | --- | ---: | ---: | --- | --- |
| `tp4-ring` | c1 decode | 82.6 tok/s | not measured on this image | the same on the build before `0799` | about -18% |
| | c16 output | 265.2 tok/s | 269.9 | +1.74%, faster | |
| | engine grid (geometric mean) | | | -4.94%, faster | |
| | prefill 32K | 8,500 tok/s | 8,700 | +2.10%, the same | -13.0% |
| | prefill 256K | 8,228 tok/s | 8,854 | +7.35%, faster | -11.5% |
| `tp4-switched` | c16 output | 260.5 tok/s | 264.6 | +1.50%, faster | |
| | engine grid | | | -4.24%, faster | |
| | prefill 32K | 8,368 tok/s | 8,663 | +3.43%, the same | |
| | prefill 256K | 8,184 tok/s | 8,752 | +7.56%, faster | |
| `tp3-switched` | c8 output | 156.4 tok/s | 157.8 (one boot) | +0.87%, the same | |
| | engine grid | | | -0.10%, the same | |
| | prefill | | not measured | | |

At `tp3-switched`, the companion suites against the round-1 base (candidate-50) are the same: single-stream c1, sampled
c16/c8/c1, and the prefill-under-decode stall steps (stall-3's prefill 1.8% faster).

## Known issues

- Verified, not release-qualified: no release set, quality evals, AgentX or cold boots.
- TP3 prefill is not measured, and TP3's c8 is one boot's: sp5's MemAvailable fell under the 4 GiB bench floor in the prefill step (first boot) and the c8 step (second boot), as it did for the earlier build.
- With the TP3 prefetch caps, TP3's 8x24 grid shapes are 1.0-2.3% slower than 1107 and 4x24 at 4K 0.8-1.4% slower; 1-request shapes are up to 1.5% faster and 8x48 1.3-1.6% faster, and the grid's mean is the same.
- c1 paired decode against 1107 is not measured on the rebuilt image; the earlier build's was the same as 1107's (about 82 tok/s, cross-session). It stays about 18% under the 100 tok/s target.
- The final gates ran one boot at each TP4 testbed and two at TP3. The boot-to-boot evidence is 0799's gate on 12, 4 and 4 boots, on 0799's first build, whose serving code is the rebuilt image's.
- Temperature-0 outputs differ from 1107's by construction (#1030: routed MoE activations rounded to BF16 before MXFP8 quantization). TP4's hashes are 18d61961/d6568649 (1107: 8e46794a/11ec073c), TP3's 5c22ce41/bce6092e.
- Only tp4-ring, tp4-switched and tp3-switched are verified; the tp3-triangle, tp3-triangle-4k and tp5..tp8-switched configurations are not measured.

## How to run it

Check out the tag, then, with your site's node map ([docs/setup.md](https://github.com/christopherowen/spark-ds41f/blob/spark-ds41f-candidate-54/docs/setup.md)):

```bash
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-54/<shape>.json build prepare
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-54/<shape>.json build image --apply
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-54/<shape>.json build distribute --apply
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-54/<shape>.json cluster start --apply
```

`<shape>` is one of `tp4`, `tp4-switched`, `tp3`, `tp3-triangle-4k`, `tp3-switched`, `tp5-switched` .. `tp8-switched`.
- `build image`, `build distribute` and `cluster start` are plans until `--apply`; read their `live:` and `shape:`
  lines first.
- `build prepare` must name the profile: the profile selects candidate-54's lock (`upstreams_config`), and without it
  `build prepare` prepares the release's.
- The image is built from the lock; no binary is published.

The freeze manifest is [manifests/frozen/candidate-54.json](https://github.com/christopherowen/spark-ds41f/blob/spark-ds41f-candidate-54/manifests/frozen/candidate-54.json), and the
candidate's record is [its README](https://github.com/christopherowen/spark-ds41f/blob/spark-ds41f-candidate-54/experiments/2026-10-26-spark-ds41f-candidate/candidate-54/README.md).
