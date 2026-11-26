# 2026-11-26 — DFly2 block speculative decoding (`AngelSlim/DS-V4.1-Flash-DFly2-B6`)

## Base and single variable

- **Base deployment commit**: `890f8654bb` (`spark-ds41f-1126`, promoted baseline `2026-11-26-spark-ds41f-1126`).
- **Base vLLM series**: `19f2c20ed4d64ae22bd009ceacadd85e2a81e354` + `candidate-74` (`0001..0869`, 742 patches, patch head `d8784b841c360f523ec20e44bf8fb0898f36d3e4`, tree `f3783b0ec61851ba481f6a3a316d0a76d0e171d8`).
- **Candidate vLLM series**: `experiments/2026-10-26-spark-ds41f-candidate/candidate-75/vllm/series` (`0001..0870`, 743 patches, patch head `58422e4d0f76627ceca1fd4072154c97d39e1baa`, tree `1847ed52251a050c289cc780f8fa7509ccaa7f77`, patchset SHA-256 `3df5afb5e785964aaf8caf9c40f112292cbdf526af098b5ed2d91bab50fc4163`).
- **Intended variable**: replace the built-in `DSpark` drafter with Tencent Hunyuan AI Infra's standalone **DFly2** checkpoint (`AngelSlim/DS-V4.1-Flash-DFly2-B6`, Apache-2.0, `arXiv:2611.34776`), keeping target weights (`DeepSeek-V4.1-Flash-1017`), GLP-27 steering, kernel backend (`tilelang`), sparknet `v0.6.0` transports, KV pool bytes, stream count, and adaptive-verification curves unchanged per shape.

## Checkpoint provenance and fleet distribution (`dgx1..dgx4`)

- **Repository**: `https://huggingface.co/AngelSlim/DS-V4.1-Flash-DFly2-B6`
- **Pinned revision**: `d08e55a6897b1974307ea9771038838de5986e66`
- **Host cache path on `dgx1`, `dgx2`, `dgx3`, `dgx4`**:
  `/home/swank/.cache/huggingface/hub/models--AngelSlim--DS-V4.1-Flash-DFly2-B6/snapshots/d08e55a6897b1974307ea9771038838de5986e66`

| File | Size (bytes) | SHA-256 (`dgx1..dgx4` verified) |
| --- | ---: | --- |
| `config.json` | `4,144` | `0d53324203ab18c69927da96aa27566195602b9ab0aac4867b2819d6a46a051e` |
| `model.safetensors.index.json` | `197,299` | `7192e8294c56f3a9064d7b3dfa4494dc7486009f3229e98e9c6d0d2fbdaf6786` |
| `model-00001-of-00004.safetensors` | `2,735,851,872` | `3a8d253390daebb7f4279ad4edd0eb0a89561cf0bae610931fbc4c47a497345a` |
| `model-00002-of-00004.safetensors` | `2,735,851,872` | `95da61fce834e4f2d1f3cc5a80656bd91ce624164cbc7f8d16726cab3e9f8701` |
| `model-00003-of-00004.safetensors` | `2,735,851,872` | `a2a1480dedc6bbcc9f8f54c928ca6ae11eaf27a979b8de90beb8bdb8bd540e2e` |
| `model-00004-of-00004.safetensors` | `639,969,728` | `0e42e694d739a7a4c07eb6c93be86a55b180f077273f11b129206d126a489a1d` |
| `README.md` | `16,548` | `d5b893bb16a1c1f1290cdf1bf08c1c92774a3fbf08015bd3ff906d215c1c19f4` |
| `LICENSE` | `11,629` | `14cac091b400701631c226dc33d34384cd19aedbf465c8765ad86ac60e3e0a6e` |
| `.gitattributes` | `1,519` | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |

### Ext4 extent layout observations

- `dgx3` and `dgx4`: `0/4` fragmented shards (`posix_fallocate` landed all 4 shards at ext4's best 128 MiB-unit extent count: `21/21/21/5` extents).
- `dgx1` and `dgx2`: written with `posix_fallocate` + `fdatasync` + `POSIX_FADV_DONTNEED` without touching live GPU or O_DIRECT Engram serving during active hold `spark-ds41f-1126-drostbench-v1`. Before running `load_format=direct` benchmarks on `dgx1`/`dgx2`, inspect and defragment inside a maintenance window:

```bash
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-75/tp4.json doctor --fragmentation-commands
```

## Architecture and SM121 integration (`0870`)

1. **Zero drafter KV cache, zero prefill drafter overhead**:
   - `DFly2` shares the target model's SWA KV cache (`128` tokens, FP8 E4M3 + UE8M0) and compressed-MLA main KV cache (`512` top-k tokens, E2M1 + E4M3) plus the target's top-512 DSA index buffers at target layers `28, 32, 36`.
   - `uses_draft_kv_cache() == False`, `num_lookahead_tokens == 0`, `num_prefill_lookahead_tokens == 0`, and `separate_draft_input_tokens == 6` (`block_size = 6`, `num_speculative_tokens = 5`): 100% of `--kv-cache-memory-bytes` stays allocated to the 40 target layers, prefix-cache block alignment is untouched, and mid-prefill chunks skip `propose()`.
2. **SM121 TileLang kernel path (`vllm/models/deepseek_v4_1/tilelang/dfly2.py`)**:
   - `sparse_mla_fwd_v41(..., segments_only=True, row_stride_div=6|1)` reuses the warp-specialized SM121 decode kernel for both SWA and top-512 compressed-MLA segments across a fixed `640`-latent context window regardless of sequence length, emitting `[rows, H_local, 2, 512]` partial numerators and `[rows, H_local, 2]` log2 LSE.
   - `@tilelang.jit` `dfly2_merge_backbone_kernel` fuses the 6-token non-causal block-key RoPE dot products, learned `attn_sink`, and log2 online softmax merge.
   - `@tilelang.jit` `dfly2_merge_correction_kernel` fuses the 5-step sequential self-forced correction block-key dot products (`<= j`), `attn_sink`, `block_sink`, and gated relative bias (`sigmoid(gate_logits + block_gate_bias) + block_rel_bias`).
   - `@tilelang.jit` `dfly2_prepare_step_kernel` extracts `anchor_positions`, `prev_hidden`, `mtp_valid`, block token/position slots, SWA slot tables, and `< anchor_pos` rejection-masked top-512 indices in a single launch.
3. **Topology & memory coverage**:
   - Supports `tp4-ring` (`tp4.json`), `tp3-triangle` (`tp3.json`, uneven `64 -> 72` heads and `8 -> 9` `o_groups` padding), `tp3-triangle-4k` (`tp3-triangle-4k.json`), `tp2-direct` (`tp2-direct.json`), `tp1` (`tp1.json`, all 128 draft experts kept in VRAM under `dfly2_head` while target keeps `108/48` io_uring slots), and `pp2-tp2-ring` (`pp2-tp2-ring.json`, Stage 1 owns target layers `20..39` including `20/28/32/36` with zero Stage-0 aux hidden-state P2P transfer).
   - Dedicated workspace lane `1` (`_num_workspace_lanes == 2`), per-batch `[1..max_num_seqs] * 6` `FULL` CUDA graph capture, vocabulary-parallel top-2 greedy draft sampling, online Newton-IRLS margin confidence calibration, and `VLLM_SINK_REJECTION=1` compatibility.

## Pre-GPU verification completed (`dgx4`, CPU-only under active cluster hold)

- All 5 `@tilelang.jit` DFly2 and `sparse_mla_fwd_v41(segments_only=True)` kernels lowered cleanly to SM121a CUDA C (`Target({"kind": "cuda", "arch": "sm_121a"})`).
- `pytest tests/models/deepseek_v4_1/test_dfly2.py`: `6/6` passed, including `JitWarmupRegistry` / `prepare.declare_modules` kernel declaration (`dfly2_prepare_step_kernel`, `padding_rows`, `select_first_argmax_kernel` top-2, `dfly2_merge_backbone_kernel`, `dfly2_merge_correction_kernel`, `sparse_mla_fwd_v41(segments_only=True)`), TP1..TP4 64-row tile-aligned `_DFly2MergedColumnParallelLinear` sharding + uninitialized-arena padding zero-fill, full `2,392`-tensor `DS-V4.1-Flash-DFly2-B6` safetensors load into all 91 parameters, and bit-exact greedy parity against an independent AngelSpec reference.
- DS4.1 gate CPU regression suite (`test_deepseek_v41_serving_contract.py`, `test_deepseek_v4_1_prepare.py`, `test_direct_loader.py`): `67/67` CPU checks passed (`50/50` direct-loader checks).

## Build and two-boot screening commands (run inside a `scripts/lab.py` window)

Compare against the recorded `spark-ds41f-1126` suites (`manifests/suites/2026-11-26-spark-ds41f-1126-tp4` and `manifests/suites/2026-11-26-spark-ds41f-1126-tp3`); do not re-measure `spark-ds41f-1126`.

```bash
# 1. Prepare sources, build spark-ds41f:candidate-75 on dgx4, and distribute identical OCI digest:
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-75/image-dgx4.json build prepare
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-75/image-dgx4.json build check
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-75/image-dgx4.json build image --apply
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-75/tp3.json build distribute --apply

# 2. Run in-image release-tests gate bundle (includes test_dfly2.py with mounted /hub):
python3 scripts/release_tests.py gate .work/build/vllm HEAD /tmp/dfly2-gate.json experiments/2026-10-26-spark-ds41f-candidate/candidate-75/release-tests.json
python3 scripts/release_tests.py bundle /tmp/dfly2-gate.json spark-ds41f:candidate-75 /tmp/dfly2-gate-bundle dfly2-1

# 3. Inside an authorized lab window (dgx1:~/spark-hold.json free), defrag dgx1..dgx2 shards,
#    then run two-boot screening against recorded spark-ds41f-1126 references:
scripts/lab.py window open --note "DFly2-B6 two-boot screen"
# While stopped inside the window, defragment fragmented DFly2 blobs on dgx1/dgx2:
bin/spark --cluster-config experiments/2026-10-26-spark-ds41f-candidate/candidate-75/tp3.json doctor --fragmentation-commands
# Run two-boot screen / release suite comparison on the live site shape (tp3-triangle or tp4-ring):
scripts/lab.py run experiments/2026-10-26-spark-ds41f-candidate/candidate-75/screen-tp3.json
# Full release comparison against manifests/suites/2026-11-26-spark-ds41f-1126-tp3*:
scripts/lab.py run experiments/2026-10-26-spark-ds41f-candidate/candidate-75/release-tp3.json
```

## Runs and conclusion

- **Status**: Source integration, fleet weight distribution (`dgx1..dgx4`), SM121a TileLang lowering, `JitWarmupRegistry` declaration checks, TP1..TP4 merged-projection sharding, and real-checkpoint parity verification complete; GPU image build and two-boot benchmark runs pending removal of foreign hold `dgx1:~/spark-hold.json` (`holder: "spark-ds41f-1126-drostbench-v1"`).
