#!/usr/bin/env python3
"""Write result.json for the PR #1016 rework from cells.json and the run metadata."""
import hashlib
import json
import sys

out_dir, cells_path, patch_path, pytest1, pytest2 = sys.argv[1:6]
C = json.load(open(cells_path))
patch_sha = hashlib.sha256(open(patch_path, "rb").read()).hexdigest()

cells = {}
for pair, rows in C["cells"].items():
    cells[pair] = {}
    for m, row in rows.items():
        r = {"r1_same_kernels_as_1107": row["r1_same_kernels_as_1107"],
             "bits_equal_across_images": row["bits_equal_across_images"],
             "all_arms_torch_equal_in_process": row["all_arms_torch_equal_in_process"]}
        for st, c in row["states"].items():
            p = c["paired_r1_minus_1107_us"]
            o = c["paired_pr1016_minus_1107_us"]
            r[st.replace("graph_", "")] = {
                "outcome": c["outcome"],
                "x_img_us": {"no1016_b1_b2": c["no1016_b1_b2"], "r1_b1_b2": c["r1_b1_b2"],
                             "delta": c["x_img_delta_us"], "ci95": c["x_img_ci95"],
                             "boot_range": c["x_img_boot_range"]},
                "paired_r1_minus_1107_us": p if p == "same-kernels" else {
                    "per_process": p["per_boot"], "pooled": p["pooled"],
                    "ci95_within": p["pooled_ci95"], "ci95_across_processes": p["process_ci95"]},
                "paired_original_1016_minus_1107_us": o if o == "same-kernels" else {
                    "pooled": o["pooled"], "ci95_within": o["pooled_ci95"],
                    "ci95_across_processes": o["process_ci95"]},
            }
        cells[pair][m] = r

result = {
    "pr": 1016,
    "round": 1,
    "verdict": "corrected-meets-mark",
    "mark": ("#1016's chained Q-B bench against 1107's actual configuration (tile-contiguous weights, "
             "threads=64 at (4096, 1280) block_M=64), CUDA-graph replay: gain where it claims, slower than "
             "1107 in no M cell in any L2 state the bench measures (warm, cold, NEXT), bit-exact"),
    "patch": "0768-perf-dsv41-align-qb-and-indexer-qb-tile-contiguous-decode.patch",
    "patch_sha256": patch_sha,
    "patch_applies": ("in place of candidate-47's 0768 on the 1107 tree: 0762..0767, this 0768, 0769 all pass "
                      "git apply --check and apply (setup_r1.sh, setup-r1.log); it changes only gemm.py and "
                      "test_deepseek_v41_tilelang_linear.py, which no other patch touches"),
    "change": ("1107's DECODE_FP8_CONFIGS plus one row change from #1016: the 64-row Q-B tiles of (12288, 1280) "
               "and (8192, 1280) go from block_N=64 (8192: num_stages=3) to block_N=128, block_K=128, "
               "num_stages=2. Everything else stays 1107's: the 16- and 32-row tiles, (4096, 1280) with "
               "threads=64 at block_M=64, and TILE_CONTIGUOUS_FP8 (no 16384/6144/2048 entries). The test "
               "asserts the 64-row tile of N >= 8192 is 128 x 128."),
    "claim": ("TP3/TP4/TP5 Q-B decode at 33..64 rows with the weights warm in L2. On the chained bench pair: "
              "TP4 -3.1 us paired (consistent in all four processes, x-img -2.9 us) and TP3 -5.1 us paired "
              "(three of four processes; x-img -7.7 us, beyond TP3's 5.2 us warm A/A floor). With the indexer "
              "kernel as the model builds it: TP4 -3.0 us (beyond the interval) and TP3 -3.9 us (within the "
              "across-process interval). Cold and NEXT: no change within the interval. M = 1..24: the same "
              "kernels as 1107."),
    "outcome_tally": {"same-kernels": 93, "gain": 2, "gain (x-img boot range; one process paired off)": 2,
                      "within": 8, "slower": 0},
    "host": "sp4 (spark-0012), NVIDIA GB10, driver 580.178.04; docker ps empty and ~/spark-hold.json absent "
            "before every run (recorded inside /tmp/spark-lab-gpu.lock in commands.txt)",
    "image": "spark-ds41f:1107 = sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce",
    "arms": {
        "no1016": "the image + candidate-47's 26 files with 0762..0767 and 0769 (no #1016), read-only; "
                  "gemm.py is 1107's (546dbbcc1812)",
        "r1": "the image + the same 26 files with 0762..0767, this 0768, 0769; gemm.py 3a7ecb3a49f6",
    },
    "method": ("One session on sp4. The ABBA order is no1016 boot1, r1 boot1, r1 boot2, no1016 boot2 (each a fresh "
               "container; h1016r1.py MODE=mark), followed by the pytest suite on r1 for two boots. A screening run "
               "(MODE=screen, no1016 overlay) came before the ABBA and chose the change. Each process times "
               "several config tables on the same weights, interleaved window by window in alternating order: "
               "1107's real table, #1016 as submitted, r1, 1107 plus only the threads=64 drop, and the image's "
               "fp8_decode_config. Identical kernels are timed once per window. Each (pair, M) block takes 40 "
               "windows by CUDA-graph replay: warm (20 back-to-back pairs in one graph), cold (48 MiB flush, then "
               "one replay) and NEXT (flush, the NEXT-window L2 prefetch of the pair's weights and scales, join, "
               "then one replay). Before each block the GPU runs 100 ms or more of that block's own graph. "
               "Pairs: mark_tp4/mark_tp3 are the own-mark pair (h1016's), whose indexer kernel is built from the "
               "(4096, 1280) decode tiles as the PR bench builds it. prod_tp4/prod_tp3 are the same Q-B with the "
               "indexer kernel built exactly as linear.py builds decode_rope_mxfp4. qb_tp8 is the TP8 Q-B plain "
               "decode, the model's only other user of the (4096, 1280) tiles. The PDL LayerChain (production "
               "builders, Q-B 8192) runs at c1/c8/c8x6."),
    "outcome_rule": ("same-kernels where r1 compiles 1107's kernels. Otherwise, gain or slower when the paired "
                     "r1-1107 delta pooled over the four processes lies beyond the larger of its within-process "
                     "CI95 and its across-process CI95 (t, 3 df), and the cross-image delta has the same sign. "
                     "'gain (x-img boot range; one process paired off)' when the within-process CI and separated "
                     "cross-image boot ranges agree but one process does not. Otherwise within."),
    "cells": cells,
    "aa_floor_x_img_us": C["aa_floor_x_img_us"],
    "aa_note": ("At M = 1..24, r1's kernels are 1107's, so their cross-image deltas are an A/A floor between "
                "processes. That floor is up to 1.0 us warm at TP4 and up to 5.2 us warm / 4.8 us cold at TP3. "
                "TP3's chained warm working set (about 20 MiB) nearly fills L2, so its warm time depends on each "
                "process's allocation. Several same-kernels cells read 'slower' or 'gain' on the boot range and "
                "are noise."),
    "layer_chain_x_img_us": C["layer_chain"],
    "layer_chain_note": ("Within the interval. c1 and c8 run the same kernels in both images, so their +13 / -24 us "
                         "deltas are the chain's cross-process A/A spread. c8x6 (the only changed rows) is "
                         "+14 +/- 36 us warm and +2 +/- 42 us cold. Digests are equal across images."),
    "bit_exact": {
        "outputs_equal_across_images_every_pair_and_M": all(
            row["bits_equal_across_images"] for rows in C["cells"].values() for row in rows.values()),
        "all_arms_torch_equal_in_process_every_pair_and_M": all(
            row["all_arms_torch_equal_in_process"] for rows in C["cells"].values() for row in rows.values()),
        "layer_chain_digests_equal_across_images": all(v["digests_equal_across_images"]
                                                       for v in C["layer_chain"].values()),
        "pytest": [pytest1, pytest2],
    },
    "m12_cause": (
        "The bench (and h1016) builds the indexer's fused RoPE + MXFP4 kernel from fp8_decode_config(4096, 1280, "
        "block_M), whose threads default is 128. The epilogue covers the block_M x block_N tile in serial passes "
        "of 2 * threads / block_N rows: one thread per column pair, 16-lane shuffles for amax, predicated on m < M. "
        "Each pass carries dependent loads (Positions[m], then that row's CosSin) and the Q/S stores. #1016 raised "
        "block_N at block_M=16 from 64 to 128, which halves the rows per pass from 4 to 2. Live passes go from "
        "ceil(M/4) to ceil(M/2): equal at M = 1..2, 2 vs 1 at M = 3, 3 vs 2 at M = 6, 6 vs 3 at M = 12, 8 vs 4 at "
        "M = 16. The halved grid (32 CTAs instead of 64) saves about 1.3 us warm at M = 1, but every extra pass "
        "costs latency, so M = 12 comes out slower in every L2 state. The screen's diag isolates this on the "
        "indexer kernel alone at block_M=16, warm/cold medians in us: bN64/t128 6.59/36.94, bN128/t128 "
        "7.94/38.86, bN128/t256 (4 rows per pass again) 6.79/37.70. At M = 16 the gap is +1.7 warm and +3.4 "
        "cold. The model's own indexer kernel (bN128/t256/2 stages, never read from DECODE_FP8_CONFIGS) does not "
        "change with #1016. On the TP8 Q-B plain kernel, which has no serial epilogue, the 16-row block_N=128 "
        "tile has a smaller crossover: -1.7 us at M = 1, +0.4 us warm at M = 12 in every process. Its mechanism "
        "was not isolated."),
    "findings": [
        ("In the model, DECODE_FP8_CONFIGS[(4096, 1280)] reaches TP8's Q-B only. The indexer Q-B decode kernel "
         "(linear.py, decode_rope_mxfp4) is built with the kernel defaults (block_M 16, block_N 128, block_K 128, "
         "2 stages, 256 threads) at every M, in 1107 and with #1016 alike. The chained bench's indexer half, built "
         "from the (4096, 1280) decode tiles, is therefore not what serving runs. In this harness the model's pair "
         "is faster than the bench's 1107 pair (TP4 M=48: 36.4 vs 45.3 us warm)."),
        ("#1016's M=48 gain in cold and NEXT L2 came almost entirely from dropping threads=64 on (4096, 1280) at "
         "block_M=64. On the bench's indexer kernel that is -4.1 warm / -8.1 cold / -7.8 NEXT us (TP4, pooled), "
         "but the model never runs that kernel. On the model's consumer, TP8 Q-B, the same drop gains nothing: "
         "+0.23 warm, +1.20 cold, +1.45 NEXT us pooled over four processes (+0.20 / +0.54 / +0.37 in the "
         "screen). It is slower in four of five processes in warm and cold."),
        ("The Q-B half of the M=48 change (block_N=128 at block_M=64 for 8192 and 12288) is the real gain. With "
         "the weights warm in L2 it is consistent at TP4 (four processes -2.75..-3.44 us on the bench pair, "
         "-2.75..-3.62 on the model's pair). At TP3 three of four processes show -6.2..-8.8 us, one shows +2.0, "
         "and the cross-image boots read 57.1/57.2 vs 50.3/48.7 us. Cold and NEXT are within the interval."),
        ("#1016's 32-row Q-B tiles gained nothing at M=24 beyond the across-process interval in any state. "
         "Single processes put the TP3 pair 4.4 us (screen) and 6.3 us (model's pair) slower with the weights "
         "in L2."),
        ("#1016's 16-row (4096, 1280) block_N=128 tile: in the bench, -1.9..-2.4 us warm at M=1..3 but +2.9 cold / "
         "+1.7 NEXT at M=12 (TP4, pooled). In the model it touches only TP8's Q-B: -1.7 us warm at M=1..3, "
         "+0.44 +/- 0.12 us warm at M=12, cold within."),
        ("NEXT matches cold in this harness (and in h1016): flush, prefetch and join before a single replay "
         "leave the pair as slow as a plain flush. The state is measured as the bench defines it."),
    ],
    "dropped_from_original": [
        "(4096, 1280) block_M=64 threads=64 drop: unstated in #1016. It helps only the bench's indexer kernel, "
        "which the model never builds from this table, and it is no faster (slower in 4 of 5 processes) on "
        "TP8 Q-B",
        "(4096, 1280) block_M=16 block_N 64->128: slower at M=12 in the bench (epilogue passes) and on TP8 Q-B",
        "(12288|8192, 1280) block_M=32 block_N 64->128: no gain at M=24; TP3 up to +4..6 us in single processes",
        "(16384|6144|2048, 1280) TILE_CONTIGUOUS_FP8 and DECODE_FP8_CONFIGS entries: TP2/TP6/TP7, outside the "
        "chained bench. TP6/TP7's trimmed rank unpacks and repacks Q-B, so they need their own change and bench",
    ],
    "caveats": [
        "The gain is small and confined to M = 33..64 rows with Q-B warm in L2. In cold and NEXT-prefetched L2, "
        "closer to serving, r1 is neutral within the interval.",
        "TP3 warm is allocation-sensitive (one process of four disagrees); TP4 is consistent.",
        "One host (sp4). No serving run; the next candidate's qualification decides end to end.",
    ],
    "reason": ("Bit-exact everywhere: outputs hash-equal across images for every pair and M, every arm torch.equal "
               "in process, LayerChain digests equal, and pytest passes on both boots. At M = 1..24 r1 compiles "
               "1107's kernels in every pair, so it cannot be slower there. At M = 48, the only changed cell, it "
               "gains beyond the interval with the weights warm in L2: TP4 -3.14 +/- 0.30 us paired (x-img -2.89, "
               "boot ranges separate) and TP3 -5.14 us paired (x-img -7.70, boot ranges separate; one process of "
               "four +2.0). It is within the interval cold and NEXT on both shapes and both indexer builds, and the "
               "model's TP8 Q-B is unchanged. No M cell in any L2 state is slower than 1107. The correction drops "
               "everything in #1016 that gained nothing, was slower, or fell outside the bench."),
}
json.dump(result, open(f"{out_dir}/result.json", "w"), indent=1)
print("wrote", f"{out_dir}/result.json")
