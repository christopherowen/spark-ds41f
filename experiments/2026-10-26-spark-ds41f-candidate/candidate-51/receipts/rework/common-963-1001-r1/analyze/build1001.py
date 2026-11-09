# usage: python3 build1001.py PULL_DIR OUT_result.json
# Assembles pr-1001's result.json from the analyses (a1001r1, asab, adiag sconf) and the pytest logs.
import json
import os
import re
import sys

P = sys.argv[1]
a = json.load(open(f"{P}/a1001r1.json"))
sab = json.load(open(f"{P}/asab.json")) if os.path.exists(f"{P}/asab.json") else None


def tail(path):
    if not os.path.exists(path):
        return None
    lines = [l for l in open(path).read().splitlines() if re.search(r"(passed|failed|error)", l) and l.startswith("=")]
    return lines[-1].strip("= ") if lines else None


def pick(key, metric, comp, ms):
    return {f"M={m}": a["cells"][key][m][f"{comp}:{metric}"] for m in ms if m in a["cells"][key]}


def band_counts(comp, metric):
    return a["counts"].get(f"{comp}:{metric}")


HI = ["72", "96", "97", "104", "107", "112", "120", "128"]
res = {
    "pr": 1001,
    "round": 1,
    "pr_head": "03e2d903daf9ea197b430e24b75e64b0eda7e737",
    "patch": "0784-Match-128-row-vocab-head-tile-pipeline-stages.patch (round 1; context-only rebase as candidate-51's, plus the screen tile)",
    "author": "Christopher Owen <3221756+christopherowen@users.noreply.github.com> (the PR's), author date Sun, 8 Nov 2026 19:35:24 +0000 kept",
    "mark": "The own mark's, unchanged: its bench_vocab_2boot.py (captured CUDA graphs) on the image, 2 boots, ABBA: projection and greedy epilogue at M=97..128 (the 128-row tile) faster beyond the interval, bit-identical logits and (max, index) pairs, M<=96 not worse; per the rework brief, at TP4 and TP3 (every cell of the own-mark bench) with no cell slower on a shape serving uses.",
    "change": {
        "kept": "TILE_CONFIGS[128] = dict(block_N=32, threads=256, step_groups=1, num_stages=2) and its test (the first version, context-only rebase)",
        "added": "SCREEN_TILE_CONFIGS[128] = dict(block_N=32, threads=256, step_groups=1, num_stages=2) (was num_stages=4, act_stages=2): the certified screen's 128-row tile, which serves the greedy pairs of 65..128 rows; test_screen_tile_config_128_uses_matched_two_stage_pipeline",
        "checked_and_left": "SCREEN_TILE_CONFIGS[32] (num_stages=4, act_stages=2): no matched config is faster in every cell (sconf, one boot, in-process, layers built s4a2, s3, s2): num_stages=2 read 1.3-2.5% faster at TP4 FP32 (0.6-0.9% faster or same at TP4 BF16) but 1.0% slower at TP3 FP32 M=24 and 0.3% at TP3 BF16 M=32; num_stages=3 5.6-7.4% slower at TP3 FP32. The in-process slot effect (up to ~5% on identical kernels) is as large as the num_stages=2 readings, so no gain is shown; the tile stays",
        "why_the_screen": "on this base serving's greedy pairs go through #997's certified two-pass screen (project_head_greedy_pairs: screen_vocab_greedy_partials with SCREEN_TILE_CONFIGS, then refine), which never reads TILE_CONFIGS; SCREEN_TILE_CONFIGS[128] had the pattern #1001 diagnosed (T.Pipelined stage=[2, 0, 3, 3]); num_stages=2 gives stage=[0, 0, 1, 1]: the same two activation stages, two weight stages fewer",
        "registers (cuobjdump of the cached cubins, sm_121a)": "vocab_screen_greedy_partials at the 128-row tile (256 threads): 193 registers with num_stages=4/act_stages=2 (10 cp.async commit groups, 36 LDGSTS), 78 with num_stages=2 (5, 18); at the 32-row tile (128 threads): 166 (s4a2), 87-88 (s3), 63-64 (s2), against 64-66 at the 16- and 64-row tiles",
    },
    "host": "sp8 (spark-0016, NVIDIA GB10)",
    "arms": {
        "base": "spark-ds41f:candidate-50 sha256:8c1ea1f62f62c3e2adc03b645c091a5499cf7163101aca32875c894b96f155db + candidate-50 no1009/overlay (4 files, ro): tree 14c5a2dc; TILE_CONFIGS[128] and SCREEN_TILE_CONFIGS[128] s4a2",
        "c51r1": "the same image + the 15 files where the corrected candidate-51 tree (35cd2a0e87a6: 0781, 0782 #963-r1, 0783, 0784 #1001-r1) differs from the image's (ro); TILE_CONFIGS[128] and SCREEN_TILE_CONFIGS[128] num_stages 2",
    },
    "evidence": "bench_vocab_r1.py = the own-mark run's bench_vocab_2boot_c51.py with SCREEN_BASE / SCREEN_CAND set per layer (base_1107: the base's s4a2 screen tiles; candidate: #1001-r1's) and M 17, 24 added (diff: common/h/bench_vocab_r1.diff), behind a >= 250 ms GPU warm-up, ABBA in one session. Comparisons as the own mark: in_process_c51r1 (#1001-r1 alone on the accumulated stack, the PR's own comparison), in_process_base (the same on the base arm), image_ab (each arm's own config: #1001-r1 with #963-r1). The layers of the in-process pairs are built base_1107 first, candidate second; a layer's slot moves graph-replay time on identical kernels (see slot_effect), so the screen gain is also measured slot-balanced (sab).",
}
res["measured"] = {
    "greedy_epilogue_graph_us (project_head_greedy_pairs, serving's greedy path)": {
        "counts in_process_c51r1": band_counts("in_process_c51r1", "greedy_graph"),
        "counts in_process_base": band_counts("in_process_base", "greedy_graph"),
        "counts image_ab": band_counts("image_ab", "greedy_graph"),
        "TP4 FP32 in_process_c51r1": pick("tp4_float32", "greedy_graph", "in_process_c51r1", HI),
        "TP3 FP32 in_process_c51r1": pick("tp3_float32", "greedy_graph", "in_process_c51r1", HI),
        "TP4 BF16 in_process_c51r1": pick("tp4_bfloat16", "greedy_graph", "in_process_c51r1", HI),
        "TP3 BF16 in_process_c51r1": pick("tp3_bfloat16", "greedy_graph", "in_process_c51r1", HI),
        "TP4 FP32 image_ab": pick("tp4_float32", "greedy_graph", "image_ab", HI),
        "TP3 FP32 image_ab": pick("tp3_float32", "greedy_graph", "image_ab", HI),
    },
    "projection_graph_us (project_head: #1001's first half, unchanged)": {
        "counts in_process_c51r1": band_counts("in_process_c51r1", "proj_graph"),
        "counts in_process_base": band_counts("in_process_base", "proj_graph"),
        "counts image_ab": band_counts("image_ab", "proj_graph"),
        "TP4 FP32 in_process_c51r1": pick("tp4_float32", "proj_graph", "in_process_c51r1", HI),
        "TP3 FP32 in_process_c51r1": pick("tp3_float32", "proj_graph", "in_process_c51r1", HI),
        "TP4 BF16 in_process_c51r1": pick("tp4_bfloat16", "proj_graph", "in_process_c51r1", HI),
        "TP3 BF16 in_process_c51r1": pick("tp3_bfloat16", "proj_graph", "in_process_c51r1", HI),
    },
    "proj_plus_argmax_graph": {
        "counts in_process_c51r1": band_counts("in_process_c51r1", "proj_argmax_graph"),
        "counts image_ab": band_counts("image_ab", "proj_argmax_graph"),
    },
}
slow = []
for key, rows in a["cells"].items():
    for m, row in rows.items():
        for comp in ("in_process_c51r1", "in_process_base", "image_ab"):
            for metric in ("greedy_graph", "proj_graph", "proj_argmax_graph"):
                if "slower" in row[f"{comp}:{metric}"]:
                    slow.append(f"{key} M={m} (tile {row['row_tile']}) {comp} {metric}: {row[f'{comp}:{metric}']}")
res["measured"]["every slower reading (graph)"] = slow
if sab:
    res["measured"]["sab: slot-balanced screen A/B (c51r1 arm, 2 boots; SCREEN_TILE_CONFIGS[128] s4a2 -> s2, four layers s4a2_a, s2_a, s2_b, s4a2_b)"] = {
        "counts": sab["counts"],
        "cells": {k: v["graph_us_s4a2_to_s2"] for k, v in sab["cells"].items()},
        "slot A/A (same config, early vs late layer)": {k: v["slot_aa_ratio"] for k, v in sab["cells"].items()},
    }
res["tests"] = {
    "files": "tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py, tests/kernels/test_deepseek_v41_tilelang_pdl.py, tests/v1/worker/gpu/test_greedy_verification.py (the PR's three), with -p tblib_stub",
    "smoke (new and changed config tests, before the bench)": tail(f"{P}/pr-common/smoke-new-tests-c51r1.txt"),
    "boot1": tail(f"{P}/pr-1001/pytest-c51r1-boot1.txt"),
    "boot2": tail(f"{P}/pr-1001/pytest-c51r1-boot2.txt"),
}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("ok")

res = json.load(open(sys.argv[2]))
res["bit_exact"] = True
res["bit_exact_detail"] = ("In every run and cell the bench asserts torch.equal of the logits of the 64-row tile, the base_1107 and "
                           "candidate heads, and of the greedy pairs of the 64-row partials, both heads' certified screens and "
                           "greedy_shard_pair on the logits (all four runs completed); the logits and pairs hashes are equal across "
                           "all four runs of both arms in every cell; sab's four layers give equal pairs in every cell of both boots.")
res["slot_effect_note"] = ("The in-process pairs build base_1107 first and the candidate second, and a layer's build slot moves graph "
                           "replay on identical kernels (sab: up to 7% at TP3 M=1, monotone with the build order; this bench at "
                           "M<=64, identical kernels: BF16 greedy pairs 1-3.8% slower in the second slot on both arms). The screen "
                           "gain is therefore also measured slot-balanced (sab), where the identical-kernel cells read same or "
                           "within 1.8% faster and the 128-row tile 3.3-16.7% faster.")
res["verdict"] = "corrected-meets-mark"
res["reason"] = (
    "Round 1 extends #1001's matched-stage change to the certified screen's 128-row tile, SCREEN_TILE_CONFIGS[128] (num_stages=4, "
    "act_stages=2 -> num_stages=2: 193 -> 78 registers), the tile through which serving's greedy pairs of 65..128 rows go on this base. "
    "Greedy epilogue (project_head_greedy_pairs) at M=97..128, by CUDA-graph replay, beyond the interval: 24 of 24 cells faster in "
    "process on the corrected tree (TP4 FP32 M=104 1275.51/1276.63 -> 1140.51/1143.63 us, 1.117x; TP3 FP32 M=120 1.168x), 24/24 in "
    "process on the base arm and 24/24 across images; slot-balanced (sab, two boots) 24/24 faster at 97..128 (1.097-1.167x) and 8/8 "
    "at M=72, 96 (1.033-1.102x). Projection (the first version's change, unchanged): 22 of 24 faster at 97..128 in process on the "
    "corrected tree (2 same, each from one outlier boot: TP3 FP32 M=97 1769.52, TP4 BF16 M=112 1342.70), 24/24 on the base arm and "
    "24/24 across images. Bit-identical throughout. M<=96 not worse: every slower reading at M<=64 is between layers with identical "
    "kernels (the in-process slot effect: BF16 greedy 1-3.8%, present on the base arm too), and at M=72 the only ones are two BF16 "
    "projection cells (0.995x, 0.998x; BF16 is not the deployed head dtype, VLLM_DS41_HEAD_DTYPE=float32), within the same slot "
    "effect's reading on identical kernels (TP4 BF16 M=48 projection 0.997x); across images every M<=96 projection cell is faster or "
    "same and every FP32 cell of every comparison is faster or same except identical-kernel cells. SCREEN_TILE_CONFIGS[32] was "
    "checked and left: no matched config was faster in every cell.")
res["notes"] = [
    "Measured on spark-ds41f:candidate-50 with the corrected candidate-51 tree mounted (15 files, verified file by file); the c51r1 arm also carries #963-r1, whose screen kernels are byte-identical to the base's, so image_ab's greedy cells are #1001-r1 alone up to placement, and its projection cells #1001 with #963-r1.",
    "The full-logits projection serves the logits processor and DSpark's Markov W2; greedy verification uses the certified screen.",
    "sconf (tile-32 and tile-128 screen configs) ran one boot, in process, layers in build order; it chose the tile-128 change, which sab then measured slot-balanced on two boots.",
    "start_utc in commands.txt is after the GPU lock was taken; other agents' jobs ran between runs.",
]
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("verdict added")
