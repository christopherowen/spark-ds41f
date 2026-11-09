# usage: python3 build963.py PULL_DIR OUT_result.json
# Assembles pr-963's result.json from the analyses (a963r1, aaa, adiag) and the pytest logs.
import json
import os
import re
import sys

P = sys.argv[1]
a = json.load(open(f"{P}/a963r1.json"))
aa = json.load(open(f"{P}/aaa.json")) if os.path.exists(f"{P}/aaa.json") else None
screen = open(f"{P}/adiag-screen.txt").read() if os.path.exists(f"{P}/adiag-screen.txt") else None


def tail(path):
    if not os.path.exists(path):
        return None
    lines = [l for l in open(path).read().splitlines() if re.search(r"(passed|failed|error)", l) and l.startswith("=")]
    return lines[-1].strip("= ") if lines else None


def pdl(path):
    if not os.path.exists(path):
        return None
    m = [l for l in open(path).read().splitlines() if "PDL sources" in l or "violation" in l.lower()]
    return m[-3:] if m else None


def cells(sec, name, key=None, ms=None):
    rows = a[sec][name]
    out = {}
    for m, r in rows.items():
        if not m.isdigit() or (ms and m not in ms):
            continue
        rr = r[key] if key else r
        out[f"M={m}"] = rr["graph_us"]
    return out


def counts(tag):
    return a["counts"].get(tag, {}).get("graph")


res = {
    "pr": 963,
    "round": 1,
    "pr_head": "e5db47f172cfef9bbe77d4af8b061f6fc683969a",
    "patch": "0782-perf-dsv41-unconditional-pack12-plane-loads-and-swar.patch (round 1; file name kept for candidate-51's series)",
    "author": "Christopher Owen <3221756+christopherowen@users.noreply.github.com> (the PR's), author date Sun, 8 Nov 2026 12:00:00 +0000 kept",
    "mark": "The own mark's, unchanged: its screen_pack12_kernels.py timing half on the image (h963.py, here h963r1.py), 2 boots, ABBA against the base image, by kernel time and CUDA-graph replay: packed BF16 GEMM (compressor_up, M=1..256) and packed vocab projection and greedy partials faster beyond the interval, 0 ULP, including through #997's certified screen (in the base); and, per the rework brief, no cell slower than the base on a shape serving uses (TP4 and TP3, decode M up to 128).",
    "change": {
        "kept": [
            "vocab.py: vllm_vocab_apply_row_exceptions / _clocal (per-fragment-column exception accumulation) in packed_vocab_projection, packed_vocab_greedy_partials and packed_vocab_markov_greedy_partials, unchanged",
            "gemm.py: vllm_packed_bf16_load_row_bounds and pdl.chain() in _packed_bf16_gemm and packed_bf16_gemm_two, now only on grids of _PACKED_WIDE_TILES = 32 column tiles or more",
            "tests/kernels/deepseek_v41_pdl_sources.py: gemm_packed_bf16_partials, gemm_packed_bf16_splitk and unpack_packed_bf16 (already chained in the base) in CHAIN",
        ],
        "restored_to_base": [
            "vocab.py vllm_vocab_decode_vec16 (the byte-level SWAR rewrite dropped): the certified screen's vocab_row_thresholds and vocab_packed_refine_partials call it, so the screen's kernels are the base's again (byte-identical SASS, out/pr-common/sass-*.txt)",
            "gemm.py vllm_packed_bf16_filter_exc (early return when there are no exceptions): serving's split-K compressor GEMM and the prefill unpack use it, so those kernels are the base's again (byte-identical SASS)",
            "packed_bf16_gemm / _two on narrow grids (512x10240: 8 tiles, 128x5120: 2 tiles): the base's bounds search and plain launch (byte-identical SASS for all 6 narrow kernels)",
        ],
        "removed_from_CHAIN": "gemm_packed_bf16 and gemm_packed_bf16_two (a narrow grid does not wait; a wide one still waits before naming a pointer, which the plugin checks for every kernel that waits)",
        "added_test": "tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py::test_packed_bf16_gemm_wide_and_narrow_grids_exact (4 cases: packed_bf16_gemm 4096 and 512 rows, packed_bf16_gemm_two 2x1024 and 2x256 rows, K 1280, with exceptions): the kernel source waits and loads row bounds exactly on wide grids, and the outputs equal bf16_gemm / bf16_gemm_two bit for bit at 1, 16, 64 and 130 rows",
        "diff_vs_first_version": "git diff --stat first..r1: 4 files, +143/-47 (gemm.py 92, vocab.py 50, test_deepseek_v41_tilelang_linear.py +46, deepseek_v41_pdl_sources.py -2); corrected 0782 against its parent: 4 files, +248/-90 (the first version: 3 files, +171/-109)",
    },
    "host": "sp8 (spark-0016, NVIDIA GB10)",
    "arms": {
        "base": "spark-ds41f:candidate-50 sha256:8c1ea1f62f62c3e2adc03b645c091a5499cf7163101aca32875c894b96f155db + candidate-50 no1009/overlay (4 files, ro): tree 14c5a2dc (the own-mark run's base arm)",
        "c51r1": "the same image + the 15 files where the corrected candidate-51 tree differs from the image's (ro): tree 35cd2a0e87a6 = 0781, 0782 #963-r1, 0783 #987, 0784 #1001-r1 on the base; verified file by file (7,831 match)",
    },
}

res["diagnosis"] = {
    "method": "d963.py on the own-mark run's c51 arm: each variant compiled from its own copy of the module (h/var, built by h/mkvar.py from the base file and #963's), the variants of a cell timed v1..vn, vn..v1 in one process by CUDA-graph replay (20 calls a graph) and isolated CUPTI kernel time; the TileLang cache key includes the imported header text, so variants never share a kernel",
    "packed_bf16_gemm (2 boots; ratios base/variant by graph replay, adiag-gemm.txt)": {
        "rb (row-bounds load only)": "512x10240 0.949-0.974 and 128x5120 0.963-0.977 (slower at all 16 cells, by isolated kernel time too); 5120x512 1.010-1.061 and 4096x1280 1.011-1.040 (faster at all 16)",
        "pdl (pdl.chain() only)": "512x10240 0.961-0.997 (6 slower, 2 same) and 128x5120 0.993-0.997 (7 slower, 1 same); 5120x512 1.000-1.026 and 4096x1280 0.999-1.024 (9 faster, 7 same)",
        "963 (both)": "512x10240 0.926-0.978, 128x5120 0.967-0.982 (all slower); 5120x512 1.013-1.101, 4096x1280 1.014-1.068 (all faster)",
        "reading": "the row-bounds load is what loses on the narrow grids, where a few CTAs (8 and 2 column tiles of 64) each run a long K loop; the bounds search it removes is redundant (its result is the row's end) but evidently not free to remove there. pdl.chain() is a small loss there too. Both gain on the wide grids.",
    },
    "vllm_vocab_decode_vec16 in the projection and greedy partials (1 boot, TP4 32320x5120, adiag-vocab.txt)": {
        "variants": "base / 963 / exc (#963's exceptions, base decode) / dec (base exceptions, #963's decode)",
        "stressed (the PR's weights: every 300th element 1e-6, 576,659 exceptions)": "963 2.41-3.77x and exc 2.41-3.78x over base; 963/exc within 0.5% (exc 0.3-0.5% slower at M=1 and 16 only); dec 0.996-1.005x of base",
        "natural (#1001's bench weights: a random head's own 24,787 exceptions)": "963 1.003-1.244x and exc 1.002-1.245x; dec 0.997-1.013x",
        "reading": "the decode rewrite does not move these kernels; the per-column exceptions carry the whole gain",
    },
    "certified screen (base vs #963 decode, the only #963 change on that path)": screen or "see raw/diag/d963-screen-boot*.txt (pending at writing)",
}

res["measured"] = {
    "A_packed_bf16_gemm_graph_us_base_boots_to_c51r1_boots": {
        "compressor_up_5120x512 (the mark)": cells("A_gemm", "compressor_up_5120x512"),
        "indexer_wq_b_4096x1280": cells("A_gemm", "indexer_wq_b_4096x1280"),
        "compressor_down_512x10240 (narrow: base code)": cells("A_gemm", "compressor_down_512x10240"),
        "indexer_wk_128x5120 (narrow: base code)": cells("A_gemm", "indexer_wk_128x5120"),
        "counts (graph)": {k.split(":")[1]: counts(k) for k in a["counts"] if k.startswith("A_gemm:")},
        "kernel_isolated counts": {k.split(":")[1]: a["counts"][k]["kernel_isolated"] for k in a["counts"] if k.startswith("A_gemm:")},
        "identical-code notes": "512x10240 and 128x5120 compile the base's kernel (byte-identical SASS in both arms): their single non-same kernel_isolated and eager readings (512x10240 M=1 kiso 0.906x from one 94.68 us sample; 128x5120 M=512 kiso 0.999x; eager 512x10240 M=256 0.985x, 128x5120 M=6 0.989x) are the interval rule's noise on identical machine code",
    },
    "B_packed_vocab_projection_graph_us (stressed 0.35% exceptions)": {
        "counts own config": counts("B_proj_own"),
        "counts same TILE_CONFIGS[128] s4a2 / s2 (#963-r1 alone)": [counts("B_proj_tile128_s4a2"), counts("B_proj_tile128_s2")],
        "TP4": cells("B_proj", "tp4_32320x5120", "own_vs_own", ["1", "16", "64", "96", "107", "128", "2048"]),
        "TP3": cells("B_proj", "tp3_43096x5120", "own_vs_own", ["1", "16", "64", "96", "107", "128", "2048"]),
        "TP4 same config s2 at the 128-row tile": cells("B_proj", "tp4_32320x5120", "same_cfg_s2", ["96", "107", "128"]),
    },
    "C_packed_vocab_greedy_partials_plus_finish_graph_us": {
        "counts own config": counts("C_greedy_own"),
        "counts same config s4a2 / s2": [counts("C_greedy_tile128_s4a2"), counts("C_greedy_tile128_s2")],
        "TP4": cells("C_greedy", "tp4_32320x5120", "own_vs_own", ["1", "16", "64", "107", "128"]),
    },
    "D_certified_greedy_screen_graph_us (project_head_greedy_pairs, FP32 head, stressed weights)": {
        "same screen-config set on both arms (#963-r1 alone: byte-identical SASS for every screen kernel)": {
            "s4a2 (the base's SCREEN_TILE_CONFIGS; the own mark compared at this set)": {
                "counts": counts("D_screen_same_set_s4a2"),
                "TP4 M=97..128": cells("D_screen", "tp4_32320x5120", "same_set_s4a2", ["97", "104", "107", "112", "120", "128"]),
                "TP3 M=97..128": cells("D_screen", "tp3_43096x5120", "same_set_s4a2", ["97", "104", "107", "112", "120", "128"]),
            },
            "r1 (#1001-r1's SCREEN_TILE_CONFIGS[128] num_stages 2)": {
                "counts": counts("D_screen_same_set_r1"),
                "TP4 M=97..128": cells("D_screen", "tp4_32320x5120", "same_set_r1", ["97", "104", "107", "112", "120", "128"]),
                "TP3 M=97..128": cells("D_screen", "tp3_43096x5120", "same_set_r1", ["97", "104", "107", "112", "120", "128"]),
            },
        },
        "own vs own (each arm's serving config: #963-r1 with #1001-r1)": {
            "counts": counts("D_screen_own"),
            "TP4 M=97..128": cells("D_screen", "tp4_32320x5120", "own_vs_own", ["97", "104", "107", "112", "120", "128"]),
        },
        "the own mark's miss, revisited": "TP4 M=97..120 was 0.974-0.994x slower (base decode vs #963's); at the same set now: M=97 0.999x same, 104 0.994x slower, 107 0.998x same, 112 0.997x same, 120 1.005x faster, 128 same, on identical SASS",
    },
    "E_split_k_compressor_decode_graph_us (TileLangSplitLinearMethod: byte-identical SASS in both arms)": {
        "counts": a["counts"]["E_splitk"]["graph"],
        "1024x5120 x2 FP32": cells("E_splitk", "compressor_wkv_wgate_1024x5120_x2_fp32"),
        "512x5120 BF16": cells("E_splitk", "compressor_wkv_512x5120_bf16"),
    },
}
if aa:
    res["measured"]["A/A rerun of D and E (identical SASS; ABBA, 2 boots per arm, aa-DE-*)"] = {
        "counts": aa["counts"],
        "the cells that read slower in the main run": {
            "D s4a2 TP4 M=6, 17, 24, 104": {m: aa["D_screen"]["tp4_32320x5120"][m]["s4a2"]["graph_us"] for m in ("6", "17", "24", "104")},
            "D r1 TP3 M=120": aa["D_screen"]["tp3_43096x5120"]["120"]["r1"]["graph_us"],
        },
    }

res["tests"] = {
    "files": "tests/kernels/quantization/test_deepseek_v41_tilelang_vocab.py, tests/kernels/quantization/test_deepseek_v41_tilelang_linear.py, tests/kernels/test_deepseek_v41_tilelang_embedding.py, tests/kernels/test_deepseek_v41_tilelang_pdl.py (the PR's four), with -p tblib_stub and -p tests.kernels.deepseek_v41_pdl_sources",
    "smoke (new and changed config tests, before the bench)": tail(f"{P}/pr-common/smoke-new-tests-c51r1.txt"),
    "boot1": tail(f"{P}/pr-963/pytest-c51r1-boot1.txt"),
    "boot2": tail(f"{P}/pr-963/pytest-c51r1-boot2.txt"),
    "pdl_sources_plugin": [pdl(f"{P}/pr-963/pytest-c51r1-boot1.txt"), pdl(f"{P}/pr-963/pytest-c51r1-boot2.txt")],
}
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("ok")

# verdict and analysis (round 1)
res = json.load(open(sys.argv[2]))
res["bit_exact"] = True
res["bit_exact_detail"] = ("Every output hash (GEMM outputs, projection logits, greedy (max, index) pairs, certified-screen pairs, "
                           "split-K outputs) is equal across all four ABBA runs of both arms in every cell, across both "
                           "TILE_CONFIGS[128] variants and both screen-config sets, and in the A/A rerun; every diagnosis variant "
                           "(gemm base/pdl/rb/963, vocab base/963/exc/dec) gives the same bits: 0 ULP.")
res["slot_effect_note"] = ("In one process, two layers with identical kernels can differ by up to 7% in graph replay (h963r1 "
                           "section D: the second-built layer 3-5% faster at TP4 M<=64 on both arms; d963 sab: monotone with the "
                           "build order at TP3 M=1). Every #963-r1 comparison here is therefore cross-arm at the same layer slot "
                           "(same set, same config), and the in-process diagnosis of the screen decode is left inconclusive.")
res["verdict"] = "corrected-meets-mark"
res["reason"] = (
    "The corrected 0782 keeps #963's gains where they hold and is the base's machine code wherever #963 lost on a path serving uses. "
    "Gains, by CUDA-graph replay, beyond the interval, 0 ULP: packed BF16 GEMM 5120x512 (compressor_up, the mark) faster at all 8 M "
    "(1.023-1.094x, e.g. M=1 7.27/7.27 -> 6.65/6.64 us) and 4096x1280 at all 8 (1.015-1.086x); packed vocab projection faster in all "
    "26 cells (2.44-3.82x at the PR's 0.35% exception density; at the same TILE_CONFIGS[128] on both arms 20/20), greedy partials in "
    "all 22 (2.40-3.80x; same config 12/12). Restored: the certified screen's kernels (vllm_vocab_decode_vec16 is the base's: "
    "byte-identical SASS for every screen, refine, threshold, x-norm, compact and finish kernel in both arms), serving's split-K "
    "compressor GEMM and the prefill unpack (vllm_packed_bf16_filter_exc is the base's: identical SASS), and packed_bf16_gemm on the "
    "narrow 512x10240 and 128x5120 grids (the base's search and launch: identical SASS), which read same at all 16 M (0.986-1.005x) "
    "where the first version was 2-7.4% slower. The own mark's screen miss (TP4 M=97..120 0.974-0.994x) is gone: at the same screen "
    "set the screen reads same or faster at 97, 107, 112, 120, 128 and 0.994x at 104. The few slower readings on identical SASS "
    "(screen s4a2 set TP4 M=6, 17, 24, 104 by 0.2-0.6%; r1 set TP3 M=120 0.950x) are the interval rule's noise: an A/A rerun of the "
    "screen and split-K sections (ABBA, 2 boots per arm, same SASS) gives 16 such slower readings in 76 cells (0.2-4.4%), and TP3 "
    "r1 M=120 there reads 1.045x faster while TP4 r1 M=120 reads 0.956x slower. The PR's own diagnosis that the decode rewrite cuts "
    "unpack ALU did not show as time: within 0.5% in the projection and greedy partials at both densities, so it is dropped.")
res["analysis"] = {
    "what_serving_runs": {
        "packed_vocab_projection (project_head)": "logits processor (sampled requests, log-probabilities) and DSpark's Markov W2 projection: gains kept",
        "packed_vocab_markov_greedy_partials": "the DSpark Markov head's greedy pairs: the same per-column exception change (kept); not separately timed",
        "packed_vocab_greedy_partials": "not serving's greedy path on this base (#997's certified screen replaced it); timed as the PR's metric",
        "certified screen (x_norm, screen, row_thresholds, compact, refine, finish)": "serving's greedy verification: now the base's kernels",
        "packed_bf16_gemm / packed_bf16_gemm_two": "no serving caller (imported by linear.py, never called): the PR's microbench shapes",
        "packed_bf16_gemm_splitk / packed_bf16_unpack": "serving's compressor decode and prefill unpack: now the base's kernels",
    },
    "why_gate_not_drop": "packed_bf16_gemm has no serving caller, so dropping its change would also be correct; the brief asks to keep the gains where they hold, and the mark names compressor_up. The gate (_PACKED_WIDE_TILES = 32 column tiles of the launch grid) sits between the measured narrow (8, 2) and wide (64, 80) grids, so all four measured shapes keep their better variant; shapes between 8 and 64 tiles are unmeasured.",
    "pdl_chain_registry": "gemm_packed_bf16 and gemm_packed_bf16_two leave deepseek_v41_pdl_sources.CHAIN because a narrow grid no longer waits; a wide one still waits first, which the plugin checks for every kernel that waits (0 violations in both test runs)",
}
res["notes"] = [
    "Measured on spark-ds41f:candidate-50 with the corrected candidate-51 tree mounted (15 files, verified file by file); the c51r1 arm also carries #1001-r1 (TILE_CONFIGS[128] and SCREEN_TILE_CONFIGS[128] num_stages 2), so #963-r1 alone is the same-config (B, C) and same-set (D) comparison.",
    "h963r1.py = the own-mark run's h963.py with section D run for both screen-config sets on each arm and M 17, 24 added (diff: common/h/h963r1.diff); h963aa.py = h963r1.py restricted by H963_SECTIONS (the A/A rerun).",
    "Diagnosis boots: gemm 2, vocab 1 (vocab boot 2 dropped to measure the corrected tree first), screen 2. d963.py gained its sab section after the gemm, vocab and sconf runs; those sections are unchanged.",
    "The stack-repo commit the c51r1 arm was built from (f9a11e3fc, WIP messages) has the same tree as the final commits (e96074f90): only the commit messages changed.",
    "start_utc in commands.txt is after the GPU lock was taken; other agents' jobs (c53 rework) ran between runs.",
    "common/h/regs.sh is an abandoned helper (its shared-memory and register greps matched nothing); screenregs.py and sass.py replaced it.",
]
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("verdict added")
