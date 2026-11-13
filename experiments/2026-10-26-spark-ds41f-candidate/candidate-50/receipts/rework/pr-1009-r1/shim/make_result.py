"""Assemble result.json from the raw outputs (run on the laptop: JSON only)."""

import json
from pathlib import Path

D = Path(__file__).resolve().parent.parent  # the rework folder
cells = json.loads((D / "raw/json/cells-own-mark.json").read_text())
cmp = json.loads((D / "raw/json/compare-harness.json").read_text())


def harness_table():
    rows = []
    for run, steps in sorted(cmp["runs"].items()):
        for s in steps:
            rows.append(dict(run=run, step=s["scenario"], T=s["T"], decode_step=s["is_decode"],
                             decode_rows=s["num_decode_tokens"], real_rows=s["real_rows"],
                             error=s["error"], error_rank=s["error_rank"],
                             converged_passes=s["passes"] if s["converged"] else None,
                             ranks_agree=s["ranks_agree_on_output"]))
    return rows


pairs = {}
for k, v in cmp["pairs"].items():
    pairs[k] = {s: ("bit-exact" if e.get("bit_exact") else e) for s, e in v.items()}

bench_cells = []
for c in cells["cells"]:
    cell = {"shape": f"TP{c['tp_size']} T={c['num_tokens']} (rank 0: {c['local_rows']} local rows)"}
    for m in ("graph", "kernels", "eager"):
        x = c[m]
        cell[m] = {
            "no1009_front_us_boot1_boot2": x["no1009_front_1107_us"],
            "corrected_front_us_boot1_boot2": x["corrected_front_1009_us"],
            "speedup": x["speedup"],
            "saved_us": x["saved_us"],
            "faster_beyond_interval": x["faster_beyond_interval"],
            "front_1107_in_corrected_tree_us": x["corrected_tree_front_1107_us"],
        }
    cell["own_bench_as_is"] = c["own_bench_eager"]
    bench_cells.append(cell)

result = {
    "pr": 1009,
    "round": 1,
    "candidate": "candidate-50",
    "patch": "0778-perf-dsv41-sp-prefill-local-attention-front.patch",
    "verdict": "corrected-meets-mark",
    "reason": (
        "Root cause: #1009's _sp_front kept the indexer head weights rank-local in every "
        "sequence-parallel forward (attention.py:1276, index_weights[: sp_rows.real_rows]), but _forward "
        "splits the indexer by rank only in a step with prefill rows (attention.py:1514-1515: split is None "
        "when swa.is_decode). In an SP decode step every rank scores all T rows, and _index_query "
        "(attention.py:1617-1622) took index_weights[0:T] of the L local rows, so the scorer got T query "
        "rows and L weight rows. The tp4-ring warm-up's first eager tail shape past the capture limit, "
        "T=214 (16 requests of 13-14 rows, within the copy-draft decode threshold of 17), is past the SP "
        "threshold of 205 and is such a step: L=54, 'expected: 64, got: 54'. Fix: _sp_front reads the "
        "step's SWA metadata first and keeps the weights local only where _forward splits the indexer; in "
        "a decode step (and under _INDEX_SPLIT_CHECK, as before) it gathers them with sp_rows.gather. "
        "Steps with prefill rows are unchanged: same kernels, same collectives, byte-identical outputs. "
        "Reproduced on one GPU through vLLM's own TileLang attention at TP4 rank shapes (the exact error "
        "at rank 0), and at TP3 (got: 8) and TP8 (got: 27); the corrected tree passes every step at TP3, "
        "TP4 and TP8 and is byte-identical to candidate-50 without #1009 on every rank, two boots plus a "
        "final run. The own mark holds: by CUDA-graph replay and by CUPTI kernel time, same-session ABBA "
        "against candidate-50 without #1009, all 6 cells faster beyond the two-boot interval "
        "(5.0x-24.5x), none slower; #1009's own bench reproduces its gains and 0 differing bytes."
    ),
    "root_cause": {
        "where": [
            "vllm/models/deepseek_v4_1/attention.py:1276 (candidate-50 tree d5778ab5): _sp_front, index_weights = index_weights[: sp_rows.real_rows]",
            "vllm/models/deepseek_v4_1/attention.py:1514-1515: split = sp_rows if sp_rows is not None and not swa.is_decode else None",
            "vllm/models/deepseek_v4_1/attention.py:1617-1622: _index_query weights = index_weights[:rows] if index_weights.shape[0] == rows else index_weights[first:last] (first=0, last=T: a slice past the L local rows)",
            "vllm/models/deepseek_v4_1/attention.py:1881 -> select_rows -> tilelang/attention.py:437 _index_chunk -> tilelang/indexer.py:2087 score: the 64-row chunk of index queries beside 54 weight rows",
        ],
        "trigger": (
            "model_runner.capture_model -> adaptive_verification.warm_tail_shapes: capture sizes up to 107 "
            "(copy drafts widen 96 to 107), so the tail shapes are 160, 214, 428, ...; SP starts at 205 rows "
            "(capacity floor 2 MiB / 10,240 B/row). _dummy_run(214): min(214, max_num_seqs 16) = 16 requests of "
            "13-14 rows; the builder's reorder threshold is max(1 + 5, 1 + num_verification_tokens 16) = 17, so "
            "the step is a decode step (is_decode) that runs sequence-parallel. TP4: L = 54."
        ),
        "why_tp3_booted": (
            "Derived, not measured: tp3's sizing (max_num_seqs 8, decode rows 59) makes its first tail shape past "
            "the SP threshold 236 rows in 8 requests of 29-30 rows, a step with prefill rows, so candidate-49's "
            "tp3-switched boot never ran an SP decode step. The harness shows the first version fails an SP "
            "decode step at TP3 rank shapes too. tp4-switched and tp8-switched take tp4-ring's sizing and would "
            "run the same T=214 step (TP8: L=27, the harness's 'got: 27')."
        ),
    },
    "fix": {
        "file": "vllm/models/deepseek_v4_1/attention.py (_sp_front only; plus the test below)",
        "change": (
            "swa = _query_metadata(metadata[swa prefix]) if metadata is a dict, read before the head weights; "
            "index_weights = sp_rows.gather(index_weights) if _INDEX_SPLIT_CHECK or (swa is not None and "
            "swa.is_decode) else index_weights[: sp_rows.real_rows]; the fast-path test uses 'swa is not None' "
            "in place of isinstance(metadata, dict). +16/-5 lines, docstring included."
        ),
        "cost": (
            "None in a step with prefill rows (the path #1009 speeds up). In an SP decode step (the warm-up's "
            "tail shape, or a serving step of only short requests past the SP threshold): one extra all-gather "
            "of 64 B/row beside the 1,848 B/row grouped gather, as _INDEX_SPLIT_CHECK already did."
        ),
        "test": (
            "test_sp_prefill_with_replicated_fused_wqa_wkv_gathers_mxfp8_and_swa_records gains an is_decode "
            "parameter (6 new cases) and asserts that the index queries and their head weights have one row "
            "count; on candidate-50's attention.py its 3 decode cases without the split check fail."
        ),
        "diff_vs_candidate_50": "0778-r1-vs-candidate-50.diff",
        "applies": (
            "In place of candidate-50's 0778 on a fresh tree (c50-0777 = 0777's commit 668bc8056), then 0779 and "
            "0780: all apply, git diff --check 19f2c20ed4d6..HEAD clean, tree 638711070bbe (raw/verify-stack.txt). "
            "The tree differs from candidate-50's (d5778ab5) in attention.py and the attention test only."
        ),
        "sha256": {
            "0778-perf-dsv41-sp-prefill-local-attention-front.patch": "065dfa181066bdc9b4299f508bbde064a23a4b5947d0c7d8f16e25572c7e7a15",
            "candidate-50's 0778 (replaced)": "7f3fc988bc05b12b1a3193e580e991e7b01f8eff8608594a94b71cc0e7324ebc",
            "overlay/vllm/models/deepseek_v4_1/attention.py": "779f45d368560ee736385519cf2d9237427b5d1bd0c4ff3ee69c8f7dccca5d9e",
        },
    },
    "overlay": {
        "files": ["overlay/vllm/models/deepseek_v4_1/attention.py"],
        "mount": [
            "<overlay dir>/vllm/models/deepseek_v4_1/attention.py",
            "/opt/spark-ds41f/candidate/vllm/vllm/models/deepseek_v4_1/attention.py",
            "ro",
        ],
        "note": (
            "Over the spark-ds41f:candidate-50 image (tree d5778ab5), as candidate-50/no1009/overlay is mounted in "
            "tp4-no1009.json: the corrected tree differs from candidate-50 in this one non-test file. "
            "Not booted here: the tp4-ring boot is the coordinator's."
        ),
    },
    "repro": {
        "harness": "shim/attn_harness.py (docstring: what is real and what is simulated)",
        "code_path": (
            "vLLM's DeepseekV41TileLangAttention for layers 20 (KV, index and candidate source, compressor ratio 1) "
            "and 24 (index source without a compressor: #1009's _sp_front), built under EngineArgs' VllmConfig "
            "from the served checkpoint's config.json with the recipe's flags (TILELANG attention, tilelang "
            "linear/MoE, block 256, max_model_len 1048576, max_num_seqs 16, max_num_batched_tokens 16384) and "
            "the TP4 recipe's speculative geometry (5 drafts, copy drafts verify 16: decode threshold 17, decode "
            "arena 272 rows); per-rank head/group counts (TP3 24/3, TP4 16/2, TP8 8/1), replicated 32-head "
            "indexer; random weights through process_weights_after_loading and prepare.declare_modules; KV "
            "caches at the specs' page sizes; metadata from DeepseekV41B12xMetadataBuilder on BlockTables slot "
            "mappings. Layer 20 runs on all T rows; layer 24 runs once per rank under "
            "sp_prefill.activate(SPRows(T, W, r)) with the rank's L local rows, the TP all-gather simulated "
            "(every rank's input to the same call, iterated to a fixed point on the real rows: 2 passes "
            "everywhere). WO is the identity, so the result is WO's input."
        ),
        "steps": {
            "decode-tail-214": "the tp4-ring warm-up step: 16 requests of 13-14 rows, no context (as _dummy_run)",
            "decode-ctx-216": "16 decode requests of 13-14 rows with 200-455 rows of context",
            "prefill-4096": "one 4096-row prefill",
            "prefill-4097": "one 4097-row prefill (tail rank short: TP4 1022, TP3 1365, TP8 506 real rows)",
            "mixed-90+4006": "15 decode requests of 6 rows (contexts 300-454) beside a 4006-row prefill",
        },
        "candidate_50_as_built": {
            "TP4": "decode-tail-214 and decode-ctx-216: RuntimeError: kernel indexer_score input Weights shape[0] violates packed ABI constraint; expected: 64, got: 54 (rank 0), both boots: the tp4-ring boot's error",
            "TP3": "both decode steps: the same error, 'expected: 64, got: 8' (L = 72: the second 64-row chunk)",
            "TP8": "both decode steps: the same error, 'expected: 64, got: 27'",
            "prefill steps": "pass, byte-identical to candidate-50 without #1009 at TP3, TP4 and TP8",
        },
        "corrected": (
            "every step passes at TP3, TP4 and TP8 on every rank, and layer 24's attention output, top-k "
            "indices and SWA cache pages are byte-identical to candidate-50 without #1009 on every rank "
            "(boot 1, boot 2, and a final run of the shipped tree); layer 20's output identical in every pair; "
            "each arm byte-identical to itself across boots"
        ),
        "steps_table": harness_table(),
        "pairs": pairs,
    },
    "own_mark": {
        "mark": "Its SP attention-front bench: gain beyond the interval, bit-exact; prefill not worse (candidate-49 own mark)",
        "how": (
            "Same session on sp8, ABBA: boot 1 corrected then no1009, boot 2 no1009 then corrected, a fresh "
            "container each (shim/batch-bench-tests.sh). shim/bench_abba.py times #1009's own two kernel "
            "sequences with its own inputs (benchmark_shapes: rank 0 of TP3/TP4, T 4096/8192/16384): front_1107 "
            "(its run_baseline: the front candidate-50 without #1009 runs; every tree has it) and front_1009 "
            "(its run_option_b; only the corrected tree has it), by CUDA-graph replay (10 sequences a graph, 60 "
            "replays, median), CUPTI kernel time (sum of the sequence's kernels, median of 5 blocks of 40) and "
            "eager CUDA events (40 warm-up, 200 timed, median). #1009's own bench, unchanged "
            "(sha256 fb3af4c7, the candidate-49 receipt's file), ran on the corrected tree at each end of the "
            "session (it cannot import on the no1009 tree)."
        ),
        "judged_on": (
            "Graph replay and kernel time (release-benchmark step 7). SP prefill runs these kernels eagerly; "
            "eager events are recorded too and agree."
        ),
        "cells": bench_cells,
        "all_cells_faster_beyond_interval": cells["all_faster_beyond_interval"],
        "slower_cells": 0,
        "own_bench": {
            "all_cells_faster_beyond_interval": cells["own_bench_all_faster_beyond_interval"],
            "bit_exactness": "14 shapes (TP3/TP4 x T 60..4096, non-divisible T included) per boot, both boots: 0 differing bytes in q_values, q_scales, SWA cache, index_weights and index_slice_mxfp8",
            "vs_candidate_49_receipt": "eager speedups 4.97x-18.1x here vs 4.97x-15.4x in candidate-49's own-mark receipt",
        },
        "sanity": "front_1107 measured in both trees agrees within 0.1-3.3% per cell (the same kernels)",
        "kernels_per_sequence": bench_cells and cells["cells"][0]["kernels_per_sequence"],
        "prefill_not_worse": (
            "Not evaluable on sp8 (no collective; the boot and suites are the coordinator's). In a step with "
            "prefill rows the corrected tree runs the kernels and collectives candidate-50 ran (harness: 4 "
            "simulated gathers on both, byte-identical results), so the fix does not change #1009's prefill."
        ),
    },
    "tests": {
        "files": [
            "tests/kernels/test_deepseek_v41_tilelang_front.py",
            "tests/kernels/test_deepseek_v41_tilelang_columns.py",
            "tests/models/test_deepseek_v4_1_attention.py",
            "tests/models/test_deepseek_v4_1_sp_prefill.py",
            "tests/models/test_deepseek_v4_1_prepare.py",
            "tests/models/test_deepseek_v41_serving_contract.py",
            "tests/kernels/attention/test_deepseek_v41_tilelang.py",
        ],
        "why": "the file list #1009's own-mark receipts ran, plus the TileLang attention kernel tests",
        "corrected_boot1": "443 passed, 1 skipped, 1 xfailed in 290.97s (cold TileLang cache)",
        "corrected_boot2": "443 passed, 1 skipped, 1 xfailed in 33.63s",
        "candidate_50_as_built": "437 passed, 1 skipped, 1 xfailed (the 6 new cases are the difference)",
        "skip_and_xfail": "pre-existing in tests/kernels/attention/test_deepseek_v41_tilelang.py on both trees: compute-sanitizer not found; test_indexer_nonfinite_query_scale (known indexer NaN xfail)",
        "new_test_on_candidate_50": "3 failed, 9 passed: the is_decode cases without the split check fail (no head-weight gather)",
    },
    "host": "sp8 (spark-0016, NVIDIA GB10)",
    "image": "spark-ds41f:1107 sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce with each arm's candidate-50 files mounted (no candidate-50 image on sp8; the 19 patches are Python and TileLang only)",
    "trees": {
        "candidate-50 (c50)": "ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9 / tree d5778ab56647a20d94710ecb94cfd9a4de27bee9 (candidate-50's lock)",
        "corrected (fix)": "tree 638711070bbeb3a0217f87f1b059f10a2e8f67af",
        "no1009": "c50-0777 + 0779 + 0780; its non-test files equal candidate-50/no1009/overlay",
        "source": "spark-ds41f origin/main e3c4a6bb7b, candidate-50/vllm/series",
    },
    "not_done": [
        "No boot: the tp4-ring boot with the overlay is the coordinator's (dgx window not touched).",
        "bin/spark doctor and bin/spark build check were not run (no repository change proposed here; the series was replayed on sp8).",
        "The attention harness's .pt tensors (about 11 GB) stay on sp8 in ~/spark-lab/agents/candidate-50-rework-1009/out/json.",
    ],
    "receipts": {
        "commands": "commands.txt",
        "raw": "raw/ (every run's output; raw/json: harness digests, comparisons, bench JSON, cells)",
        "scripts": "shim/",
    },
}
(D / "result.json").write_text(json.dumps(result, indent=1) + "\n")
print("result.json written")
