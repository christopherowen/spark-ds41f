#!/usr/bin/env python3
"""Fill result-skeleton.json's verdict, results and history (laptop, after the runs), then
build_result.py adds the cells, stages and test runs. Usage: fill_result.py <rework dir>"""
import json
import sys
from pathlib import Path

D = Path(sys.argv[1])
s = json.loads((D / "result-skeleton.json").read_text())

s["verdict"] = "meets-mark"
s["verdict_basis"] = [
    "Fast path engages in the server: the renderer now tells its processor (InputProcessingContext.mm_prefix_metadata_cache) when there is no processor cache and prefix caching is on. Through the real AsyncLLM frontend every workload's turn 2+ request goes with its repeated images stripped (6-25 KB instead of 10-41 MB), and the DS4.1-checkpoint test shows it only with --mm-processor-cache-gb 0.",
    "The mark, round 1's bench unchanged, ABBA against candidate-54, two boots per arm: all 12 image cells are faster beyond the interval. Turn 2+ is 4.2x to 9.5x faster (1302x1302 JPEG x1 7.05x, x4 8.74x; 1920x1080 PNG 5.01x; 3840x2160 PNG x1 4.24x, x4 4.29x; 4032x3024 JPEG x4 9.51x). Turn 1 is 2% to 15% faster (JPEG 1302 x1 0.964, x4 0.959; PNG 1920 0.892; 3840 x1 0.849, x4 0.875; JPEG 4032 x4 0.981), so no cell is slower. Prompt token ids and mm hashes equal candidate-54's in every workload and turn.",
    "Text-only requests through the same frontend (short, long, n=4), ABBA, two boots per arm: no cell slower beyond the interval (all within); a text request runs no round-2 code beyond two empty checks.",
    "The JPEG slowdown was #978's own and is removed: the lazy-image header cache (moved decodes into the processor's thread, 3x the page faults in the HF work) now runs only with the fast path, and MultiModalHasher's multithreaded blake3 (woke the processor's thread onto Cortex-A725 cores) is reverted.",
    "The four risks are closed with tests: a schedule-time retryable miss, a retry materializer for every sampled child, bounded marks (LRU 4096), and retries off the output handler. Round 1's fixes and #972's rollback and shielded abort hold: the gate files pass (68, 6 more than round 1), and the six new AsyncLLM tests fail on round 1's tree.",
    "check_978_image_chat.py could not run (no live server on sp8; the command for the testbed is given). Caveat for the owner: with the recipe's process environment added to the bench (not the mark), 1920x1080 PNG turn 1 was slower beyond the interval (80.9/79.7 ms against 72.0/70.2) while the other 11 cells were faster or within; its stage diagnostic shows the processor's thread running the HF work on Cortex-A725 cores in one of two r2 boots (46 ms against 31 ms), the other boot faster than base. Production also pins the API server mostly to A725 cores (0792), which no bench here applies.",
]
s["serving_path_cells"]["mark"]["result"] = "met: turn 2+ faster beyond the interval in 6 of 6 workloads (4.2x-9.5x); turn 1 faster beyond the interval in 6 of 6; no cell slower; prompt token ids and mm hashes equal candidate-54's"
s["serving_path_cells"]["text_only"]["result"] = "not slower: every cell (submit and round trip, short, long and n=4) within the interval"
s["serving_path_cells"]["supplementary_recipe_environment"]["result"] = "turn 2+ faster beyond the interval in 6 of 6 (4.6x-11.3x); turn 1 faster beyond the interval in 4, within in 1 (1302x1302 JPEG x1), slower beyond the interval in 1 (1920x1080 PNG x1: r2 80.92/79.67 ms against 72.04/70.19); identity equal. Its stage diagnostic (prodenv-diag) puts the slow boot's HF work on A725 cores (little-core share 1.0, 46.0 ms against base 31.3/31.1), the other r2 boot at 65.1 ms against base 70.0/70.6."
s["development_history"] = [
    "screen1, diag3, diag4 (r2dev: SHA-256 header key inline, then pre-key plus a background record): fast path engaged (stripped items, 7-10x turn 2+); 1920x1080 PNG turn 1 about 1 ms slower with the key inline, bimodal with the background record.",
    "mark-pre-amend and mark2 (r2 with blake3 on every core for items of 4 MiB or more, background record): turn 2+ 7-18x faster in both; 1920x1080 PNG turn 1 faster in the first (55.5/58.0 against 58.0/58.6) and 25% slower in the second (73.4/71.8 against 58.5/58.1).",
    "mark3 (r2, and r2inline with the record inline): r2's 1920x1080 PNG turn 1 54.8 or 71.2 ms against base 58.8/58.3; inline 59.7/58.7. diag6 found the cause (processor thread on A725 cores after the rayon hash).",
    "final-mark (multithreaded hash removed, digest alongside the decode): no rayon effect, but three turn-1 cells slower beyond the interval by 0.8-3% (the hash-first order of the metadata path and the inline digest of small images); cold_path_microbench located them (load +0.45 ms for a 0.9 MB JPEG; apply +1 ms for a 1920x1080 PNG).",
    "r2f (final): new images take the processor's usual path with hashes on a helper thread; digests of 64 KiB or more alongside the decode; the mark above. Pre-final test runs (raw/superseded/tests-pre-final) found the scheduler's stripped_mm_misses missing on schedulers built without __init__ (2 tests of test_scheduler.py and test_async_scheduler.py), fixed with a class-level default before the final runs.",
]
s["tests"]["interpretation"] = {
    "gate_files (test_request_submission_cleanup.py, test_api_cancellation.py)": "r2: 68 passed (round 1: 62; +6 new). Round 1's tree with round 2's tests: 6 failed, 62 passed - exactly the six new AsyncLLM tests (children's materializers, retry off the handler, miss after returned output, stop-string finish, bounded marks, streaming input).",
    "pr978 (test_prefix_cache_fastpath, test_stripped_mm_scheduling, test_hasher, media/test_image, test_cache, test_deepseek_v4_1_image_span; round 1's -k exclusions)": "r2: 56 passed on two boots. Round 1's tree with round 2's tests: 9 failed, 5 errors (the four scheduler tests - the encoder scheduled without data, and the attribute the cached-encoder test reads; the serving-frontend test [0] and [4] - no flag on the server's context; the renderer conditions; loading without the fast path; the self-heal test's index-keyed materializers; the header-cache and processor tests error in their fixture without set_validated_image_meta_cache_enabled). candidate-54 (the four files it has): 40 passed.",
    "dirs (tests/v1/engine, tests/entrypoints/serve/utils, --continue-on-collection-errors)": "r2: 261 passed, 17 failed, 43 skipped, 6 errors; candidate-54 (round 1's recorded run, same image and command): 244 passed, 17 failed, 43 skipped, 6 errors. The 17 failures and 6 errors are identical (environmental: async tests without an async plugin, hub models absent offline, no ray, CUDA before fork); +17 passed are round 1's 11 and round 2's 6 new cases.",
    "extra (test_scheduler, test_async_scheduler, test_encoder_cache_manager, multimodal test_processing/test_registry/test_inputs/test_image, renderers test_warmup/test_multimodal_hashes)": "EXTRA_TEXT",
    "development": "raw/dev/: CPU-only and GPU development runs; dev10-r2f mounted a wrongly built arm (the whole tree) and is superseded by dev11-r2f",
}
(D / "result-skeleton.json").write_text(json.dumps(s, indent=1) + "\n")
print("filled")
