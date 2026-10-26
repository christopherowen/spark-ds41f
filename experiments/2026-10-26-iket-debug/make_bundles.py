#!/usr/bin/env python3
"""Write this experiment's kernel-lab bundles from the vLLM branch.

usage: make_bundles.py VLLM_REPO [--only NAME ...]

Each bundle runs on candidate 6's image, whose vLLM tree is the release's (BASE,
26382f5cf). Every package file a bundle's ref changes from BASE is mounted over it
(overlay/vllm); a package module the ref deletes is mounted as one that fails to
import (as experiments/2026-10-26-tp3-budget-memory/make_bundles.py does). ``hot-base``
runs next-candidate (NEXT, the branch's base), the others the IKET branch (IKET).

The steps run in order (run.py); each prints its exit status and the bundle fails if
any failed. ``{python}`` in a step is the image's interpreter and ``{out}`` the
bundle's output root under the node's compile cache, /c/iket/<bundle>-<UTC stamp>,
which the runner prints first. dgx4's Nsight Systems is mounted at /opt/nsys, and the
step bundles mount the node's Hugging Face hub (DS4.1-Flash-1017's config and
tokenizer; their weights are dummies) at /hf/hub.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = "/opt/spark-ds41f/candidate/vllm"
IMAGE = "spark-ds41f:19f2c20ed4d6-next-6"
BASE = "26382f5cf"
NEXT = "577091997"  # next-candidate when the branch was cut
IKET = "iket-debug"
NSYS = ["/opt/nvidia/nsight-systems/2025.3.2", "/opt/nsys"]
HUB = ["~/.cache/huggingface/hub", "/hf/hub"]
NVDISASM = ["/usr/local/cuda-13.0/bin/nvdisasm", "/opt/nvdisasm"]
MISSING = '''# Deleted at this ref; importing it fails as importing a missing module does.
raise ModuleNotFoundError(f"No module named {__name__!r}", name=__name__)
'''
HOT = "/b/iket_hot.py"
STEP = "/b/iket_step.py"
TABLE = "/b/iket_table.py"
PROFILE = ["run-iket", "--output-dir"]
NSYS_PROFILE = ["/opt/nsys/bin/nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none",
                "--export=sqlite", "--force-overwrite=true", "--output"]

TESTS = ["tests/kernels/test_deepseek_v41_tilelang_iket.py"]
MOE_OUTPUT_TEST = "tests/kernels/moe/test_moe_output_outlives_workspace.py"
PYTEST = ["{python}", "-m", "pytest", "-q", "-rfEs", "-p", "no:cacheprovider", "--noconftest"]
# The production recipe's variables that the TP1 engine reads (config/cluster.json).
STEP_ENV = {
    "TRANSFORMERS_OFFLINE": "1",
    "MAX_JOBS": "8",
    "VLLM_USE_V2_MODEL_RUNNER": "1",
    "VLLM_DS41_HEAD_DTYPE": "float32",
    "VLLM_DS41_PACKED_BF16_LM_HEAD": "1",
    "VLLM_L2_PREFETCH_KERNELS": "tilelang",
    "VLLM_L2_PREFETCH_GRID": "2",
    "SPARK_DS41F_DSPARK_DEAD_ROWS_TAU": "0.2",
    "SPARK_DS41F_DSPARK_MAIN_PROJ_TP": "1",
    "SPARK_DS41F_DSPARK_VOCAB_PARALLEL": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6",
}

# The recipe's environment (config/cluster.json "environment") for a one-GPU step:
# - Breakable CUDA graphs are off, as the recipe serves. Unset, vLLM turns them on for
#   DS4.1, runs the graphs' break points (the context-KV write, global-KV preparation)
#   eagerly and drops torch.compile.
# - Late admission's layer is scaled to six layers: the recipe's 30 of 40 is 4.5 of 6.
# - Left out: build parallelism (MAX_JOBS), cache paths, the multi-rank settings (NCCL,
#   sparknet, sockets) and the b12x loader plugin. Dummy weights load without the plugin.
# - TP3's display carve-out weights are left out too: a TP1 head is whole, about
#   2.5 GiB, more than the carve-out holds.
RECIPE_ENV = {
    **STEP_ENV,
    "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
    "VLLM_DS41_KERNEL_BACKEND": "tilelang",
    "VLLM_LATE_ADMISSION_LAYER": "4",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_MMAP_THRESHOLD_": "1048576",
    "OMP_NUM_THREADS": "8",
    "SAFETENSORS_FAST_GPU": "1",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    "VLLM_DISABLE_PYNCCL": "0",
    "VLLM_ENABLE_PCIE_ALLREDUCE": "0",
    "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
    "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "900",
    "VLLM_MAX_IMAGE_PIXELS": "16777216",
}
# The recipe's sizing for each shape: its stream count, and its context length,
# which sets each block-table row's width.
RECIPE_SIZING = {4: ["--max-num-seqs", "16"], 3: ["--max-num-seqs", "8"]}


def step_spec(ref, tp=1, requests=1, tokens=32, ctas=4, iters=1, skip="^(vision_|topk_topp)", grid=8):
    """A bundle timing DS4.1 decode steps, kernel by kernel and phase by phase: iket_step.py
    at TP1 with one rank's shapes at ``tp``, ``requests`` concurrent requests of ``tokens``
    tokens, then ``grid`` runs of the engine grid's dummy step at their shape; on under
    run-iket (``ctas`` CTAs a launch and ``iters`` iterations a loop record ranges), then
    nsys of off with the graphs, the records against the dry run's count, the timeline
    and tables of the measured requests, and their decode steps against the grid's."""
    shape = ["--tp-shape", str(tp), "--requests", str(requests), "--tokens", str(tokens),
             "--grid", str(grid), *RECIPE_SIZING.get(tp, []), "--max-model-len", "1048576",
             "--kv-bytes", str(6 << 30), "--prefix-caching"]
    return dict(
        ref=ref,
        what=f"DS4.1 decode steps timed by kernel and phase: TP{tp} rank shapes at TP1 (six layers "
             f"and the drafter, dummy weights, the production recipe's kernels, graphs, "
             f"environment and sizing), c{requests}, {tokens} tokens a request, then {grid} "
             f"engine-grid steps at that shape; on under run-iket ({ctas} CTAs a launch, {iters} "
             f"iteration(s) a loop), nsys of off with the graphs, the timeline and tables, and a "
             f"decode step against the grid's",
        env=RECIPE_ENV,
        mounts=[HUB],
        steps=[
            # The dry run alone, then the measured run sized from its largest process count
            # (run-iket reads only the first process's: iket_profile.py). Debug logs name
            # each launch, for a run that hangs (run 30's dry run spun in a graph replay).
            ("decode steps, on, under run-iket",
             ["{python}", "/b/iket_profile.py", "{out}", "--log-level", "debug", "--",
              "{python}", STEP, "on", "--iket", "{out}/trace-run/iket", "--ctas", str(ctas),
              "--iters", str(iters), "--skip", skip, *shape, "--out", "{out}/trace-run"], 7200),
            ("records against the dry run's count",
             ["{python}", "/b/trace_counts.py", "{out}/trace", "{out}/trace-run/iket/sources",
              "--tracker", "{out}/dry/tracker"], 1800),
            ("decode steps, nsys, off, graphs",
             [*NSYS_PROFILE, "{out}/nsys-graphs", "--cuda-graph-trace=node", "{python}", STEP, "off",
              *shape, "--out", "{out}/nsys-graphs-run"], 5400),
            # prepare_pos_seq_lens_kernel is the first of the kernels an engine step
            # launches once (run 29); a step that prepares prefill inputs is a prefill step.
            ("timeline and tables",
             ["{python}", TABLE, "{out}/trace", "--sources", "{out}/trace-run/iket/sources",
              "--between", "iket_step_marker_kernel", "--gap-us", "200",
              "--step-kernel", "prepare_pos_seq_lens_kernel",
              "--nsys", "graphs={out}/nsys-graphs.sqlite", "--json", "{out}/table.json"], 1800),
            ("nsys timeline", ["{python}", "/b/nsys_window.py", "{out}/nsys-graphs.sqlite",
                               "--gap-us", "200"], 600),
            ("decode steps against the grid's step",
             ["{python}", "/b/step_diff.py", "{out}/nsys-graphs.sqlite", "--grid-runs", str(grid + 1),
              "--table", "{out}/table.json", "--json", "{out}/step-diff.json"], 600),
        ],
    )


def dry_loop_spec(what, harness, timeout=600, repeats=8):
    """A bundle running the TP3 c1 step bundle's run-iket dry run ``repeats`` times, each
    killed after ``timeout`` s, with ``harness`` as iket_step.py's mode and options."""
    run = [*harness[:1], *(["--iket", "{out}/dry-{k}/run/iket"] if harness[0] == "on" else []),
           *harness[1:]]
    return dict(
        ref=IKET,
        what=what,
        # step-tp3-c1's compile caches: each dry run starts as run 42's did.
        env={**STEP_ENV, **{variable: f"/c/{prefix}-iket-step-tp3-c1" for variable, prefix in (
            ("TILELANG_CACHE_DIR", "tilelang"), ("TRITON_CACHE_DIR", "triton"),
            ("VLLM_CACHE_ROOT", "vllm"))}},
        mounts=[HUB],
        steps=[
            (f"dry run {k}",
             ["{python}", "/b/iket_profile.py", f"{{out}}/dry-{k}", "--timeout", str(timeout),
              "--log-level", "debug", "--dry-only", "--", "{python}", STEP,
              *(part.replace("{k}", str(k)) for part in run), "--tp-shape", "3", "--requests", "1",
              "--tokens", "32", "--grid", "8", "--out", f"{{out}}/dry-{k}/run"], timeout + 300)
            for k in range(1, repeats + 1)
        ],
    )


# The decode step's instrumented kernels (run 18) by family. Each step-counts dry run
# instruments one family, to find the kernel whose warps record past the static bound.
COUNT_FAMILIES = {
    "gemm": "gemm_bf16_kernel gemm_bf16_partials_kernel gemm_bf16_splitk_kernel gemm_bf16_two_kernel "
            "gemm_fp8w_bf16_cast_kernel gemm_fp8w_bf16_partials_kernel gemm_mxfp8_kernel "
            "gemm_mxfp8_partials_kernel gemm_splitk_reduce_cast_kernel gemm_splitk_reduce_kernel "
            "vocab_packed_projection_kernel vocab_embedding_kernel l2_prefetch_kernel",
    "moe": "moe_combine_kernel moe_dispatch_kernel moe_grouped_fp8_fp4_gemm_kernel "
           "moe_swiglu_cast_kernel_kernel moe_topk_gate_forward_kernel_kernel router_partials_kernel "
           "routing_masks_kernel activation_swiglu_clamp_kernel",
    "attention": "sparse_mla_kernel indexer_mxfp4_quantize_kernel indexer_score_kernel "
                 "indexer_topk_merge_kernel indexer_topk_segment_kernel compress_gather_partial_state_kernel "
                 "compress_kernel cache_writer_kernel rope_kernel metadata_builder_kernel "
                 "metadata_chunk_kernel metadata_decode_kernel metadata_pages_kernel",
    "mhc": "mhc_collapse_column_kernel mhc_collapse_kernel mhc_finalize_kernel mhc_post_kernel "
           "mhc_project_streams_kernel norm_forward_and_per_token_cast_kernel_kernel "
           "per_token_cast_kernel_kernel split_columns_kernel",
    "inputs": "ced_copy_rows_kernel ced_indices_kernel ced_requests_kernel ced_routing_mask_kernel "
              "ced_slots_kernel ced_token_metadata_kernel compute_slot_mappings_kernel "
              "gather_block_tables_kernel context_inputs_kernel prepare_pos_seq_lens_kernel "
              "prepare_prefill_inputs_kernel prepare_dflash_inputs_kernel expand_idx_mapping_kernel "
              "combine_sampled_and_draft_tokens_kernel post_update_kernel fill_ids_kernel fill_rows_kernel "
              "padding_rows_kernel scatter_rows_gather_kernel gather_rows_kernel clear_rows_kernel "
              "apply_write_kernel insert_kernel zero_packs_kernel iket_step_marker_kernel",
    "sampler": "bias_kernel bad_words_kernel bincount_kernel penalties_kernel temperature_kernel min_p_kernel "
               "topk_ids_kernel topk_ids_only_kernel topk_log_softmax_kernel topp_mask_kernel "
               "topp_stats_kernel topp_step_kernel gumbel_finish_kernel gumbel_sample_kernel resample_kernel "
               "finish_argmax_kernel partial_argmax_kernel logits_rows_kernel local_stats_kernel "
               "token_counts_kernel token_ranks_kernel prompt_ids_kernel grammar_bitmask_kernel "
               "flatten_sampled_kernel greedy_rejection_kernel verify_kernel dspark_confidence_kernel "
               "sink_candidates_kernel sink_cut_kernel sink_flags_kernel sink_merge_kernel sink_policy_kernel "
               "sink_rows_kernel sink_shard_top_kernel",
}

BUNDLES = {
    "hot-base": dict(
        ref=NEXT,
        what="next-candidate without the IKET mode: the hot kernels' generated sources and output bits",
        steps=[("hot kernels, next-candidate", ["{python}", HOT, "off", "--out", "{out}/off"], 3600)],
    ),
    "hot": dict(
        ref=IKET,
        what="the IKET mode on the hot kernels (routed experts, sparse MLA decode, the mHC entry, "
             "sparknet's all-reduce with simulated peers): its tests, off, on outside the profiler, "
             "on under run-iket, and nsys durations of both builds",
        tests=TESTS,
        steps=[
            ("IKET mode tests", [*PYTEST, "--basetemp", "{out}/pytest", *TESTS], 1800),
            ("off", ["{python}", HOT, "off", "--out", "{out}/off"], 3600),
            ("on, outside the profiler",
             ["{python}", HOT, "on", "--iket", "{out}/on/iket", "--out", "{out}/on"], 3600),
            ("on, under run-iket",
             [*PROFILE, "{out}/trace", "profile", "--postprocess", "all", "--",
              "{python}", HOT, "on", "--iket", "{out}/trace-run/iket", "--out", "{out}/trace-run",
              "--iters", "3"], 5400),
            ("all-reduce, peers 20 us late, under run-iket",
             [*PROFILE, "{out}/trace-late", "profile", "--postprocess", "all", "--",
              "{python}", HOT, "on", "--iket", "{out}/trace-late-run/iket", "--out", "{out}/trace-late-run",
              "--iters", "3", "--only", "allreduce", "--delay-us", "20"], 3600),
            ("nsys, off", [*NSYS_PROFILE, "{out}/nsys-off", "{python}", HOT, "off",
                           "--out", "{out}/nsys-off-run", "--iters", "50"], 3600),
            ("nsys, on", [*NSYS_PROFILE, "{out}/nsys-on", "{python}", HOT, "on",
                          "--iket", "{out}/nsys-on-run/iket", "--out", "{out}/nsys-on-run", "--iters", "50"],
             3600),
            ("per-phase table",
             ["{python}", TABLE, "{out}/trace", "--sources", "{out}/trace-run/iket/sources",
              "--nsys", "off={out}/nsys-off.sqlite", "on={out}/nsys-on.sqlite", "--json", "{out}/table.json"],
             600),
            ("per-phase table, peers 20 us late",
             ["{python}", TABLE, "{out}/trace-late", "--sources", "{out}/trace-late-run/iket/sources",
              "--json", "{out}/table-late.json"], 600),
        ],
    ),
    "sass": dict(
        ref=IKET,
        what="where ptxas puts the IKET events' timer reads around barriers, in the hot kernels",
        env={"NVDISASM": "/opt/nvdisasm"},
        mounts=[NVDISASM],
        steps=[
            ("hot kernels, on", ["{python}", HOT, "on", "--iket", "{out}/on/iket", "--out", "{out}/on",
                                 "--iters", "1"], 3600),
            ("SASS check", ["{python}", "/b/sass_check.py", "{out}/on/iket/sources", "{out}/sass",
                            "oneshot", "moe_grouped", "sparse_mla", "mhc_finalize"], 1800),
        ],
    ),
    "step": dict(
        ref=IKET,
        what="decode steps end to end at TP1 (six layers and the drafter, dummy weights, the "
             "production recipe's kernels and graphs): on under run-iket (four CTAs a launch "
             "and the middle iteration of each loop record ranges; the vision tower and the "
             "sampler's top-k/top-p kernel uninstrumented), nsys durations off without and with "
             "the graphs, and the decode-step table",
        env=STEP_ENV,
        mounts=[HUB],
        steps=[
            # run-iket's pool is a bump allocator for the whole run: every warp of every
            # instrumented launch gets the run's largest warp record (5,016 timestamps
            # unsampled: 898 GB). Sampled CTAs and iterations bound that record, once
            # every sampled event is in a loop of zero or one iterations: a warp takes a
            # slot for every event it issues, and ptxas predicated guarded events (the
            # dry run's 424 a warp, runs 14-20; a cap under it crashed the exit dump).
            # The vision tower and the top-k/top-p sampler are not in a greedy text decode
            # step; left uninstrumented, their warps reserve no records.
            # run-iket reads only the first process's count (run 28): iket_profile.py.
            ("decode steps, on, under run-iket",
             ["{python}", "/b/iket_profile.py", "{out}", "--",
              "{python}", STEP, "on", "--iket", "{out}/trace-run/iket", "--ctas", "4",
              "--iters", "1", "--skip", "^(vision_|topk_topp)", "--out", "{out}/trace-run"], 7200),
            ("what sized the buffer", ["{python}", "/b/trace_counts.py", "{out}/trace",
                                       "{out}/trace-run/iket/sources", "--tracker", "{out}/dry/tracker"],
             1800),
            ("decode steps, nsys, off, eager",
             [*NSYS_PROFILE, "{out}/nsys-eager", "{python}", STEP, "off", "--eager",
              "--out", "{out}/nsys-eager-run"], 5400),
            ("decode steps, nsys, off, graphs",
             [*NSYS_PROFILE, "{out}/nsys-graphs", "--cuda-graph-trace=node",
              "{python}", STEP, "off", "--out", "{out}/nsys-graphs-run"], 5400),
            ("decode-step table",
             ["{python}", TABLE, "{out}/trace", "--sources", "{out}/trace-run/iket/sources",
              "--between", "iket_step_marker_kernel", "--gap-us", "200",
              "--nsys", "eager={out}/nsys-eager.sqlite", "graphs={out}/nsys-graphs.sqlite",
              "--json", "{out}/table.json"], 1800),
            *((f"nsys timeline, {mode}",
               ["{python}", "/b/nsys_window.py", f"{{out}}/nsys-{mode}.sqlite", "--gap-us", "200"], 600)
              for mode in ("eager", "graphs")),
        ],
    ),
    "guards": dict(
        ref=IKET,
        what="why the dry run's per-warp maximum exceeds the warps' records under sampling: "
             "the sampled hot kernels' SASS (which events ptxas predicated rather than "
             "branched around) and a dry run's own log of its maximum",
        env={"NVDISASM": "/opt/nvdisasm"},
        mounts=[NVDISASM],
        steps=[
            ("hot kernels, on, sampled",
             ["{python}", HOT, "on", "--iket", "{out}/on/iket", "--out", "{out}/on", "--iters", "1",
              "--sample-ctas", "4", "--sample-iters", "1"], 3600),
            ("SASS check, sampled", ["{python}", "/b/sass_check.py", "{out}/on/iket/sources", "{out}/sass",
                                     "oneshot", "moe_grouped", "sparse_mla", "mhc_finalize", "moe_dispatch"],
             1800),
            ("dry run, debug log",
             [*PROFILE, "{out}/dry", "--skip-run", "--log-level", "debug", "profile", "--keep",
              "--postprocess", "none", "--", "{python}", HOT, "on", "--iket", "{out}/dry-run/iket",
              "--out", "{out}/dry-run", "--iters", "3", "--sample-ctas", "4", "--sample-iters", "1"], 3600),
            ("the dry run's maximum", ["{python}", "-c",
                "import glob, sys\n"
                "for p in sorted(glob.glob(sys.argv[1] + '/tracker/pid_*/smodel.injection.log')):\n"
                "    for line in open(p, errors='replace'):\n"
                "        if 'MaxTimestamp' in line or 'recommended' in line:\n"
                "            print(p.split('/')[-2], line.rstrip()[:300])",
                "{out}/dry"], 60),
        ],
    ),
    # The c3/c8 NaN off the IKET branch (coordinator, 2026-10-26): next-candidate's head,
    # 1025's runtime tree (the image's 26382f5cf; its 0489-0490 change only tests), and
    # the fix (vLLM moe-output-fresh: a modular MoE kernel's output never lives in the
    # workspace arena). Runs 38-40: both trees fail; the NaN enters layer 1's mHC pre
    # through its previous_output, layer 0's routed output, which lies in the workspace
    # lane its scratch reuses (at c1 too).
    **{name: dict(
        ref=ref,
        what=f"the c3/c8 NaN on {label}, IKET off: the fix's test, then c1 and c3 eager with a "
             f"non-finite hook on every module, c3 with the graphs, and c8",
        # 1025's tree still defaults to the B12X family; the recipe picks TileLang.
        env={**STEP_ENV, "VLLM_DS41_KERNEL_BACKEND": "tilelang"},
        mounts=[HUB],
        tests=[MOE_OUTPUT_TEST],
        tests_ref="moe-output-fresh",
        steps=[
            ("the fix's test", [*PYTEST, "--basetemp", "{out}/pytest", MOE_OUTPUT_TEST], 600),
            *((f"off, TP4, c{requests}, eager, nan hooks",
               ["{python}", STEP, "off", "--tp-shape", "4", "--requests", str(requests), "--eager",
                "--nan-hooks", "--out", f"{{out}}/c{requests}-eager"], 1800)
              for requests in (1, 3)),
            *((f"off, TP4, c{requests}, graphs",
               ["{python}", STEP, "off", "--tp-shape", "4", "--requests", str(requests),
                "--out", f"{{out}}/c{requests}"], 1800)
              for requests in (3, 8)),
        ],
    ) for name, ref, label in (("nan-next", "c579b3fe1", "next-candidate c579b3fe1"),
                               ("nan-1025", BASE, "1025's tree (26382f5cf, no overlay)"),
                               ("nan-fix", "moe-output-fresh", "the fix over next-candidate"))},
    "step-counts": dict(
        ref=IKET,
        what="which kernels' warps record past the static bound in the eager decode steps: one "
             "run-iket dry run (--skip-run) per kernel family, the others uninstrumented, and "
             "each dry run's per-warp maximum",
        env=STEP_ENV,
        mounts=[HUB],
        steps=[
            *((f"dry run, {family} instrumented",
               [*PROFILE, f"{{out}}/{family}", "--skip-run", "profile", "--keep", "--postprocess", "none",
                "--", "{python}", STEP, "on", "--eager", "--iket", f"{{out}}/{family}-run/iket",
                "--ctas", "4", "--iters", "1",
                "--skip", f"^(?!(?:{'|'.join(names.split())})$)", "--out", f"{{out}}/{family}-run"], 1800)
              for family, names in COUNT_FAMILIES.items()),
            ("per-warp maximum per family", ["{python}", "-c",
                "import glob, json, sys\n"
                "for d in sys.argv[1:]:\n"
                "    for p in sorted(glob.glob(d + '/tracker/pid_*/tracker.json')):\n"
                "        t = json.load(open(p))\n"
                "        print(d.rsplit('/', 1)[1], p.split('/')[-2], t.get('recommended_max_ts_cnt_per_warp'), "
                "len(t['launches']), 'launches')",
                *(f"{{out}}/{family}" for family in COUNT_FAMILIES)], 60),
        ],
    ),
    "hot-sampled": dict(
        ref=IKET,
        what="what sizes run-iket's buffer: the hot kernels under run-iket unsampled and with "
             "four CTAs a launch, the middle iteration of each loop and lane 0 recording, the "
             "tracker kept; its per-warp maximum against the traces' and the static bounds",
        steps=[
            ("unsampled, under run-iket",
             [*PROFILE, "{out}/full", "profile", "--keep", "--postprocess", "json", "--",
              "{python}", HOT, "on", "--iket", "{out}/full-run/iket", "--out", "{out}/full-run",
              "--iters", "3"], 3600),
            ("sampled, under run-iket",
             [*PROFILE, "{out}/sampled", "profile", "--keep", "--postprocess", "json", "--",
              "{python}", HOT, "on", "--iket", "{out}/sampled-run/iket", "--out", "{out}/sampled-run",
              "--iters", "3", "--sample-ctas", "4", "--sample-iters", "1"], 3600),
            # The dry run's maximum is above every warp's record; a cap between the two:
            # does the run's exit dump hold (it crashed in the engine runs 15 and 18)?
            ("sampled, under run-iket, capped at 64",
             [*PROFILE, "{out}/capped", "profile", "--keep", "--max-ts-cnt-per-warp", "64",
              "--postprocess", "json", "--",
              "{python}", HOT, "on", "--iket", "{out}/capped-run/iket", "--out", "{out}/capped-run",
              "--iters", "3", "--sample-ctas", "4", "--sample-iters", "1"], 3600),
            ("unsampled counts", ["{python}", "/b/trace_counts.py", "{out}/full",
                                  "{out}/full-run/iket/sources"], 1800),
            ("sampled counts", ["{python}", "/b/trace_counts.py", "{out}/sampled",
                                "{out}/sampled-run/iket/sources"], 1800),
            ("capped counts", ["{python}", "/b/trace_counts.py", "{out}/capped",
                               "{out}/capped-run/iket/sources"], 1800),
        ],
    ),
    **{f"step-tp{tp}-c{requests}": step_spec(IKET, tp, requests) for tp in (4, 3) for requests in (1, 8)},
    # Run 42's TP3 c1 dry run stopped launching between a kernel and the next graph replay
    # (as run 30's did at TP4): the dry run repeated, each bounded, while a host-side
    # watcher (hangcatch.sh) records stacks and GPU power when a run's log stops growing.
    "dry-loop-tp3-c1": dry_loop_spec(
        "the TP3 c1 step bundle's run-iket dry run eight times, each killed after 600 s, to "
        "catch the hang of runs 30 and 42 with a host-side watcher",
        ["on", "--ctas", "4", "--iters", "1", "--skip", "^(vision_|topk_topp)"]),
    # Run 43: the dry run hangs in run-iket's graph-launch callback, waiting on the GPU, about
    # every second time. Which part it needs: the mode's events at all (off), CUDA graphs
    # (eager), or the events in the eager kernels before the graph launches it stops at.
    "dry-skip-tp3-c1": dry_loop_spec(
        "the TP3 c1 dry run eight times, each killed after 300 s, with metadata_builder_kernel "
        "and padding_rows_kernel not instrumented (the eager kernels before each hang's graph "
        "launch)",
        ["on", "--ctas", "4", "--iters", "1", "--skip",
         "^(vision_|topk_topp|metadata_builder|padding_rows)"], timeout=300),
    "dry-off-tp3-c1": dry_loop_spec(
        "the TP3 c1 dry run eight times, each killed after 300 s, with the IKET mode off: "
        "run-iket's tracker over kernels without events",
        ["off"], timeout=300),
    "dry-eager-tp3-c1": dry_loop_spec(
        "the TP3 c1 dry run eight times, each killed after 600 s, without CUDA graphs",
        ["on", "--ctas", "4", "--iters", "1", "--skip", "^(vision_|topk_topp)", "--eager"]),
    "c8-nan": dict(
        ref=IKET,
        what="why c8 stops at boot (runs 33: route_partials' gate logits nan or inf in vLLM's sampler "
             "warmup, eight 7-token requests): off at TP4 rank shapes with the packed FP4 experts and "
             "their scales zeroed, then filled with 0x77, and c1 filled",
        env=STEP_ENV,
        mounts=[HUB],
        # Run 34: filled experts stop at boot the same way. Run 35: so do the whole model's
        # shapes at c8, and c4 and c3 boot but stop in the requests. So: eager, with a
        # hook on every module for the first output that is not finite.
        steps=[
            (f"off, TP4, c{requests}, eager, nan hooks",
             ["{python}", STEP, "off", "--tp-shape", "4", "--requests", str(requests), "--eager",
              "--nan-hooks", "--out", f"{{out}}/tp4-c{requests}-eager"], 1800)
            for requests in (3,)
        ],
    ),
    "tests": dict(
        ref=IKET,
        what="the IKET mode's tests alone",
        tests=TESTS,
        steps=[("IKET mode tests", [*PYTEST, "--basetemp", "{out}/pytest", *TESTS], 1800)],
    ),
    "inspect": dict(
        ref=IKET,
        what="the image's run-iket CLI sources, copied out for reading (how it injects into a "
             "process and its children, the dry run, the buffer sizing)",
        steps=[("copy iket.cli", ["{python}", "-c",
                                  "import iket.cli, pathlib, shutil, sys; "
                                  "src = pathlib.Path(iket.cli.__file__).parent; "
                                  "shutil.copytree(src, sys.argv[1], ignore=shutil.ignore_patterns('__pycache__'))",
                                  "{out}/iket-cli"], 300),
               ("copy iket.profiler's Python", ["{python}", "-c",
                                                "import iket.profiler, pathlib, shutil, sys; "
                                                "src = pathlib.Path(iket.profiler.__file__).parent; "
                                                "shutil.copytree(src, sys.argv[1], ignore=shutil.ignore_patterns("
                                                "'__pycache__', '*.so', 'perfetto_trace_pb2.py'))",
                                                "{out}/iket-profiler"], 300)],
    ),
    "step-bits": dict(
        ref=IKET,
        what="decode steps end to end at TP1 with uninitialized tensors filled, at 512-token "
             "chunks in a 1,024-token step: off twice, and on with no sampling, four CTAs a "
             "launch, and four CTAs and two iterations a loop; whether their tokens and top-5 "
             "logprobs agree exactly",
        env=STEP_ENV,
        mounts=[HUB],
        steps=[
            *(
                (f"decode steps, {label}, filled",
                 ["{python}", STEP, *mode, "--filled", "--batched", "1024", "--chunk", "512",
                  "--out", f"{{out}}/{name}"], 5400)
                for name, label, mode in (
                    ("off", "off", ["off"]),
                    ("off-again", "off again", ["off"]),
                    ("on", "on, unsampled", ["on", "--iket", "{out}/on/iket"]),
                    ("on-ctas", "on, CTAs sampled", ["on", "--iket", "{out}/on-ctas/iket", "--ctas", "4"]),
                    ("on-iters", "on, CTAs and iterations sampled",
                     ["on", "--iket", "{out}/on-iters/iket", "--ctas", "4", "--iters", "2"]),
                )
            ),
            ("tokens and logprobs agree", ["{python}", "-c",
                "import json, sys; names = ['off', 'off_again', 'on', 'on_ctas', 'on_iters']; "
                "r = dict(zip(names, (json.load(open(p)) for p in sys.argv[1:]))); "
                "same = {n: (r[n]['token_ids'], r[n]['logprobs']) == (r['off']['token_ids'], r['off']['logprobs']) "
                "for n in names[1:]}; "
                "print(json.dumps({'tokens': {n: r[n]['token_ids'] for n in names}, 'equal_to_off': same})); "
                "sys.exit(0 if all(same.values()) else 1)",
                *(f"{{out}}/{n}/result.json" for n in ("off", "off-again", "on", "on-ctas", "on-iters"))],
             60),
        ],
    ),
    "step-off": dict(
        ref=IKET,
        what="decode steps end to end at TP1 (six layers and the drafter, dummy weights, the "
             "production recipe's kernels and graphs) with the IKET mode off: does the engine run",
        env=STEP_ENV,
        mounts=[HUB],
        steps=[("decode steps, off", ["{python}", STEP, "off", "--out", "{out}/off"], 5400)],
    ),
}

RUN = '''#!/usr/bin/env python3
"""Run this bundle's steps in order; each step's exit status is printed, and the bundle
fails if any step failed. ``{python}`` is this interpreter and ``{out}`` the output root."""
import json
import os
import subprocess
import sys
import time

BUNDLE = os.environ["IKET_BUNDLE"]
OUT = f"/c/iket/{BUNDLE}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
os.makedirs(OUT)
print(f"=== output {OUT}", flush=True)
codes = []
for label, argv, timeout in json.load(open("/b/steps.json")):
    argv = [a.replace("{python}", sys.executable).replace("{out}", OUT) for a in argv]
    print(f"=== {label}", flush=True)
    started = time.monotonic()
    try:
        code = subprocess.run(argv, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    print(f"=== {label}: exit {code} in {time.monotonic() - started:.0f} s", flush=True)
    codes.append((label, code))
print(json.dumps({"output": OUT, "steps": codes}), flush=True)
sys.exit(0 if all(code == 0 for _, code in codes) else 1)
'''
SCRIPTS = {"iket_hot.py": HERE / "harness" / "iket_hot.py", "iket_step.py": HERE / "harness" / "iket_step.py",
           "sass_check.py": HERE / "harness" / "sass_check.py",
           "trace_counts.py": HERE / "harness" / "trace_counts.py", "iket_table.py": HERE / "iket_table.py",
           "nsys_window.py": HERE / "nsys_window.py", "iket_profile.py": HERE / "harness" / "iket_profile.py",
           "step_diff.py": HERE / "step_diff.py"}


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True, text=True).stdout


def show(repo, ref, path):
    return subprocess.run(["git", "-C", repo, "show", f"{ref}:{path}"], check=True,
                          capture_output=True).stdout


def write_bundle(repo, name, spec, bundle, image=IMAGE, base=BASE):
    """Write bundle ``name`` (a BUNDLES entry's shape) to the directory ``bundle``: the
    package files ``spec["ref"]`` changes from ``base``, the image's vLLM tree, as an
    overlay, this experiment's scripts, and the steps."""
    ref, tests = spec["ref"], spec.get("tests", [])
    head = git(repo, "rev-parse", "--short=9", ref).strip()
    status = [line.split("\t") for line in
              git(repo, "diff", "--no-renames", "--name-status", base, ref).splitlines() if line]
    package = [(kind, path) for kind, path in status if path.startswith("vllm/")]
    bundle.mkdir(parents=True, exist_ok=True)
    planned = {"run.py", "steps.json", "candidate.json", *SCRIPTS}
    planned |= {f"overlay/{path}" for _, path in package}
    planned |= {path.replace("/", "_") for path in tests}
    # Files are overwritten in place, never deleted: a file this run would not
    # write stops it, for a person to remove.
    stale = sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")
                   if p.is_file() and str(p.relative_to(bundle)) not in planned)
    if stale:
        raise SystemExit(f"{bundle} holds files this run would not write: {stale}")
    for kind, path in package:
        target = bundle / "overlay" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(MISSING.encode() if kind == "D" else show(repo, ref, path))
    for path in tests:
        # ``tests_ref``: a test from another ref, run against this one's tree.
        (bundle / path.replace("/", "_")).write_bytes(show(repo, spec.get("tests_ref", ref), path))
    for target, source in SCRIPTS.items():
        shutil.copy(source, bundle / target)
    (bundle / "run.py").write_text(RUN)
    (bundle / "steps.json").write_text(json.dumps(spec["steps"], indent=1) + "\n")
    candidate = {
        "description": f"{spec['what']} (vLLM {ref} {head}, {len(package)} package files over "
                       f"{base}) on {image}.",
        "image": image,
        "env": {
            "HF_HUB_OFFLINE": "1",
            "IKET_BUNDLE": name,
            "TILELANG_CACHE_DIR": f"/c/tilelang-iket-{name}",
            "TRITON_CACHE_DIR": f"/c/triton-iket-{name}",
            "VLLM_CACHE_ROOT": f"/c/vllm-iket-{name}",
            "PYTHONPATH": "/b:/opt/spark-ds41f/candidate/vllm",
            **spec.get("env", {}),
        },
        "workdir": "vllm",
        "argv": ["/b/run.py"],
        "verdict": "status",
        "mounts": [["run.py", "/b/run.py"], ["steps.json", "/b/steps.json"],
                   *([target, f"/b/{target}"] for target in SCRIPTS), NSYS, *spec.get("mounts", []),
                   *([path.replace("/", "_"), f"{ROOT}/{path}"] for path in tests)],
    }
    (bundle / "candidate.json").write_text(json.dumps(candidate, indent=1) + "\n")
    return f"{name}: {ref} {head}, {len(package)} package files, {len(spec['steps'])} steps"


def main():
    repo = sys.argv[1]
    only = sys.argv[sys.argv.index("--only") + 1:] if "--only" in sys.argv else []
    for name, spec in BUNDLES.items():
        if only and name not in only:
            continue
        print(write_bundle(repo, name, spec, HERE / "bundles" / name))


if __name__ == "__main__":
    main()
