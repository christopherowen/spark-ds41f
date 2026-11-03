#!/usr/bin/env python3
"""Print a Compute Sanitizer investigation plan; never execute its commands.

Paths name files inside the eventual GPU execution environment. They need not
exist on the machine producing this plan. The serving command is an argv list:
pass it last, after --serving-command, without a shell command string.
"""

import argparse
import json
from pathlib import Path
import re


SOURCE_COMMIT = "5b7cfa37fee42f96bfafbe09b7d30e8c0eda8575"
SOURCE_TREE = "af0fb1525fc6aa4149b18dee7d8c3ae5120b6f33"
ERROR_EXITCODE = 86

# SHA-256 of exact committed bytes (git show SOURCE_COMMIT:path), usable without .git.
SOURCE_SHA256 = {
    "vllm/v1/worker/gpu/spec_decode/copy_drafts.py": "caed58e8c641cae78626fb8dab34758573bac407fc092915691bd9ac6e9e1cc8",
    "vllm/v1/worker/gpu/spec_decode/tilelang_copy.py": "89cd4cf39ad75b8992504afd5bc7d91a088dda0c64a66d314126553a4379187d",
    "vllm/v1/worker/gpu/model_runner.py": "55841d2d94c4540f05421a4d8a6d1d25cbbd6eafe0d439fd81e94eafdb94c8cc",
    "vllm/v1/worker/gpu/input_batch.py": "1ab6eccef4d4d3de1cf0cab4df4503d396bc5029e50c63ecb7637db898a8c339",
    "vllm/v1/worker/gpu/tilelang_inputs.py": "be6c93c468ca14dc645353f7a0069ef8f464b7dee356fb035e7abd269c3626f7",
    "vllm/models/deepseek_v4_1/tilelang/router.py": "eb10a557044a4a222cfc53abfef42c72ea78134fc6eb2dd5fc410471820452e3",
    "vllm/models/deepseek_v4_1/tilelang/gemm.py": "0a456a01acd9aa38b88fd996d982610730e8effd7b510f8778003f85575ffed8",
    "vllm/models/deepseek_v4_1/tilelang/mhc.py": "c6fbcff5b6ce3af8fa3147b9265f7c79612eb839a35d4a1b46a0a4339949c7e4",
    "vllm/models/deepseek_v4_1/tilelang/linear.py": "203db0f8c55d08313ecdd8a9036fc17898d269ed66ea92f2cca24ef220fb5436",
    "vllm/models/deepseek_v4_1/tilelang/indexer.py": "857d27bfcd209120dd4a4a0869643487efed7232f17c70a981c2d701383b2ada",
    "vllm/models/deepseek_v4_1/tilelang/pdl.py": "38d9f90bc1031a95b8f5b4fe11d34f5bef4b0ee7444bb61e34c7a8c68d2bde85",
    "tests/v1/spec_decode/test_copy_drafts.py": "c253edd11175d372a4978027372538796b12731cde16b883fcbfb5b04dff13f4",
    "tests/v1/worker/test_dead_rows.py": "097aa80482a040f5753d6d9d6d88b09dfbbb658aa7702bc2c3882f2680c49abc",
    "tests/kernels/moe/test_deepseek_v41_tilelang_router.py": "b71be28f93c2a577bd23c901009846c836acd720954195194cab2c922e67decf",
    "tests/kernels/test_deepseek_v41_tilelang_pdl.py": "7e843ab60b8d25bc2666be4f3c0ff8ceedcec76d075a24b6b49a6ddc5c796c69",
    "tests/kernels/attention/deepseek_v41_indexer_reads.py": "ac58983956910a8c6b4a6f6f3ba86379a59f7e4027bf33fb7e3d7feccf115fbb",
}

# These selectors exist at SOURCE_COMMIT. Deliberately broken PDL readers and
# nonfinite-input tests are not part of the clean-run sweep.
MICROTESTS = (
    (
        "copy-state",
        [
            "tests/v1/spec_decode/test_copy_drafts.py::test_copy_select_follows_the_reference_over_a_copy",
            "tests/v1/spec_decode/test_copy_drafts.py::test_copy_select_judges_only_its_pending_copy",
            "tests/v1/spec_decode/test_copy_drafts.py::test_kernels_replay_in_a_cuda_graph",
            "tests/v1/spec_decode/test_copy_drafts.py::test_a_slot_starts_afresh_with_each_request",
            "tests/v1/spec_decode/test_copy_drafts.py::test_prepare_alloc_matches_propose_and_updates_slot_draft_limits",
            "tests/v1/spec_decode/test_copy_drafts.py::test_stale_wide_allocation_rejects_placeholder_tail_and_doubles_cap_without_sync",
        ],
        "Copy proposal growth, rejection, slot reuse, stale-wide allocation and graph replay.",
        "Verification uses fabricated target logits; no neural target or DSpark forward.",
    ),
    (
        "dead-rows",
        [
            "tests/v1/worker/test_dead_rows.py::test_dead_rows_mark_rows_past_the_survival_cut",
            "tests/v1/worker/test_dead_rows.py::test_dead_row_cut_clamps_committed_tokens",
        ],
        "Real input-combine padding mask and committed-token cutoff.",
        "Does not connect CopyDrafter state to attention, caches or a model forward.",
    ),
    (
        "gate-chain",
        [
            "tests/kernels/moe/test_deepseek_v41_tilelang_router.py::test_gate_router_matches_the_gate_then_the_router",
            "tests/kernels/moe/test_deepseek_v41_tilelang_router.py::test_padding_rows_are_not_routed",
            "tests/kernels/test_deepseek_v41_tilelang_pdl.py::test_bf16_partials_edges_are_programmatic",
        ],
        "Actual gate projection/router for target and drafter shapes; padding; mHC-to-gate graph edges.",
        "Random finite inputs and weights, no failing-step replay or distributed transport.",
    ),
    (
        "indexer-visible-reads",
        ["tests/kernels/attention/deepseek_v41_indexer_reads.py"],
        "Actual indexer reads with only visible cache bytes initialized, dense and candidate modes.",
        "Indexer only; does not execute the full attention/model chain.",
    ),
)


def absolute_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("use an absolute path in the GPU execution environment")
    return str(path)


def image_id(value):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise argparse.ArgumentTypeError("use the immutable sha256 image ID, not an image tag")
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sanitizer-path", type=absolute_path, required=True)
    result.add_argument("--source-root", type=absolute_path, required=True)
    result.add_argument("--report-dir", type=absolute_path, required=True)
    result.add_argument("--image-id", type=image_id,
                        help="immutable image ID for the exported-source identity route")
    result.add_argument("--python", default="python3", help="Python executable in that environment")
    result.add_argument("--sanitizer-version", default="2025.3.1.0")
    result.add_argument("--cuda-toolkit-version", default="13.0")
    result.add_argument("--serving-command", nargs=argparse.REMAINDER,
                        help="optional direct serving-process argv; must be the last option")
    return result


def make_plan(args):
    serving = args.serving_command
    if serving is not None and serving[:1] == ["--"]:
        serving = serving[1:]
    if serving is not None and not serving:
        raise ValueError("--serving-command needs an executable and its arguments")

    cases = []
    for name, selectors, coverage, limitation in MICROTESTS:
        runner = [args.python]
        if name != "indexer-visible-reads":
            runner += ["-m", "pytest", "-q", "-rA"]
        cases.append((name, runner + selectors, coverage, limitation))
    if serving:
        cases.append((
            "serving-repro", serving,
            "The operator-supplied serving process and its local child processes.",
            "Remote workers require their own instrumented launch. Reproduction workload and rank setup must be recorded separately.",
        ))

    jobs = []
    # Entire memcheck pass precedes initcheck. No command is launched here.
    for tool in ("memcheck", "initcheck"):
        for name, command, coverage, limitation in cases:
            job_id = f"{tool}-{name}"
            jobs.append({
                "id": job_id,
                "tool": tool,
                "requires": [] if tool == "memcheck" else [f"memcheck-{name}: clean completed run"],
                "cwd": args.source_root,
                "argv": [
                    args.sanitizer_path, "--tool", tool,
                    "--target-processes", "all", "--error-exitcode", str(ERROR_EXITCODE),
                    "--log-file", str(Path(args.report_dir) / f"{job_id}.%p.log"),
                    *command,
                ],
                "env_set": {"PYTORCH_NO_CUDA_MEMORY_CACHING": "1"},
                "env_unset": ["CUDA_LAUNCH_BLOCKING"],
                "coverage": coverage,
                "limitation": limitation,
                "retain": "Sanitizer per-process logs, complete stdout/stderr, exit status, pytest executed/skipped counts and elapsed time.",
            })

    return {
        "plan_only": True,
        "commands_executed": False,
        "source": {
            "candidate": "candidate-37",
            "expected_vllm_commit": SOURCE_COMMIT,
            "expected_vllm_tree": SOURCE_TREE,
            "root": args.source_root,
            "identity_status": "Not checked by this planner.",
            "identity_routes": {
                "git_checkout": {
                    "applicable_when": "The execution source retains its Git metadata.",
                    "checks_to_record": [
                        ["git", "-C", args.source_root, "rev-parse", "HEAD"],
                        ["git", "-C", args.source_root, "status", "--porcelain"],
                    ],
                    "requirement": "Match expected commit and record a clean checkout or separately identified diagnostic overlay.",
                },
                "exported_image": {
                    "applicable_when": "The execution source was exported without .git; Git commands are unnecessary.",
                    "image_id": args.image_id,
                    "expected_labels": {
                        "local.spark-ds41f.vllm.patch-head": SOURCE_COMMIT,
                        "local.spark-ds41f.vllm.tree": SOURCE_TREE,
                    },
                    "inspect_location": "Docker host, not necessarily inside the GPU container.",
                    "inspect_argv": [
                        "docker", "image", "inspect", "--format",
                        '{"id":{{json .Id}},"layers":{{json .RootFS.Layers}},"labels":{{json .Config.Labels}}}',
                        args.image_id,
                    ] if args.image_id else None,
                    "requirement": "Record the actual container's immutable .Image ID and matching build labels, plus the source hashes below. Supply --image-id to produce its inspect argv; a mutable tag is insufficient.",
                },
            },
            "focused_source_hashes": {
                "cwd": args.source_root,
                "argv": ["sha256sum", "--", *SOURCE_SHA256],
                "expected_sha256": SOURCE_SHA256,
                "scope": "Selected runtime files and microtest entrypoints, not a complete source-tree or dependency attestation.",
                "requirement": "Compare every listed file before execution. Record any mounted diagnostic overlay and its hashes separately; do not describe modified sources as an unchanged c37 run.",
            },
            "requirement": "Use the applicable identity route and focused source hashes; an exported image does not require .git. Missing identity evidence remains unresolved.",
        },
        "toolchain": {
            "compute_sanitizer_version": args.sanitizer_version,
            "cuda_toolkit_version": args.cuda_toolkit_version,
            "version_status": "Declared versions, not probed by this planner; retain actual output before execution.",
            "version_argv": [[args.sanitizer_path, "--version"], ["nvcc", "--version"]],
        },
        "execution_requirements": [
            "No authorization to run or change the cluster is supplied by this plan; use the approved coordinated window and launch path.",
            "Prepare the report directory in the execution environment; verify SM120/SM121 hardware, candidate dependencies and tool versions.",
            "Run memcheck first. Investigate every reported error before running initcheck on that case; do not add broad suppressions or skip filters.",
            "Do not enable force-blocking-launches or CUDA_LAUNCH_BLOCKING. Preserve graph/PDL settings and record all inherited runtime settings.",
            "Disabling allocator caching and sanitizer instrumentation change timing. A passing run cannot exclude a timing-dependent failure.",
            "A clean result requires exit 0, zero sanitizer errors, and actual test execution. Skips or a missing reproducer are not passing coverage.",
            "These microtests can isolate memory faults but cannot establish the reported serving crash's cause without a matching failure artifact.",
        ],
        "positive_controls": {
            "requirement": "Validate each tool on the same GPU/toolchain before trusting a clean report; controls run separately from the clean-run sweep.",
            "memcheck": "Use a reviewed CUDA fixture with an intentional out-of-bounds global read; require its expected diagnostic and exit 86.",
            "initcheck": "Use a reviewed CUDA fixture with an intentional uninitialized global read; require its expected diagnostic and exit 86.",
            "arithmetic_recorder": "Inject a nonfinite value into a designated live row and a masked row; require exact model/layer/row attribution and preservation of original input/partials before continuation.",
        },
        "interpretation": {
            "memcheck": "Checks invalid/misaligned memory accesses; a clean run does not prove numerical correctness.",
            "initcheck": "Checks uninitialized reads; stale but initialized values and arithmetic overflow need other diagnostics.",
            "next_evidence": "Preserve the first nonfinite producer's input/output, model/layer/rank, row identity, masks and exact shapes. Replay that boundary on one GPU; retain multi-rank reproduction for transport/order hypotheses.",
        },
        "serving_repro": "Included from operator argv." if serving else "Not supplied; no serving reproduction is covered by this plan.",
        "jobs": jobs,
    }


def main():
    cli = parser()
    args = cli.parse_args()
    try:
        plan = make_plan(args)
    except ValueError as exc:
        cli.error(str(exc))
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
