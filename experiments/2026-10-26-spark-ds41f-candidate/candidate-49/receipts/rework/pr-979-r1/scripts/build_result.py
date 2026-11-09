import hashlib, json, sys
from pathlib import Path
R = Path(sys.argv[1])
a = json.loads((R / "analysis.json").read_text())
aa = json.loads((R / "aa.json").read_text())
patch = next(R.glob("0766-*.patch"))

def cell(topo, op, T):
    c = a["topologies"][topo]["ops"][op][str(T)]
    A = c["arms"]; k = f"{topo} {op} T={T}"
    out = {
        "corrected_max_rank_us": A["corrB"]["max_rank_medians_us"], "corrected_rank0_us": A["corrB"]["rank0_medians_us"],
        "baseline_max_rank_us": A["base0"]["max_rank_medians_us"], "baseline_rank0_us": A["base0"]["rank0_medians_us"],
        "baseline_aa_boots_3_4_max_rank_us": aa["cells"][k]["base0_boot3_4_max_rank_us"],
        "baseline_aa_boots_3_4_rank0_us": aa["cells"][k]["base0_boot3_4_rank0_us"],
        "original_979_max_rank_us": A["orig"]["max_rank_medians_us"], "original_979_rank0_us": A["orig"]["rank0_medians_us"],
        "paths": {"corrected": A["corrB"]["path"], "baseline": A["base0"]["path"], "original_979": A["orig"]["path"]},
        "corrected_vs_baseline": c["corrB_vs_base0"], "original_979_vs_baseline": c["orig_vs_base0"],
        "bit_exact_across_runs": c["bit_exact_across_runs"], "equal_to_in_process_anchor": c["equal_to_in_process_anchor"],
    }
    return out

mark = {f"{t}_T{T}": cell(t, "project_exchange_rows", T) for t in ("ring4", "direct") for T in (6144, 8192)}
miss1 = {f"direct_T{T}": cell("direct", "gather_mxfp8", T) for T in (160, 256, 512, 1024, 2048)}
miss2 = {f"{t}_T{T}": cell(t, "project_exchange_rows", T) for t, T in (("ring4", 160), ("ring4", 2048), ("ring4", 4096), ("direct", 4096))}
ring4_gather = {f"ring4_T{T}": dict(cell("ring4", "gather_mxfp8", T),
                                    ring4_gated_variant_corrA_max_rank_us=a["topologies"]["ring4"]["ops"]["gather_mxfp8"][str(T)]["arms"]["corrA"]["max_rank_medians_us"],
                                    ring4_gated_variant_corrA_rank0_us=a["topologies"]["ring4"]["ops"]["gather_mxfp8"][str(T)]["arms"]["corrA"]["rank0_medians_us"],
                                    corrA_vs_baseline=a["topologies"]["ring4"]["ops"]["gather_mxfp8"][str(T)]["corrA_vs_base0"])
                for T in (160, 256, 512, 1024)}
# every cell where the corrected arm runs different code from the baseline, and every 'loss' reading
differing, loss_cells = [], []
for topo, t in a["topologies"].items():
    for op, rows in t["ops"].items():
        for T, c in rows.items():
            v = c["corrB_vs_base0"]
            if not v["identical_code"]:
                differing.append(f"{topo} {op} T={T}: {v['change_pct']:+.1f}% max-rank {v['max_rank']}, rank0 {v['rank0_change_pct']:+.1f}% {v['rank0']}")
loss_cells = aa["corrB_loss_cells"]
all_exact = all(c["bit_exact_across_runs"] and c["equal_to_in_process_anchor"] for t in a["topologies"].values() for rows in t["ops"].values() for c in rows.values())
aa_digests = all(c["digests_equal_to_boot1"] for c in aa["cells"].values())

result = {
    "pr": 979,
    "round": 1,
    "candidate": 49,
    "pr_head": "c1d1bf3b76bc412aa4a87892726d107caa07bd6e",
    "original_patch": {"name": "0766-perf-dsv41-sp-prefill-sparknet-routing.patch", "sha256": "1a95ed84e8c0018d205b8a26ae236712160919cac861d2da420e675e91c7ce2f"},
    "corrected_patch": {"name": patch.name, "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
                        "author": "Christopher Owen <3221756+christopherowen@users.noreply.github.com> (original author and date kept)",
                        "sp_prefill_py_sha256": "cd10236f284af1ed43e1d7a00c74971935cc82864a4729a2925ac811de573f1a",
                        "applies": "on 0762..0765 in series order; 0767 applies cleanly after it; 0762..0765 + this + 0767 reproduces the measured overlay byte for byte (15 files)",
                        "git_diff_check": "clean"},
    "verdict": "corrected-meets-mark",
    "reason": ("The corrected 0766 keeps #979's mark: project_exchange_rows at T>=6144 is faster than the baseline (candidate-49 without #979) beyond the two-boot interval on both served topologies, slowest rank and rank 0: "
               "ring4 T=6144 -16.4% (rank0 -17.2%), T=8192 -18.3% (-17.2%); direct T=6144 -18.9% (-18.1%), T=8192 -14.3% (-15.2%); the original #979 measured -17.9/-18.7/-18.8/-15.0% in the same session. "
               "Outputs are bit-exact. Everywhere else it runs the baseline's code: gather_mxfp8 is 1107's grouped PyNCCL all-gather at every T on both topologies, and project_exchange_rows below 6,144 tokens is 1107's serial path. "
               "So the only cells where its code differs from the baseline are the four T>=6144 project_exchange_rows cells, all gains. "
               "The 13 cells where it reads slower than the baseline beyond the two-boot interval all run identical code, and the A/A shows that reading is the interval, not the code: the baseline against itself (boots 3-4 vs 1-2, same session) reads beyond the interval in 8 of 32 cells on the slowest rank and 8 of 32 on rank 0, and identical-code pairs across arms in 45 of 113 (slowest rank). "
               "Miss 1 is removed (direct gather_mxfp8 no longer takes the one-shot route; #979 reproduced its loss at T=2048: +46.3% slowest rank, +18.2% rank0) and Miss 2 is removed (no one-slice side-stream path)."),
    "gating_rule": {
        "rule": ("project_exchange_rows: under the one-shot RoCE policy, one slice (T < ENGRAM_SLICE_TOKENS = 6,144) takes 1107's path (full-row projection, then(), exchange_rows); two or more slices (T >= 6,144) take the PyNCCL copy_self=False side-stream pipeline (#979's gain). "
                 "gather_many / gather_mxfp8: 1107's grouped PyNCCL all-gather on every topology and T; the one-shot RoCE gather route is not taken anywhere."),
        "code": "sp_prefill.py: `if comm is None or (roce_rows and slices == 1):` -> serial 1107 path, where roce_rows = the TP group's one-shot communicator offers should_all_to_all_rows and comm = _tp_pynccl_comm() (now returned beside the one-shot one); gather_many reverted to 1107 byte for byte.",
        "how_chosen": ("Two corrected variants were measured as arms, with the choice pre-registered in raw/plan.txt before the first run: corrA kept #979's one-shot gather on sparknet ring4 only (gate: runtime.topology == 'ring4', T<=1024 at the 2 MB cap), corrB dropped it. "
                       "corrA was to be chosen only if, on ring4 at the routed T, both corrA and #979 (identical gather code) were faster than the baseline beyond the interval at >=1 T and neither slower at any T. "
                       "#979 read slower beyond the interval at ring4 T=512 (+17.1% slowest rank) and T=1024 (+59.5%, rank0 +98.6%), while corrA (same gather code) read faster at T=512 (-30.8%): no consistent win, so corrB."),
        "evidence": {
            "one_shot_gather_direct": "never faster than grouped PyNCCL beyond the interval on the slowest rank in either session: own mark +9..+50% (loss at T=160..2048); this session #979 vs baseline T=160 -7.4% within, 256 -8.8% within, 512 -11.0% within, 1024 -6.1% within (rank0 -25.5% 'gain', own mark -0.5% within), 2048 +46.3% loss (rank0 +18.2% loss)",
            "one_shot_gather_ring4": "own mark: -33%/-24% gain at T=160/256, within at 512/1024; this session #979 vs baseline: +6.6% within, +42.6% within, +17.1% loss, +59.5% loss (rank0 +98.6% loss) at T=160/256/512/1024; corrA (same code): +0.5% within, -13.5% within, -30.8% gain, +1.3% within. The own mark's ring4 gains did not repeat.",
            "one_slice_overlap": "#979's side-stream overlap of the one-slice exchange with then(): own mark loss at ring4 T=160/2048/4096 and gains at ring4 256/512, direct 1024/2048; this session #979 vs baseline loss at ring4 T=512/4096, direct 256/512/1024, gain at ring4 2048. No gain held across sessions; it can hide at most then(), one elementwise RankOrderParts sum.",
            "sliced_pipeline": "the T>=6144 code is the same in the corrected arm as in #979 and so is the gain: corrected vs #979 slowest-rank boots ring4 T=6144 [9914, 10061] vs [9762, 9868] us (+1.8%, reads loss), T=8192 [12630, 12753] vs [12608, 12626] (+0.6%, reads loss), direct T=6144 [8983, 8836] vs [9068, 8759] (within), T=8192 [11829, 11948] vs [11679, 11900] (within); the two ring4 readings are identical-code A/A pairs",
        },
        "served_shapes": "every shape in config/cluster.json uses a one-shot transport (oneshot-ring4 = ring4 for tp4-ring; oneshot-direct / oneshot-switched = direct for tp3-triangle and the switched shapes), so the rule covers serving on both topologies the bench measures.",
    },
    "key_cells": {
        "mark_project_exchange_rows_T_ge_6144": mark,
        "miss1_gather_mxfp8_direct_one_shot_route": miss1,
        "miss2_project_exchange_rows_below_6144": miss2,
        "ring4_gather_mxfp8_one_shot_route": ring4_gather,
    },
    "cells_where_corrected_code_differs_from_baseline": differing,
    "corrected_slower_cells_all_identical_code": {
        "count": len(loss_cells),
        "all_identical_code": all(c["identical_code"] for c in loss_cells),
        "cells": loss_cells,
        "why_not_regressions": ("In each of these cells the timed function is the same code in the corrected arm and the baseline (gather_mxfp8: 1107's gather_many byte for byte; project_exchange_rows below 6,144: the same project, then(), exchange_rows calls). "
                                "The baseline against itself reads beyond the two-boot interval in 25% of cells (aa.json), and the in-process anchor, identical bench code in every arm, moves with the boot: at ring4 gather_mxfp8 T=1024 the anchor's slowest-rank medians were 1,162/1,338 us in the corrected boots against 610/910/758/980 us in the four baseline boots, while integrated-minus-anchor was 58/1 us against 222/77/32/47 us. "
                                "The same identical-code readings hit the arms in both directions (e.g. corrected vs #979 at ring4 T=6144, identical sliced-pipeline code, reads 'loss'; corrA vs #979 at ring4 gather T=512/1024, identical one-shot code, reads 'gain')."),
    },
    "aa": {
        "baseline_vs_itself": {k: aa[k] for k in ("cells_total", "beyond_max_rank", "beyond_rank0", "false_positive_rate_max_rank", "false_positive_rate_rank0")},
        "identical_code_pairs_across_arms": {k: a["identical_code_aa"][k] for k in ("pairs", "beyond_max_rank", "beyond_rank0", "false_positive_rate_max_rank", "false_positive_rate_rank0")},
        "in_process_anchor_across_arms": {k: a["anchor_aa"][k] for k in ("pairs", "beyond_max_rank", "beyond_rank0")},
        "how": "base0 boots 3-4 ran right after the main sequence (12:34-12:37Z), same session, same code; compared with base0 boots 1-2 by the same two-boot rule (aa.json). Identical-code pairs: every pair of arms whose timed function takes the same path at a cell (analysis.json identical_code_aa).",
    },
    "bit_exact": all_exact and aa_digests,
    "bit_exact_detail": "At every T, op and topology, every rank's output digest (sha256 of the real rows of project_exchange_rows; of values, scales, weights and ids for gather_mxfp8) is equal across all 20 runs here (4 arms x 2 boots x 2 topologies, plus the 4 A/A boots) and the own mark's 8 runs, the same on ring4 and direct, and every rank equals its in-process anchor (then() output included).",
    "tests": {
        "host": "sp2 (spark-0010), under /tmp/spark-lab-gpu.lock, spark-ds41f:1107 with the corrected tree's 15 touched files bound read-only, tests/conftest.py with scripts/release_tests/tblib_stub.py",
        "corrected_gpu": "tests/models/test_deepseek_v41_sp_project_reduce_scatter.py tests/models/test_deepseek_v4_1_sp_prefill.py: 396 passed, 14 warnings (104.44 s)",
        "corrected_cpu": "tests/models/test_deepseek_v4_1_sp_prefill.py tests/models/test_deepseek_v4_1_engram.py, --gpus all with CUDA_VISIBLE_DEVICES empty: 219 passed, 17 skipped (55.25 s)",
        "corrA_variant": "GPU 396 passed; CPU 219 passed, 17 skipped",
        "same_counts_as_original": "the own mark's #979 counts were 396 GPU and 219 passed / 17 skipped CPU",
        "changed_tests": "#979's test_project_exchange_rows_cuda_with_active_roce_comm and test_project_exchange_rows_roce_and_pynccl_coexistence now assert the one-slice order (projection, then(), row exchange through exchange_rows; one-shot when it fits) and the two-slice PyNCCL pipeline; test_gather_many_routes_roce_and_packs_small_tensors is replaced by test_gather_many_keeps_grouped_pynccl_beside_roce; test_engram_slices and test_exchange_into_passes_send_and_recv_to_should_exchange are unchanged.",
    },
    "hosts": "sp1..sp4 (spark-0009..spark-0012), NVIDIA GB10, one rank per node, 200G dual-rail switched RoCE; tp4-switched profile of releases/spark-ds41f-1107/cluster.json (the own mark's rendered container command, only name/port/overlay/arm/label substituted)",
    "image": "spark-ds41f:1107 sha256:3f32a962ce04b7c8d85337cd427a8946b8b6bcaadd97cc8d8a8cd14e8d68a6ce on all four nodes, no build; arms bind 7 vLLM files read-only",
    "arms": {
        "corrected (corrB)": "0762..0765 + corrected 0766 + 0767; sp_prefill.py cd10236f284a...",
        "baseline (base0)": "candidate-49 without #979: 0762..0765 + 0767; sp_prefill.py 880ff323bde3..., identical to the own mark's no979 overlay",
        "original #979 (orig)": "0762..0767; sp_prefill.py 58e7f2579b73..., identical to the own mark's c49 overlay",
        "ring4-gated variant (corrA, not delivered)": "corrected + #979's one-shot gather on ring4 only; sp_prefill.py cf8b0302dd38...; variants/corrA-vs-corrected.diff",
    },
    "method": ("Same session, 12:20:06-12:34:10Z: per topology (ring4, then direct) corrA, corrB, base0, orig, orig, base0, corrB, corrA (mirrored ABBA), two boots per arm, each boot a fresh 4-container launch; then base0 boots 3-4 per topology as the A/A (12:34:33-12:37:44Z). "
               "Bench: the own mark's bench_979_integrated.py unchanged (sha256 b05cd7965f53...): the PR's workload, T=160..8192, 12 warmup + 40 timed iterations with a barrier before each, CUDA events, each rank >=200 ms of back-to-back 4096^3 BF16 GEMMs before timing (minimum measured 200.1 ms). "
               "Metric: median over the 40 iterations of the per-iteration max over the 4 ranks, and rank 0's median; interval = an arm's two boot medians; 'gain'/'loss' when two arms' intervals do not overlap."),
    "notes": [
        "Narrowness: against the baseline, the corrected 0766 changes only project_exchange_rows' branch condition (one slice under the one-shot policy -> 1107's path; two or more slices -> the existing PyNCCL pipeline), _tp_pynccl_comm() / _tp_roce_comm(), ENGRAM_SLICE_TOKENS 2048 -> 6144, and exchange_into's should_exchange(send, recv) (routing unchanged wherever 1107's call did not raise, since the one-shot exchange itself checks both). Against #979 it removes the one-shot gather route, the small-tensor packing, and the one-slice side-stream overlap (variants/corrected-vs-pr979.diff).",
        "ENGRAM_SLICE_TOKENS = 6144 also moves where a profile without the one-shot communicator (pure PyNCCL) starts slicing (2,048 -> 6,144). No served shape runs without it, and this bench does not measure that profile.",
        "With the one-shot communicator present but lacking should_all_to_all_rows (B12X PCIe all-reduce), project_exchange_rows keeps 1107's behaviour (pipeline at any slice count), because the one-slice condition tests should_all_to_all_rows as 1107's _tp_pynccl_comm() did.",
        "The bench's small-T cells are dominated by barrier skew and boot-to-boot drift (the slowest-rank medians at T=160 are 2-3x the per-rank minima). Two boots per arm cannot resolve differences there; the decision for those cells rests on the code being identical, shown by the A/A.",
        "Miss 1 reproduced at direct T=2048 only in this session (#979 +46.3% slowest rank, +18.2% rank0); at T=160..1024 #979's direct one-shot gather read within the interval here, never a gain on the slowest rank in either session.",
        "Every node was idle, without a hold or request file, with all four GPU locks held, before each run (run records in commands.txt). The first bench run waited on sp2's lock for this agent's own test job.",
        "The A/A appendix was added after the main runs were seen; it only re-runs the baseline arm, so it cannot favour the candidate.",
    ],
    "files": {
        patch.name: "the corrected patch, git format-patch, original author and date, rationale rewritten",
        "commands.txt": "setup commands, the pre-registered plan, every bench run's record (lock/start/end UTC, per-node image/load/docker ps/hold, exact per-rank command, exit codes) and every test run's record",
        "raw/": "each run's per-rank stdout+stderr (<label>-sp1..sp4.txt), rank 0's JSON (<label>.json, per-iteration times of every rank), sequence logs, driver logs, plan.txt, test outputs (test-*.txt), arm overlay hashes",
        "analysis.json": "scripts/analyze4.py over raw/ (all arms, all cells, identical-code A/A, anchor A/A, digests)",
        "aa.json": "scripts/aa.py: baseline boots 3-4 vs 1-2, and every corrected 'loss' cell against the A/A and against the baseline's four boots",
        "scripts/": "bench (unchanged), render.py + templates (the own mark's rendered commands), run4.sh, sequence.sh, runtest.sh, tests.sh, analyze4.py, show4.py, aa.py",
        "variants/": "corrected-vs-pr979.diff, corrA-ring4-gated-gather-vs-pre0766.diff (the measured, not delivered variant), corrA-vs-corrected.diff",
    },
}
(R / "result.json").write_text(json.dumps(result, indent=1) + "\n")
print(result["verdict"], result["bit_exact"], len(differing), len(loss_cells))
for d in differing: print(" ", d)
