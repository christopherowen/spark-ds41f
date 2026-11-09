# Assemble the selection receipts for candidates 51-53 from the sp8 outputs (sp8-out/), the PR scan
# (scan.json, prs.json) and the classification below. Run on the laptop from the scratch directory:
#   python3 -I scripts/receipts.py <scratch dir> <receipts dir>
import json, os, sys, hashlib, shutil, re

S, R = sys.argv[1], sys.argv[2]
os.makedirs(R, exist_ok=True)
O = f"{S}/sp8-out"
scan = json.load(open(f"{S}/scan.json"))
prs = {p["number"]: p for p in json.load(open(f"{S}/prs.json"))}
alone = json.load(open(f"{O}/alone.json"))
rebase = json.load(open(f"{O}/rebase-context.json"))
splits = json.load(open(f"{O}/splits.json"))
pairs = json.load(open(f"{O}/pairs.json"))
stack = json.load(open(f"{O}/stack.json"))

BASE = dict(
    description="candidate-50's vLLM series without #1009 (0778-perf-dsv41-sp-prefill-local-attention-front.patch): "
                "spark-ds41f-1107 (0001..0761) plus 0762..0777, 0779, 0780",
    how="sp8 copy of candidate-50's prepared tree (HEAD ba7400d9b, tree d5778ab5, the expected_tree of "
        "candidate-50/upstreams.lock.json), then git revert 0a231de28 (0778)",
    vllm_revision="19f2c20ed4d64ae22bd009ceacadd85e2a81e354",
    candidate_50_patch_head="ba7400d9b6715ea1be6fa98b1d9ccedf5fcb65f9",
    candidate_50_tree="d5778ab56647a20d94710ecb94cfd9a4de27bee9",
    base_commit=alone["base_commit"], base_tree=alone["base_tree"],
    check="the four files 0778 touches outside tests equal candidate-50/no1009/overlay byte for byte (SHA-256)",
    tilelang="2123e412 plus 1107/tilelang/series (13 patches): tree e2b001828fd4, the 1107 label",
    sparknet="1fd6a1fb (no series): tree 8cc28a5cd3e6, the 1107 label",
    where="sp8, ~/spark-lab/agents/round-51-53-select",
)

# ---------------------------------------------------------------- classification
HOUSEKEEPING = {
    938: ("docs: pool reservations, disk hygiene, inspiration watchlist", "owner: adds a second coordination file beside spark-hold.json and edits AGENTS.md; unchanged since the last triage"),
    949: ("DSML provenance proposal on the 1030 base (external)", "close as superseded by #955, which the base carries (0768); its patch conflicts with 0768 in vllm/parser/deepseek_v41.py, parser_engine.py, parser_engine_config.py and parser_manager.py"),
    958: ("TP3 6-channel NCCL profile and topology check", "review the scripts/topology.py change, then merge after the unit suite on a dgx node"),
    959: ("build: Git auto-GC off, batched git am in build checkouts", "merge after the unit suite on a dgx node (spark_ds41f/build.py)"),
    970: ("docs: 1107 is production", "merge; reconcile with the owner's uncommitted README.md and docs/current-state.md edits on the laptop checkout first (both files are modified there)"),
    974: ("CUDA 13.5.37 toolchain", "owner: its own track; it changes every kernel, so settle it before a kernel batch is measured; #994 (PTX 9.2), #1015 and #967 belong to the same track"),
    975: ("sequential quality stop rules (external)", "owner: wire it into methodology or hold"),
    976: ("restore #877 LSA evidence, 25.9 MB (with the clustered-indexer capture patches)", "owner: repository size call; the capture patches are study instrumentation, not candidate material"),
    985: ("GLP-27 spec-v3 default", "owner: changes the steered name's behaviour and records it in runtime-defaults; needs a quality eval"),
    1000: ("second-site check of 1107 (external)", "merge as evidence"),
    1007: ("TODO: two measured dead ends", "merge"),
    1017: ("second-site opt-in LSA on 1107 (external, README only; new since the last triage)", "merge as evidence. It reports VLLM_DS41_LSA=1 at +37-43% 1M prefill with its quality probes unchanged; whether to open an LSA-by-default experiment (with quality evals) is an owner call. Not candidate material: the recipe leaves LSA off"),
}
PARK = {
    954: ("LSA FP8 group32 summaries", "does nothing on this deployment: VLLM_DS41_LSA is opt-in and the recipe leaves it unset; conflicts with #1014; needs a context-only test rebase on the base"),
    1014: ("LSA block8 midrange blend", "does nothing on this deployment (opt-in LSA); conflicts with #954"),
    1004: ("DSpark top-k selection reuse", "does nothing on this deployment: DeepSeek V4.1 Flash's draft layers have no indexer; also conflicts with the base's #1012 in attention.py"),
    1002: ("online DSpark adaptation", "modelled gains only; session memory crosses requests; conflicts with #953, #971, #1003"),
    981: ("TP=2 NVMe expert slot cache", "does nothing on this deployment: TP=2 is not a served shape (a new topology needs qualification); conflicts with #1006 in moe.py"),
    983: ("Engram 2-bit Clock prefill shield, pinned scales", "evidence is a synthetic access pattern; its pinned-scales part does nothing here (opt-in SPARK_DS41F_ENGRAM_PIN_SCALES, two-plane tables; production reads packed rows); conflicts with #969 in _row_plan.h"),
    989: ("sparknet pipelined collectives study", "external repository study; applies on sparknet 1fd6a1fb; conflicts with #962 and #982"),
    982: ("sparknet PDL chain study", "external repository study; its sparknet patch and its vLLM test patch apply; conflicts with #962 and #989"),
    962: ("sparknet IKET/ROCE_TRACE", "tooling; applies on vLLM and sparknet; conflicts with #982 and #989 in sparknet"),
    992: ("producer MXFP8 bf16x2 casts", "writes root patches/vllm (0045 and the root series), not an experiment on the release series; noisy evidence. Restage it on the release series with a same-session A/B. Applies as a top patch; touches tilelang/gemm.py"),
    994: ("sparse MLA bf16x2 dequant", "needs CUDA 13.5 (PTX 9.2): the #974 track; writes root patches/; its vLLM patch conflicts with the base's #964 in sparse_mla.py and its test file"),
    1015: ("TileLang upstream audit", "pair it with the #974 toolchain track; its six TileLang patches apply on 1107's TileLang tree"),
}
CAT = {  # candidate material: category
    1030: "F", 995: "F/P", 960: "K1", 961: "K1", 963: "K1", 987: "K1", 1001: "K1", 957: "K1/P", 967: "K1",
    956: "K2", 984: "H", 998: "H", 969: "H", 978: "H", 1006: "H", 1008: "H",
    971: "P", 977: "P", 968: "P", 1003: "P", 953: "P", 988: "P",
}
TITLE = {n: prs[n]["title"] for n in prs}

CANDIDATES = {
    "51": dict(testbed="tp4-ring", nodes="dgx1..dgx4", shape="tp4-ring", theme="bit-identical TP4 decode kernels (gemm/vocab tiles at 97..128 rows, L2 prefetch pacing)"),
    "52": dict(testbed="tp4-switched", nodes="sp1..sp4 (head sp1)", shape="tp4-switched", theme="the routed-MoE rounding fix, the Engram reader and the max_tokens verify-row cap"),
    "53": dict(testbed="tp3-switched", nodes="sp5..sp7 (head sp5)", shape="tp3-switched", theme="every PR that edits model_runner.py (one candidate must hold them all), with the TP3-only display carve-out and the copy-drafter pair"),
}
MARKS = {
    961: dict(kind="perf", category="K1", own_mark="Its bench_l2_prefetch_ab.py (16 MiB cold prefetch, then the consumer read) on the candidate image on sp8, 2 boots, ABBA against the base image, with the recipe's VLLM_L2_PREFETCH_GRID=2: the consumer read is faster beyond the interval and the L2 hit share higher; its pytest passes. No decode cell of the engine grid worse", declared_output_change="none"),
    963: dict(kind="perf", category="K1", own_mark="Its screen_pack12_kernels.py on the image, 2 boots, ABBA against the base image, by kernel time and CUDA-graph replay: packed BF16 GEMM (compressor_up, M=1..256) and packed vocab projection/greedy partials faster beyond the interval, 0 ULP, including through #997's certified screen (in the base)", declared_output_change="none"),
    987: dict(kind="perf", category="K1", own_mark="WO-B part only. Its WO-B bench (mxfp8_gemm and TileLangWOProjection at M=104..128, K=3072/2048/1024) on the image, 2 boots, ABBA: faster beyond the interval at M=97..128, torch.equal against the prefill GEMM; the engine grid's 16-stream cell not worse. It acts only at TP4 (rows past 96 need 16 streams with copy widening: 107 rows at max_num_seqs 16; TP3's 8 streams reach 59)", declared_output_change="none"),
    1001: dict(kind="perf", category="K1", own_mark="Its bench_vocab_2boot.py (captured CUDA graphs) on the image, 2 boots, ABBA: projection and greedy epilogue at M=97..128 (the 128-row tile) faster beyond the interval, bit-identical logits and (max, index) pairs, M<=96 not worse. TP4 only (as #987)", declared_output_change="none"),
    1030: dict(kind="fix", category="F", own_mark="Its bench_reference_alignment.py on the image, 2 boots: routed moe_swiglu_cast FP8 bytes and UE8M0 scales identical to DeepSeek's Expert.forward reference (0 diffs at TP4 inter=576 and TP3 inter=768), and the end-to-end MoE closer to the reference than 1107's; kernel time not worse beyond the interval at the decode, mid and prefill tiers. It is a precision change: the determinism gate fails by construction, so the gate checks #968 and #969 on tp4-switched-no1030.json (1107's moe.py mounted; no other patch here touches it), speed is judged on distinct prompts, and its correctness evidence is the reference alignment plus release qualification's quality evals", declared_output_change="temperature-0 text changes on every shape: routed MoE activations are rounded to BF16 before MXFP8 quantization, as in DeepSeek's reference"),
    968: dict(kind="perf", category="P", own_mark="Its screen_remaining_token_cap.py on the image, 2 boots: verify rows past a request's remaining max_tokens are not scheduled (functional), outputs bit-exact; paired decode not worse; the tail steps of bounded requests at 16 streams lighter", declared_output_change="none"),
    969: dict(kind="perf", category="H", own_mark="Its Engram row-cache bench (run_cache_and_packed_benchmark) on the image, 2 boots, ABBA against the base image in the same session (the last triage parked it for comparing with a recorded number): cold-miss and Zipfian lookups faster beyond the interval; the engram-row tests pass; rows identical", declared_output_change="none"),
    960: dict(kind="perf", category="K1", own_mark="Its screen_cudagraph_wins.py on the image, 2 boots, ABBA against the base: per-step pre-launch metadata and CED staging on FULL decode steps lower beyond the interval (the PR: 438-451 us to 33-41 us from (1,1) to (16,64)), bit-exact; the engine grid and paired decode not worse. Its receipts predate its head commit (6cfd0bd3 dropped the post-gate MoE fork), so they must be re-run on the current patch", declared_output_change="none"),
    971: dict(kind="perf (policy)", category="P", own_mark="Its screen_copy_drafter_smooth.py 58-layer step screen on the image, 2 boots: the single-stream initial copy step and the post-reject shrunk step faster beyond the interval at TP3; paired decode at 1 stream better and at 8 streams not worse; temperature-0 text unchanged", declared_output_change="drafting and verification rows (copy continuation yield, single-request 16-row cap, wide-step padding mask); temperature-0 text must not change; sampled realizations may change"),
    977: dict(kind="perf (policy)", category="P", own_mark="Its bench_copy_match_long_context.py on the image, 2 boots: copy_match at 64K..1M faster beyond the interval, unchanged at <=32K (still a dense scan); copy drafts identical at <=32K", declared_output_change="copy drafts at contexts past 32K (the bounded scan can miss a far match); temperature-0 text must not change; sampled realizations may change"),
    984: dict(kind="perf (memory)", category="H", own_mark="Its bench_uma_allocator_hygiene.py on the image, 2 boots, and the screen's startup record: each rank's lowest MemAvailable at startup higher than the base's beyond the interval, time to ready not worse, no JIT after readiness; short-request TTFT during a long prefill not worse (its post-prefill cache flush runs on the serving path); KV capacity unchanged (the recipe fixes --kv-cache-memory-bytes)", declared_output_change="none"),
    998: dict(kind="perf (memory)", category="H", own_mark="Its bench_carveout_expansion.py on the image, 2 boots (the PR commits no receipts: run it fresh): regular UMA saved per rank (the PR: 1,263 MiB at TP3), the moved DSpark MoE layers not slower (the PR: 8.5-12.3% faster), torch.equal; the startup floor higher. It acts only where SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1, the TP3 shapes", declared_output_change="none"),
}
ORDER = {"51": ["961", "963", "987-wob", "1001"], "52": ["1030", "968", "969"], "53": ["960", "971", "977", "984", "998"]}
PLACEMENT = {
    961: "every shape prefetches; with the gemm/vocab tile PRs it makes 51 a bit-identical TP4 decode batch on production's shape, so 51's determinism gate stays a pure identity check",
    963: "shares tilelang/gemm.py with #987 and tilelang/vocab.py with #1001, so the three share a candidate; its large-M gains are TP4's 16-stream rows",
    987: "acts only at TP4 (97..128 rows); measured on sp4",
    1001: "acts only at TP4 (97..128 rows); shares vocab.py with #963",
    1030: "the only fix and the only output change: kept apart from 51's bit-identical batch; TP4 is where its mid tier gains (1.25x); its no1030 overlay is possible because no other patch in 52 touches moe.py",
    968: "scheduler.py only; touches no other candidate's file",
    969: "Engram reader (engram_on_disk at every shape); touches no other candidate's file",
    960: "edits model_runner.py, which #971, #984 and #998 edit too: candidates overlap at most in test files, so all four share one candidate, and #998 acts only at TP3",
    971: "edits model_runner.py (see #960); its step screen was measured at TP3; the most promising of the conflicting policy PRs (#953, #971, #1003, #1002, #995) for a TP3 testbed",
    977: "shares tilelang_copy.py with #971 (they apply together), so it joins 53 or waits; it conflicts with no policy PR here",
    984: "edits model_runner.py, cudagraph_utils.py with #960; TP3's tighter memory shows its startup-floor gain best",
    998: "TP3-only (SPARK_DS41F_DISPLAY_CARVEOUT_WEIGHTS=1 on tp3-triangle and tp3-switched, 0 on tp4-ring and tp4-switched)",
}
LEFT_OUT = [
    dict(pr=987, patch="0762-perf-dsv41-extend-WO-B-WIDE_DECODE-to-128-rows-and-g.patch", part="vllm/models/deepseek_v4_1/sp_prefill.py and tests/models/test_deepseek_v4_1_sp_prefill.py (resolve_min_rows)", reason="does nothing on this deployment: at max_num_seqs <= 16 the SP floor stays at the 205-row RoCE limit; it acts only past 32 streams", disposition="left out at selection (file-level split)"),
    dict(pr=987, patch="same", part="vllm/v1/worker/gpu/cudagraph_utils.py and its test in tests/v1/cudagraph/test_cudagraph_manager.py (_FULL_DEPTH_VARLEN_DECODE_MAX_REQS = 16)", reason="does nothing on this deployment: the PR states zero change to captured graphs at max_num_seqs <= 16, and the recipe serves 16 (TP4) and 8 (TP3)", disposition="left out at selection (file-level split)"),
    dict(pr=987, patch="same", part="vllm/models/deepseek_v4_1/tilelang/collectives.py and vllm/distributed/device_communicators/cuda_communicator.py (all_reduce_rank_order through RoCE reduce-scatter + all-gather)", reason="not inert: it moves CED decoder steps of 205..408 rows from the PyNCCL rank-order path to RoCE, a second causal variable unrelated to the WO-B mark", disposition="split out; re-submit as its own experiment (prefill collectives, tp4-switched or tp3)"),
    dict(pr=988, patch="0762-perf-spec-decode-sampler-dead-row-skip-and-sparse-topk.patch", part="sparse top-k Markov sampling: topk_* kernels in tilelang_sampling.py, _sample_sequential_topk in dspark/speculator.py, apply_markov_bias_topk in nvidia/dspark.py, apply_bias_topk in qwen3_dspark.py's DSparkMarkovHead, tests/v1/spec_decode/test_dspark_topk.py", reason="does nothing on this deployment: it runs only when dspark_draft_topk is set, and neither the recipe nor the 1017 checkpoint's config sets it (drafts are greedy); Qwen3DSparkForCausalLM.apply_markov_bias_topk is another model", disposition="PR queued: hunk-level split needed before it can join"),
    dict(pr=995, patch="0762-perf-spec-decode-copy-drafter-guards-and-wide-ramp.patch", part="SPARK_DS41F_COPY_MAX_DRAFTS (vllm/config/speculative.py), greedy_rejection_kernel widths above 32 (tilelang_sampling.py), benchmark --lookahead, test_wide_lookahead_31_and_63_greedy_verification_and_ramp", reason="does nothing on this deployment: the 31/63-draft ramp is opt-in; the recipe keeps max_drafts at 16 (17 rows), where the kernel keeps its one-pass code", disposition="PR queued: hunk-level split needed"),
    dict(pr=1006, patch="0762-perf-dsv41-vectorize-postload-add-artifact-cache-and.patch", part="vllm/models/deepseek_v4_1/tilelang/postload_cache.py and its hooks (artifact cache)", reason="does nothing on this deployment: it reads or writes only when VLLM_POSTLOAD_CACHE_DIR or SPARK_DS41F_POSTLOAD_CACHE_DIR is set; the recipe sets neither", disposition="PR queued: hunk-level split into its variables needed"),
    dict(pr=1006, patch="same", part="vllm/models/deepseek_v41/nvidia/vl_model.py", reason="does nothing on GB10: vllm/models/deepseek_v41 routes capability family 120 to vllm/models/deepseek_v4_1; this file is the other-NVIDIA path", disposition="PR queued (file-level)"),
    dict(pr=983, patch="0762-perf-dsv41-engram-clock-pro-prefill-shield-and-pinned-scales.patch", part="pin_scales (rows.py and the two-plane reader path)", reason="does nothing on this deployment: opt-in SPARK_DS41F_ENGRAM_PIN_SCALES for two-plane tables; production reads packed rows", disposition="PR parked"),
    dict(pr=957, patch="0762-perf-dsv41-tp3-decode-compressor-mla-and-spec-fusions.patch", part="part 3, fused post-step verification bookkeeping (acceptance_estimator.py, speculator.py, model_runner.py)", reason="superseded: the base's #965 and #980 already fuse the dead-row cut and copy state into _accumulate_kernel; these hunks conflict with them", disposition="PR queued for rework"),
    dict(pr=954, patch="whole PR", part="", reason="does nothing on this deployment (opt-in LSA)", disposition="parked"),
    dict(pr=1014, patch="whole PR", part="", reason="does nothing on this deployment (opt-in LSA)", disposition="parked"),
    dict(pr=981, patch="whole PR", part="", reason="does nothing on this deployment (TP=2 is not served)", disposition="parked"),
    dict(pr=1004, patch="whole PR", part="", reason="does nothing on this deployment (the draft layers have no indexer)", disposition="parked"),
]
NOT_INERT_NOTES = [
    "#960's vllm/v1/worker/gpu/spec_decode/dflash/speculator.py hunk acts: DSparkSpeculator subclasses DFlashSpeculator and inherits propose() and uses_target_attn_metadata",
    "#998's and #988's vllm/model_executor/models/qwen3_dspark.py: DSparkMarkovHead and DSparkConfidenceHead there are DeepSeek V4.1's DSpark heads (vllm/models/deepseek_v4_1/nvidia/dspark.py imports them); #998's markov_w1 change acts, #988's top-k additions do not",
    "#961's vllm/models/glm5next/nvidia/l2_prefetch.py and tilelang/l2_prefetch.py: DeepSeek V4.1's prefetch (vllm/models/deepseek_v4_1/l2_prefetch.py) imports both; the CuTe DSL kernel class does nothing under VLLM_L2_PREFETCH_KERNELS=tilelang and the _GRID default change (4 to 1 on sm_121) does nothing because the recipe sets VLLM_L2_PREFETCH_GRID=2, but the file holds the shared constants, so it is carried whole",
    "#1003's _sample_sequential_topk halt does nothing (dspark_draft_topk unset); the PR is queued anyway",
    "#971's wide-step padding mask relies on VLLM_MOE_SKIP_PADDING, which defaults to 1",
    "CopyDrafter is always on in the recipe (DSpark, greedy drafts, adaptive verification), so #968, #971, #977 and #995 act",
]
QUEUED = [
    dict(pr=1003, category="P", why="conflicts with #971 (adaptive_verification.py and its test), #953, #988 and #1002; edits model_runner.py, which 53 holds", next="a TP4 candidate next round, on the base with #971's verdict (rebase there); its c1 +6% is modelled (identical numbers on both boots)", rebase="context-only (test append) on today's base"),
    dict(pr=953, category="P", why="conflicts with #971, #1003 and #1002; its mark is the TP6..TP8 switched prose c16 regression, and no testbed this round runs TP6..TP8; edits adaptive_verification.py, which 53 holds", next="a round with tp6..tp8-switched (sp1..sp8)", rebase="context-only (test append) on today's base"),
    dict(pr=995, category="F/P", why="its negative-slot guard (copy_alloc_kernel and copy_apply_kernel index total_len/int_state/float_state/history with slot < 0) is a fix; its 31/63 ramp does nothing here; conflicts with #971 and #977 in tilelang_copy.py", next="rework: split the guard (F, first next round, rebased onto #971/#977 if they pass), the cap decay and anchor pruning (P), and leave out the ramp", rebase="applies alone on today's base; hunk-level split needed"),
    dict(pr=988, category="P", why="the sparse top-k half does nothing here; conflicts with #1003 (dspark/speculator.py); edits qwen3_dspark.py, which 53's #998 edits", next="rework: keep the dead-row verification sampler skip (acts at temperature > 0; the recipe's default temperature is 1.0)", rebase="applies alone; hunk-level split needed"),
    dict(pr=957, category="K1/P", why="five variables; part 3 is superseded by the base's #965/#980 and its hunks conflict with them (model_runner.py, acceptance_estimator.py, speculator.py, test_adaptive_verification.py); edits model_runner.py and compressor.py", next="rework into: replicated BF16 compressor at TP<=4; sparse-MLA branchless stage 2; DSpark confidence-calibration fusion with the (4, 24) dummy-run fix; TP3 adaptive constants (P). The TP3 parts to tp3-switched", rebase="code conflict with the base"),
    dict(pr=956, category="K2", why="code conflict with the base's #1012 in attention.py (the decode-overlap branch); edits ced.py and nvidia/model_state.py, which 53's #960 edits", next="rebase onto the next base (with #960's verdict); tp4-switched, where its served prefill screen was taken", rebase="code conflict with the base"),
    dict(pr=978, category="H", why="code conflict with the base's #972 in vllm/v1/engine/async_llm.py (abort shielding); edits scheduler.py, which 52's #968 edits", next="rebase onto #972; then a candidate with #968's verdict", rebase="code conflict with the base"),
    dict(pr=1006, category="H", why="five variables in one patch; the artifact cache does nothing by default; conflicts with #998 (packed_bf16.py, embedding.py) and #981; overlaps #963 (vocab.py, gemm.py) and #1030 (moe.py)", next="rework into its variables, leaving out the cache and deepseek_v41/nvidia/vl_model.py", rebase="applies alone; split needed"),
    dict(pr=1008, category="H", why="needs a sparknet patch series in the lock (1107 pins sparknet with none); conflicts with #984 in gpu_worker.py; overlaps #969 in engram_rows/_reader.c", next="a round where #984 and #969 have verdicts; its sparknet patch applies on 1fd6a1fb", rebase="applies alone (vLLM and sparknet)"),
    dict(pr=967, category="K1", why="a compiler-wide TileLang pass change (needs a TileLang series entry and a TileLang rebuild; every kernel recompiles)", next="the #974/#1015 toolchain track; its TileLang patch applies on 1107's TileLang tree", rebase="context-only (test append) for its vLLM patch"),
]

# ---------------------------------------------------------------- pr-patch-targets.json
targets = {}
for n in sorted(int(k) for k in scan):
    o = scan[str(n)]; t = {}
    for e in o["patches"]:
        t.setdefault(e["component"], set()).update(e["targets"])
    targets[str(n)] = dict(title=o["title"], author=o["author"], head_ref=prs[n]["headRefName"], head=o["head"],
                           behind_main=o["behind_main"],
                           patches=[dict(path=e["path"], component=e["component"], commits=e["commits"],
                                         sha256=hashlib.sha256(open(e["local"]).read().encode()).hexdigest()) for e in o["patches"]],
                           targets={k: sorted(v) for k, v in t.items()},
                           other_files=[f["path"] for f in prs[n]["files"] if not f["path"].endswith(".patch")])
json.dump(targets, open(f"{R}/pr-patch-targets.json", "w"), indent=1)

# ---------------------------------------------------------------- pairwise-conflicts.json
res = alone["results"]
applies = {}
for n, r in res.items():
    if r["plain"] == "ok": st = "applies"
    elif n in rebase and rebase[n]["result"] == "context-only": st = "context-only rebase (test append at the same spot as a base patch)"
    else: st = "code conflict with the base"
    applies[n] = dict(status=st, patches=r["patches"], plain_apply_failed=r["plain_fail"], three_way=r["three_way"], three_way_conflicts=r["conflicts"])
    if n in rebase: applies[n]["rebase"] = {k: rebase[n][k] for k in ("result", "resolved_test_files", "sha256")}
other = open(f"{O}/other-upstreams.txt").read().replace("/home/chris/spark-lab/agents/round-51-53-select/", "")
other_pairs = open(f"{S}/sp8-out/other-pairs.txt").read() if os.path.exists(f"{S}/sp8-out/other-pairs.txt") else ""
conf = [dict(a=p["a"], b=p["b"], files=sorted(set(sum([v for k, v in p.items() if "_after_" in k and v != "clean"], []))),
             **{k: v for k, v in p.items() if "_after_" in k}) for p in pairs["pairs"] if p["conflict"] and {p["a"], p["b"]} != {"987", "987-wob"}]
clean = [dict(a=p["a"], b=p["b"], common=p["common"]) for p in pairs["pairs"] if not p["conflict"]]
json.dump(dict(
    base=BASE,
    method="each PR's vLLM patches alone on the base (git apply --index; on failure git apply -3 to tell a context-only test append from a code conflict). "
           "Context-only rebases resolve each test-file conflict as the base's tests, then the PR's, and are verified to add and remove exactly the PR's lines. "
           "Then, for every pair of the resulting patch sets that touch a common file: A applied, B checked with git apply --check, both orders. "
           "TileLang and sparknet patches on 1107's pinned trees the same way",
    applies_alone=applies,
    not_paired=sorted([n for n, a in applies.items() if a["status"] == "code conflict with the base"], key=int),
    sets_paired=pairs["labels"],
    overlapping_pairs=pairs["overlapping_pairs"] - 1,
    conflicting_pairs=conf,
    clean_overlapping_pairs=clean,
    note="987-wob is #987's WO-B part (a file-level split); its pair with #987 itself is not counted. #960 conflicts with the whole #987 only in tests/v1/cudagraph/test_cudagraph_manager.py, which 987-wob leaves out",
    tilelang_and_sparknet=dict(alone=other.strip().splitlines(), pairs=other_pairs.strip().splitlines()),
), open(f"{R}/pairwise-conflicts.json", "w"), indent=1)

# ---------------------------------------------------------------- selection-proposal.json
cands = {}
for c, meta in CANDIDATES.items():
    st = stack["candidates"][c]; rows = []
    for row in st["patches"]:
        n = row["pr"]
        src = next(e for e in scan[str(n)]["patches"] if e["component"] == "vllm")
        rows.append(dict(pr=n, title=TITLE[n], pr_head=scan[str(n)]["head"], source_patch=src["path"], source_sha256=row["source_sha256"],
                         method=row["method"], proposed_patch=("rebases/" + f"{c}-" + row["file"]) if row["method"] != "git am" else src["path"],
                         proposed_sha256=row["sha256"], **MARKS[n], placement=PLACEMENT[n]))
    cands[c] = dict(**meta, order=[r["pr"] for r in rows], prs=rows, stack_tree=st["stack_tree"], stack_head_on_sp8=st["stack_head"])
proposal = dict(
    round="candidates 51, 52 and 53 (PR validation, steps 1-8 of docs/release-benchmark.md)", date="2026-11-09",
    base=BASE, candidates=cands,
    union=dict(applies=stack["candidates"]["union"]["applies"], tree=stack["candidates"]["union"]["tree"],
               note="all three stacks on one tree in the order 51, 52, 53; the three candidates touch no common file, test files included"),
    rules_checked=[
        "no conflicting pair inside a candidate (pairwise-conflicts.json)",
        "the candidates share no file at all (the rule allows test files)",
        "one spec-decode policy PR of each conflicting group per candidate: #971 in 53; #968 (52) and #977 (53) conflict with no policy PR they share a candidate with",
        "only patches that act on DeepSeek V4.1 Flash on GB10 (left_out)",
    ],
    left_out=LEFT_OUT, not_inert_notes=NOT_INERT_NOTES, queued=QUEUED,
    rebase_needed=[dict(pr=int(n), kind=applies[n]["status"], files=applies[n]["three_way_conflicts"]) for n in sorted(applies, key=int) if applies[n]["status"] != "applies"],
    housekeeping=[dict(pr=n, what=w, action=a) for n, (w, a) in sorted(HOUSEKEEPING.items())],
    park=[dict(pr=n, what=w, action=a) for n, (w, a) in sorted(PARK.items())],
    decided_not_in_pool=dict(passed_in_base=[966, 997, 965, 1012, 1016, 964, 955, 980, 1013, 972, 1011, 973, 999, 993, 991, 979, 996], in_rework=[1009], dropped=[986, 1005, 1010], process=[1038, 1040]),
    attribution_notes=[
        "51: #961's two l2_prefetch files are touched by no other patch, so a tp4-no961 overlay control is possible; #963, #987 and #1001 share gemm.py/vocab.py, so none of them has one",
        "52: #1030 (moe.py), #968 (scheduler.py) and #969 (engram_rows/_reader.c and _row_plan.h, compiled at runtime by _native.py) each touch files no other patch in 52 touches, so all three have overlay controls; the determinism gate for #968/#969 runs on the no1030 control",
        "53: #960, #971, #984 and #998 all edit model_runner.py and #977 shares tilelang_copy.py with #971, so no PR in 53 has a pure-Python overlay control; attribution there needs per-PR image variants. #984 and #998 both move the startup floor: their own benches separate them",
    ],
    for_assembly="Patch files, in order, are the PR heads' patches (method git am) or rebases/<candidate>-<file> (context-only rebase or split). The coordinator writes vllm/series, selection.json (with left_out carried over) and the configurations.",
)
json.dump(proposal, open(f"{R}/selection-proposal.json", "w"), indent=1)

# ---------------------------------------------------------------- rebased patches and their record
os.makedirs(f"{R}/rebases", exist_ok=True)
rb = {}
for c in ORDER:
    for row in stack["candidates"][c]["patches"]:
        if row["method"] == "git am": continue
        dst = f"{R}/rebases/{c}-{row['file']}"; shutil.copy(f"{O}/out/{c}/{row['file']}", dst)
        rb[f"{c}-{row['file']}"] = dict(pr=row["pr"], label=row["label"], method=row["method"], source_patch=row["source_patch"],
                                        source_sha256=row["source_sha256"], sha256=row["sha256"],
                                        verification=(rebase.get(str(row["pr"]), {}).get("result") if row["label"] in rebase else "file-level split of the context-only rebase"),
                                        kept_files=splits.get(row["label"], {}).get("kept_files"))
json.dump(dict(note="Context-only: each conflicting hunk is a test appended where a base patch appended one; the union keeps the base's tests, then the PR's, two blank lines apart. Every added and removed line equals the PR's (rebase-context check). The split keeps three files of #987's rebased patch unchanged and adds one line to the message saying so.",
               context_only_rebases_checked={k: {kk: v[kk] for kk in ("result", "resolved_test_files", "sha256")} for k, v in rebase.items()},
               files=rb), open(f"{R}/rebases/rebases.json", "w"), indent=1)
print("written", sorted(os.listdir(R)))
