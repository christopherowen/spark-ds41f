# Assemble the selection receipts for candidates 55-57 from the sp8 outputs (sp8/), the PR scan (scan.json,
# prs.json, authors.tsv) and the classification below. Run on the laptop from the scratch directory:
#   python3 -I scripts/receipts.py <scratch dir> <receipts dir>
import json, os, sys, hashlib, shutil, re, glob, subprocess

S, R = sys.argv[1], sys.argv[2]
O = f"{S}/sp8"
os.makedirs(f"{R}/rebases", exist_ok=True); os.makedirs(f"{R}/scripts-as-run/outputs", exist_ok=True)
scan = json.load(open(f"{S}/scan.json"))
prs = {p["number"]: p for p in json.load(open(f"{S}/prs.json"))}
alone = json.load(open(f"{O}/alone.json")); splits = json.load(open(f"{O}/splits.json"))
ctx = json.load(open(f"{O}/rebase-context.json")); pairs = json.load(open(f"{O}/pairs.json"))
stack = json.load(open(f"{O}/stack.json")); files = json.load(open(f"{S}/files.json"))
shared = json.load(open(f"{O}/base-shared-files.json"))
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
def src(n): return sorted(glob.glob(f"{S}/v/{n}/vllm/*.patch"))[0]

BASE = dict(
    description="candidate-54 (merged base of round 2): spark-ds41f-1107 (0001..0761) plus candidate-50's 0762..0780 "
                "(#1009 round 1 as 0778) and round 2's 0781..0791; 671 patches",
    how="built on sp8 from candidate-54/vllm/series (origin/main 920e1d50): vLLM 19f2c20ed + git am of the 671 active "
        "patches with the fixed committer and --committer-date-is-author-date (scripts-as-run/build_base.sh)",
    vllm_revision="19f2c20ed4d64ae22bd009ceacadd85e2a81e354",
    patch_head=alone["base_commit"], tree=alone["base_tree"],
    check="patch_head 4e8133ff and tree 5f021d9a equal candidate-54/upstreams.lock.json's patch_head and expected_tree; "
          "git diff --check clean",
    tilelang="2123e412 plus 1107/tilelang/series (13 patches): tip 3b0c7814, tree e2b001828fd4 (candidate-54's lock "
             "pins TileLang as 1107)",
    sparknet="1fd6a1fb (no series): tree 8cc28a5cd3e6 (as 1107)",
    where="sp8, ~/spark-lab/agents/round-55-57-select (CPU only; sp8's GPU was not touched)",
)

# ------------------------------------------------------------------ the pool and the sets derived from it
POOL = [995, 1003, 953, 988, 957, 956, 978, 1006, 1008, 967]
SETS = {  # label -> (pr, how, part)
    "1008": (1008, "as published (git am)", "whole PR (vLLM 0762); sparknet 0001 rides in a sparknet series"),
    "978": (978, "context-only rebase", "whole PR"),
    "988-deadrow": (988, "hunk-level split", "part 1: dead verification rows skip the sampler's logits processing"),
    "988-topk": (988, "hunk-level split", "part 2: sparse top-k DSpark draft sampling (left out: inert)"),
    "957-p1": (957, "hunk-level split", "part 1: replicated BF16 compressor fused_wkv_wgate at TP<=4"),
    "957-p2": (957, "hunk-level split", "part 2: branchless sparse MLA stage-2 online softmax"),
    "957-p3": (957, "hunk-level split (does not apply)", "part 3: fused post-step verification bookkeeping (left out: superseded)"),
    "957-p4": (957, "hunk-level split", "part 4: fused DSpark confidence calibration, preallocated Markov bias scratch, (4, 24) dummy-run fix"),
    "957-p5": (957, "hunk-level split (does not apply)", "part 5: TP3 adaptive-verification constants"),
    "967": (967, "context-only rebase", "vLLM half (TileLang 0014 applies on 1107's TileLang tree)"),
}

# ------------------------------------------------------------------ pr-patch-targets.json
targets = {}
for n in POOL + [1005, 1047]:
    p = prs[n]; s = scan.get(str(n))
    ent = dict(title=p["title"], author=p["author"]["login"], head_ref=p["headRefName"], head=p["headRefOid"],
               created=p["createdAt"], updated=p["updatedAt"])
    if s:
        ent["behind_main"] = s["behind_main"]
        ent["patches"] = [dict(path=e["path"], component=e["component"], commits=e["commits"], targets=e["targets"],
                               sha256=sha(e["local"]),
                               from_line=re.search(r"^From: (.*)$", open(e["local"]).read(), re.M).group(1))
                          for e in s["patches"]]
    else:
        ent["patches"] = []
    ent["other_files"] = sorted(f["path"] for f in p["files"] if not f["path"].endswith(".patch"))
    targets[str(n)] = ent
targets["1047"]["patches"] = [dict(
    path="experiments/2026-10-26-spark-ds41f-candidate/1107/vllm/0762-fix-dsv41-escape-the-literal-image-token-in-text.patch",
    component="vllm", commits=1, sha256=sha(f"{S}/ext/1047/0762-fix-dsv41-escape-the-literal-image-token-in-text.patch"),
    from_line="Jordi Posthumus <jordi.posthumus@gmail.com>",
    targets=["tests/tokenizers_/test_deepseek_v41.py", "vllm/tokenizers/deepseek_v41.py", "vllm/tokenizers/deepseek_v41_encoding.py"],
    on_candidate_54="plain apply fails; git apply -3 leaves vllm/tokenizers/deepseek_v41.py conflicted (the base's 0774, #973, fast-path chat rendering)")]
targets["1005"]["note"] = ("dropped after round-1 rework (candidate-47/receipts/rework/pr-1005-r1); two commits pushed since: "
                           "c42ad695 (2026-11-09 15:00Z) and 8071740d (2026-11-10 00:19Z). Not in this pool; not scanned")
json.dump(dict(date="2026-11-10", note="Every pool PR's patches at its fetched head (git fetch origin pull/<N>/head), with "
               "SHA-256, the From: line and the files each patch targets. Heads equal the ones candidate-51's triage recorded.",
               prs=targets), open(f"{R}/pr-patch-targets.json", "w"), indent=1)

# ------------------------------------------------------------------ rebases/
rows = {}
for cand, c in stack["candidates"].items():
    if cand == "union": continue
    for r in c["patches"]:
        name = f"{cand}-{r['file']}"
        shutil.copy(f"{O}/out/{cand}/{r['file']}", f"{R}/rebases/{name}")
        assert sha(f"{R}/rebases/{name}") == r["sha256"]
        rows[name] = dict(pr=r["pr"], label=r["label"], method=r["method"], source_patch=r["source_patch"],
                          source_sha256=r["source_sha256"], sha256=r["sha256"])
        if r["label"] in splits:
            rows[name]["kept"] = splits[r["label"]]["kept"]
            rows[name]["verification"] = ("hunk-level split: the kept hunks applied plainly on the base and the result adds "
                                          "and removes exactly the kept hunks' lines (lines_exact)")
            assert splits[r["label"]]["lines_exact"]
        elif r["label"] in ctx:
            rows[name]["verification"] = "context-only (rebase-context check: every added and removed line equals the PR's)"
PROPOSED = {"978": f"{O}/out/55/02-perf-multimodal-prefix-cache-fastpath-under-mm-proce.patch"}
for lab in ("978", "967"):
    c = ctx[lab]; before = src(lab)
    shutil.copy(f"{O}/rebased/{lab}.diff", f"{R}/rebases/{lab}-context-only.diff")
    after = PROPOSED.get(lab, f"{O}/rebased/{lab}.diff")  # the proposed format-patch keeps the PR's message
    d = subprocess.run(["diff", "-u", "--label", f"PR #{lab} patch (before)", "--label", f"rebased on candidate-54 (after)",
                        before, after], capture_output=True, text=True).stdout
    open(f"{R}/rebases/{lab}-context-only.interdiff", "w").write(d)
json.dump(dict(
    note="The proposed patches, as format-patch of each candidate's stack on the candidate-54 base (sp8). A PR that applies "
         "as published is its own file byte for byte. A hunk-level split keeps whole hunks of the PR's patch, every kept line "
         "unchanged, and adds a paragraph to the PR's message naming what it keeps. The two context-only rebases (#978, "
         "proposed; #967, not proposed) are recorded with their before and after SHA-256, the rebased diff "
         "(<N>-context-only.diff) and the diff between the PR's patch and it (<N>-context-only.interdiff).",
    context_only_rebases={lab: dict(before=os.path.relpath(src(lab), S), before_sha256=ctx[lab]["source_sha256"],
                                    after=f"{lab}-context-only.diff", after_sha256=ctx[lab]["sha256"],
                                    resolved=ctx[lab]["resolved"], lines_check=dict(
                                        extra_added=ctx[lab]["extra_added"], missing_added=ctx[lab]["missing_added"],
                                        extra_removed=ctx[lab]["extra_removed"], missing_removed=ctx[lab]["missing_removed"]))
                          for lab in ("978", "967")},
    context_only_resolution={
        "978": "vllm/v1/engine/async_llm.py, AsyncLLM.abort: against the merge base the PR inserts two lines (pop each aborted "
               "request's _pending_mm_retries entry) before the `await self.engine_core.abort_requests_async(...)` line that "
               "the base's #972 (0771) replaced with its shielded abort block. The rebase puts the PR's two lines before #972's "
               "block. No line of either side is edited; the pops are synchronous, so the shield still covers the only await",
        "967": "tests/kernels/attention/test_deepseek_v41_tilelang.py: a test appended where a base patch appended one; base, then PR"},
    files=rows), open(f"{R}/rebases/rebases.json", "w"), indent=1)

# ------------------------------------------------------------------ pairwise-conflicts.json
semantic = {k: v for k, v in ctx.items() if v["result"] == "semantic"}
json.dump(dict(
    base=BASE,
    method="each pool PR's vLLM patch alone on the base (git apply --index; on failure git apply -3). A conflict whose only "
           "kind is a test appended where the base appended one, or a PR insertion next to a region the base changed, is "
           "rebased context-only (rebase_context.py: no line of either side edited; added and removed lines equal the "
           "PR's); a conflict where the PR edits a line the base also edits is semantic and goes to rework. Splits keep "
           "whole hunks (split_hunks.py). Then, for every pair of applicable sets touching a common file: A applied, B "
           "checked with git apply --check, both orders (pairs.py). TileLang and sparknet patches on 1107's pinned trees",
    applies_alone={n: dict(plain=r["plain"], plain_fail=r["plain_fail"], three_way=r["three_way"], three_way_conflicts=r["conflicts"],
                           classification=("applies as published" if r["plain"] == "ok" else
                                           ctx[n]["result"] + " rebase" if n in ctx and ctx[n]["result"] == "context-only" else
                                           "semantic conflict with the base: rework"))
                   for n, r in alone["results"].items()},
    semantic_conflicts={k: dict(refused_files=v["refused"], regions=v["kinds"]) for k, v in semantic.items()},
    base_patches_in_the_conflicts={
        "953": "0789 (#971 round 1): _step_cost (num_sampling_requests, SINGLE_REQ_TILE_SPILL_SCALE) and the budget code in get_num_tokens",
        "1003": "0789 (#971 round 1): _step_cost",
        "957-p5": "0789 (#971 round 1): the constants block and _step_cost's scale",
        "957-p3": "0764 (#965 round 1) already carries the same observe_verification fusion and verified-count reuse in model_runner.py; 0769 (#980 round 1) in speculator.py",
        "995": "0790 (#977): copy_match's two-tier long-context scan (the anchor-pruning hunks); copy_alloc and copy_apply hunks apply 3-way-clean (split 995-alloc-apply)",
        "956": "0765 (#1012 round 1) decode overlap and 0778 (#1009 round 1) sp_front_call in attention.py; 0788 (#960 round 1) CED pinned ring in ced.py",
        "1006": "0791 (#998 round 1): the display_carveout import beside pack_bf16_weight in packed_bf16.py and embedding.py; the PR's lines there add cache_tag (the artifact cache)",
    },
    splits={k: v for k, v in splits.items()},
    sets_paired=pairs["labels"], overlapping_pairs=pairs["overlapping_pairs"], conflicting_pairs=pairs["conflicting_pairs"],
    pairs=pairs["pairs"],
    file_overlaps_with_pending_reworks={
        "note": "the PRs that need a rework cannot be pair-checked until the rework exists; these are the files they share "
                "with the proposed sets (from their current patches). Each reworked patch is pair-checked before assembly",
        "overlaps": [
            {"a": "957-p1", "b": "956", "common": ["vllm/models/deepseek_v4_1/compressor.py"], "consequence": "same candidate (56); #956's rework builds on 957-p1"},
            {"a": "957-p1", "b": "1006", "common": ["vllm/models/deepseek_v4_1/tilelang/linear.py"], "consequence": "#1006 queued"},
            {"a": "978", "b": "953", "common": ["vllm/v1/engine/core.py"], "consequence": "#953 queued"},
            {"a": "1008", "b": "953", "common": ["vllm/v1/engine/core.py"], "consequence": "#953 queued"},
            {"a": "957-p4", "b": "1003", "common": ["vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu/spec_decode/dspark/speculator.py"], "consequence": "#1003 queued"},
            {"a": "957-p2", "b": "967", "common": ["vllm/models/deepseek_v4_1/tilelang/sparse_mla.py"], "consequence": "#967 not proposed (toolchain track); clean both orders"},
            {"a": "957-p5", "b": "1003", "common": ["vllm/v1/worker/gpu/spec_decode/adaptive_verification.py", "tests/v1/spec_decode/test_adaptive_verification.py"], "consequence": "conflicting spec-decode policy: one per candidate"},
            {"a": "957-p5", "b": "953", "common": ["vllm/v1/worker/gpu/spec_decode/adaptive_verification.py", "tests/v1/spec_decode/test_adaptive_verification.py"], "consequence": "conflicting spec-decode policy: one per candidate"},
            {"a": "1003", "b": "953", "common": ["vllm/v1/worker/gpu/spec_decode/adaptive_verification.py", "tests/v1/spec_decode/test_adaptive_verification.py"], "consequence": "conflicting spec-decode policy"},
            {"a": "956", "b": "1006", "common": ["vllm/models/deepseek_v4_1/attention.py", "vllm/models/deepseek_v4_1/nvidia/model.py"], "consequence": "#1006 queued"},
        ]},
    shared_with_base_patches=shared,
    tilelang_and_sparknet=open(f"{O}/other-upstreams.txt").read().splitlines(),
    files_by_set=files,
), open(f"{R}/pairwise-conflicts.json", "w"), indent=1)

# ------------------------------------------------------------------ copies of the scripts and raw outputs
for f in glob.glob(f"{S}/scripts/*"):
    if os.path.basename(f) != "README.md": shutil.copy(f, f"{R}/scripts-as-run/")
for f in ("alone.json", "splits.json", "rebase-context.json", "pairs.json", "stack.json", "conflict-regions.json",
          "other-upstreams.txt", "base-shared-files.json"):
    shutil.copy(f"{O}/{f}", f"{R}/scripts-as-run/outputs/{f}")
shutil.copy(f"{O}/stack.log", f"{R}/stack.log")
shutil.copy(f"{S}/authors.tsv", f"{R}/scripts-as-run/outputs/authors.tsv")
print("wrote", R)
