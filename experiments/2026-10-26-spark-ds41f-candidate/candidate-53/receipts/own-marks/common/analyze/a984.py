"""#984 analysis: bench-<arm>-boot<k>.json (the PR's lifecycle bench, --mode both, run on both
images, ABBA), flush-c53-boot*.json, pytest-c53-boot*.txt."""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(__file__)); from common import cell
d = sys.argv[1]
R = {l: json.load(open(f"{d}/bench-{l}.json")) for l in ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2")}
M = [("stage1_after_load", "vm_rss_mib", False), ("stage1_after_load", "anon_mmap_rss_mib", False), ("stage1_after_load", "mem_avail_mib", True),
     ("stage2_after_profile", "vm_rss_mib", False), ("stage2_after_profile", "peak_reserved_during_attn_mib", False),
     ("stage2_after_profile", "non_kv_cache_memory_mib", False), ("stage2_after_profile", "mem_avail_mib", True),
     ("stage3_startup_complete", "vm_rss_mib", False), ("stage3_startup_complete", "torch_reserved_mib", False),
     ("stage3_startup_complete", "torch_slack_mib", False), ("stage3_startup_complete", "mem_avail_mib", True), ("stage3_startup_complete", "cuda_free_mib", True),
     ("stage4_after_prefill_transition", "vm_rss_mib", False), ("stage4_after_prefill_transition", "torch_reserved_mib", False),
     ("stage4_after_prefill_transition", "torch_slack_mib", False), ("stage4_after_prefill_transition", "mem_avail_mib", True),
     ("stage4_after_prefill_transition", "cuda_free_mib", True), ("stage4_steady_state_decode", "p50_us", False)]
out = {"in_process_deltas_candidate_minus_baseline": {}, "cross_image_c53_candidate_mode_vs_base_baseline_mode": {},
       "cross_image_candidate_mode_base_vs_c53": {}, "chunk_reuse": {}}
for l in R:
    out["in_process_deltas_candidate_minus_baseline"][l] = {f"{s}.{m}": R[l]["deltas_candidate_minus_baseline"][s].get(m) for s, m, _ in M}
    out["chunk_reuse"][l] = {mode: R[l][mode]["stage4_after_prefill_transition"]["chunk1_to_chunk2_reused"] for mode in ("baseline", "candidate")}
for s, m, hi in M:
    b = [R[l]["baseline"][s][m] for l in ("base-boot1", "base-boot2")]
    c = [R[l]["candidate"][s][m] for l in ("c53-boot1", "c53-boot2")]
    out["cross_image_c53_candidate_mode_vs_base_baseline_mode"][f"{s}.{m}"] = cell(b, c, higher=hi)
    b2 = [R[l]["candidate"][s][m] for l in ("base-boot1", "base-boot2")]
    out["cross_image_candidate_mode_base_vs_c53"][f"{s}.{m}"] = cell(b2, c, higher=hi)
fl = {}
for k in (1, 2):
    p = f"{d}/flush-c53-boot{k}.json"
    if os.path.exists(p):
        fl[f"boot{k}"] = [(c["live_unfrozen_objects"], c["flush_host_ms_median"], c["slack_mib_before"][-1], c["slack_mib_after"][-1]) for c in json.load(open(p))["cases"]]
out["post_prefill_flush_host_ms"] = fl
pt = {}
for k in (1, 2):
    p = f"{d}/pytest-c53-boot{k}.txt"
    if os.path.exists(p):
        last = [l for l in open(p).read().splitlines() if re.search(r"(passed|failed|error)", l) and "==" in l]
        pt[f"boot{k}"] = last[-1] if last else None
out["pytest"] = pt
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
for l, v in out["in_process_deltas_candidate_minus_baseline"].items(): print(l, json.dumps(v))
for k, v in out["cross_image_c53_candidate_mode_vs_base_baseline_mode"].items(): print("X", k, v["base"], v["candidate"], v["delta"], v["verdict"])
for k, v in out["cross_image_candidate_mode_base_vs_c53"].items(): print("Y", k, v["base"], v["candidate"], v["delta"], v["verdict"])
print(json.dumps(out["chunk_reuse"]), json.dumps(fl), json.dumps(pt))
