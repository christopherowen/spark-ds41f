"""#960 analysis: ab-*.jsonl (image A/B) and script-c53-boot*.json (the PR's in-process screen)."""
import json, sys, os, statistics
sys.path.insert(0, os.path.dirname(__file__)); from common import cell, jl
d = sys.argv[1]
ab = {lab: jl(f"{d}/ab-{lab}.jsonl") for lab in ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2")}
out = {"serving_path_image_ab": {}, "in_process_candidate_image": {}, "bit_exact": {}, "pr_script": {}}
def get(lab, nr, nt, path):
    for r in ab[lab]:
        if r.get("kind") == "step" and r["num_reqs"] == nr and r["num_tokens"] == nt and r["path"] == path:
            return r
shapes = sorted({(r["num_reqs"], r["num_tokens"]) for r in ab["base-boot1"] if r.get("kind") == "step"})
for nr, nt in shapes:
    k = f"({nr},{nt})"
    B = [get(l, nr, nt, "eager") for l in ("base-boot1", "base-boot2")]
    C = [get(l, nr, nt, "graph") for l in ("c53-boot1", "c53-boot2")]
    CF = [get(l, nr, nt, "graph_floors_change") for l in ("c53-boot1", "c53-boot2")]
    CE = [get(l, nr, nt, "eager") for l in ("c53-boot1", "c53-boot2")]
    out["serving_path_image_ab"][k] = {
        "wall_us": cell([x["wall_us_median"] for x in B], [x["wall_us_median"] for x in C],
                        samples_b=sum((x["wall_us_windows"] for x in B), []), samples_c=sum((x["wall_us_windows"] for x in C), [])),
        "wall_us_floors_change_every_step": cell([x["wall_us_median"] for x in B], [x["wall_us_median"] for x in CF]),
        "device_us": cell([x["device_us_median"] for x in B], [x["device_us_median"] for x in C]),
        "device_us_floors_change_every_step": cell([x["device_us_median"] for x in B], [x["device_us_median"] for x in CF]),
        "host_enqueue_us_max": {"base": [x["host_enqueue_us_max"] for x in B], "candidate": [x["host_enqueue_us_max"] for x in C]},
    }
    out["in_process_candidate_image"][k] = {
        "eager_wall_us_base_image_vs_candidate_image": cell([x["wall_us_median"] for x in B], [x["wall_us_median"] for x in CE]),
        "eager_pr_wall_us_candidate_image": [get(l, nr, nt, "eager_pr")["wall_us_median"] for l in ("c53-boot1", "c53-boot2")],
    }
    be = {}
    for l in ab:
        for r in ab[l]:
            if r.get("kind") == "bit_exact" and r["num_reqs"] == nr and r["num_tokens"] == nt:
                be[l] = r["digests"]
    out["bit_exact"][k] = {"digests": be,
                           "graph_equals_eager_in_candidate": all(be[l].get("graph") == be[l]["eager"] for l in ("c53-boot1", "c53-boot2")),
                           "candidate_graph_equals_base_eager_across_images": len({be[l].get("graph") for l in ("c53-boot1", "c53-boot2")} | {be[l]["eager"] for l in ("base-boot1", "base-boot2")}) == 1}
for k in (1, 2):
    p = f"{d}/script-c53-boot{k}.json"
    if os.path.exists(p):
        s = json.load(open(p))
        out["pr_script"][f"boot{k}"] = {"target_full": [(r["num_reqs"], r["num_tokens"], r["baseline_us"], r["candidate_us"], r["bit_exact"], r["mem_delta_bytes"]) for r in s["target_full_metadata_and_ced_replay"]],
                                        "dflash_padding_rows": [(r["live_tokens"], r["padded_tokens"], r["baseline_us"], r["candidate_us"], r["bit_exact"]) for r in s["dflash_padding_rows_dedup"]],
                                        "moe_post_gate_fork_bit_exact_all": all(r["bit_exact"] for r in s["moe_post_gate_fork"])}
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
for k, v in out["serving_path_image_ab"].items():
    print(k, "wall", v["wall_us"]["base"], "->", v["wall_us"]["candidate"], v["wall_us"]["verdict"], "| device", v["device_us"]["base"], "->", v["device_us"]["candidate"], v["device_us"]["verdict"], "| fc wall", v["wall_us_floors_change_every_step"]["candidate"], "dev", v["device_us_floors_change_every_step"]["candidate"], v["device_us_floors_change_every_step"]["verdict"])
for k, v in out["bit_exact"].items():
    print(k, v["graph_equals_eager_in_candidate"], v["candidate_graph_equals_base_eager_across_images"])
