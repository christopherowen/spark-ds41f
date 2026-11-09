"""#998 analysis: ab-*.json (image A/B at TP3), script-c53-boot*.txt (the PR's in-process bench), pytest."""
import json, sys, os, re
sys.path.insert(0, os.path.dirname(__file__)); from common import cell
d = sys.argv[1]
labs = ("base-boot1", "c53-boot1", "c53-boot2", "base-boot2")
A = {l: json.load(open(f"{d}/ab-{l}.json")) for l in labs}
out = {"memory": {}, "kernels": {}, "placement": {l: A[l]["memory"]["placement"] for l in labs},
       "carveout_buffers_mib": {l: A[l]["memory"]["carveout_buffers_mib"] for l in labs}}
for m in ("carveout_total_mib", "carveout_weight_mib", "uma_mem_available_used_mib", "uma_cuda_free_used_mib", "cuda_allocated_mib", "cuda_reserved_mib", "cuda_peak_reserved_mib"):
    out["memory"][m] = cell([A[l]["memory"][m] for l in ("base-boot1", "base-boot2")], [A[l]["memory"][m] for l in ("c53-boot1", "c53-boot2")], higher=m.startswith("carveout"))
for k in A["base-boot1"]["kernels"]:
    c = cell([A[l]["kernels"][k]["us"] for l in ("base-boot1", "base-boot2")], [A[l]["kernels"][k]["us"] for l in ("c53-boot1", "c53-boot2")])
    c["bit_identical_across_images_and_boots"] = len({A[l]["kernels"][k]["digest"] for l in labs}) == 1
    out["kernels"][k] = c
def pjson(p):
    t = open(p).read(); i = t.find('{\n  "boot"')
    return json.loads(t[i:t.rindex("}") + 1]) if i >= 0 else None
pr = {}
for k in (1, 2):
    p = f"{d}/script-c53-boot{k}.txt"
    if os.path.exists(p):
        j = pjson(p)
        if j:
            pr[f"boot{k}"] = {s: {"savings": j[s]["savings"], "baseline": j[s]["baseline"], "candidate": j[s]["candidate"],
                                   "kernels": {n: (v["regular_us"], v["carveout_us"], v["ratio"], v["bit_identical"]) for n, v in j[s]["kernels"].items()}} for s in ("tp3", "tp4")}
out["pr_script_candidate_image"] = pr
pt = {}
for k in (1, 2):
    p = f"{d}/pytest-c53-boot{k}.txt"
    if os.path.exists(p):
        last = [l for l in open(p).read().splitlines() if re.search(r"\d+ (passed|failed)", l)]
        pt[f"boot{k}"] = last[-1] if last else None
out["pytest"] = pt
json.dump(out, open(f"{d}/analysis.json", "w"), indent=1)
for k, v in out["memory"].items(): print("mem", k, v["base"], v["candidate"], v["delta"], v["verdict"])
for k, v in out["kernels"].items(): print("kern", k, v["base"], v["candidate"], v["delta_pct"], v["verdict"], v["bit_identical_across_images_and_boots"])
print(json.dumps(out["placement"])); print(json.dumps(out["carveout_buffers_mib"]))
for b, v in pr.items():
    for s in v: print("pr", b, s, json.dumps(v[s]["savings"]), json.dumps(v[s]["kernels"]))
print(json.dumps(pt))
