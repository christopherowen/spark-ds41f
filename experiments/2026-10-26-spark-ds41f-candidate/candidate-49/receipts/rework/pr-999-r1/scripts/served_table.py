#!/usr/bin/env python3
"""Condense served-topk-configs.json into the reachability table for result.json."""
import json
import sys

r = json.load(open(sys.argv[1]))
out = {
    "lsa_env": r["lsa_env"],
    "slow_selector": r["slow_selector"],
    "slow_selector_kernel_calls": r["slow_selector_calls"],
    "slow_selector_route": r["slow_selector_route"],
    "slow_selector_kernels_declared_anywhere": r["slow_selector_kernels_declared_anywhere"],
    "slow_selector_reachable": r["slow_selector_reachable"],
    "per_shape_layers": {},
    "distinct_served_topk_configs": [
        {k: c[k] for k in ("width", "topk", "group", "candidates", "gather_scores", "route")}
        | {"used_by": sorted({u.split(" L")[0] + " L" + u.split(" L")[1] for u in c["used_by"]})}
        for c in r["distinct_served_topk_configs"]
    ],
}
for shape, sh in r["shapes"].items():
    out["per_shape_layers"][shape] = {
        lid: {
            "role": l["role"],
            "compress_ratio": l["compress_ratio"],
            "modes": {m: [v["plan_args"]["width"], v["rows_per_call"]] for m, v in l["modes"].items()},
        }
        for lid, l in sh["layers"].items()
    }
print(json.dumps(out, indent=1))
