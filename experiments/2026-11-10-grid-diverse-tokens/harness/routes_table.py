"""The real expert ids of grid version 2's rows, for gridstep.py --routes.

usage: python3 routes_table.py REF_ROUTES_PT OUT_JSON

ref_routes.py ran the 16 sample windows' prompts and replies through the checkpoint's
reference forward; position 64 + k of sequence r is window r's token k. A grid shape's row
(request q, k-th row) is window q % 16's token k (servebench.engine.token_ids), so its real
routes are routes[layer, q % 16, 64 + k]. Writes, per grid shape, [40 layers][rows][6] ids.
"""

import json
import sys

import torch

sys.path.insert(0, "/sb")
from servebench import engine  # noqa: E402

GRID = ["1x2", "4x8", "8x16", "16x32", "1x3", "4x12", "8x24", "16x48",
        "1x6", "2x12", "4x24", "8x48", "12x72", "16x96"]

d = torch.load(sys.argv[1], weights_only=False)
routes, c = d["routes"], d["context"]
windows = [w["token_ids"] for w in engine.tokens()["windows"]]
out = {"source": {"ref_routes": sys.argv[1], "context": c, "grid_tokens_sha256": engine.tokens_sha256()},
       "shapes": {}}
for shape in GRID:
    r, t = map(int, shape.split("x"))
    spec = {"num_reqs": r, "num_tokens": t}
    ids = engine.token_ids(spec, windows)
    rows = []
    for q, n in enumerate(engine.rows_per_request(spec)):
        for k in range(n):
            seq = q % 16
            assert d["sequences"][seq][c + k] == windows[seq][k]  # the row is that window's token
            rows.append((seq, c + k))
    assert len(rows) == t and [d["sequences"][s][p] for s, p in rows] == ids
    out["shapes"][shape] = [[routes[layer, s, p].tolist() for s, p in rows] for layer in range(routes.shape[0])]
with open(sys.argv[2], "w") as fh:
    json.dump(out, fh)
print("wrote", sys.argv[2])
