#!/usr/bin/env python3
"""Write the top-k tie-break arms from the r5l candidate (run from the repository root)."""
import copy
import json
from pathlib import Path

E = Path("experiments/2026-09-29-topk-ties")
OVERLAY = "{home}/spark3-overlay/topk-ties/b12x/attention/dsa_indexer/tiled_topk.py"
TARGET = "/opt/spark3/candidate/b12x/b12x/attention/dsa_indexer/tiled_topk.py"

base = json.loads(Path("experiments/2026-09-29-r5l/cluster-candidate.json").read_text())
ties = copy.deepcopy(base)
ties["container"]["mounts"].append([OVERLAY, TARGET, "ro"])
# Validation boot: patch 0025's split check, which should now find no
# difference at all, split against full or full against full.
check = copy.deepcopy(ties)
check["environment"]["SPARK3_DS41_INDEXER_SPLIT_CHECK"] = "1"

for name, config in (("ties", ties), ("ties-check", check)):
    path = E / f"cluster-{name}.json"
    path.write_text(json.dumps(config, indent=2) + "\n")
    print("wrote", path)
