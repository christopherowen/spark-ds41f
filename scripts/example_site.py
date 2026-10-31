#!/usr/bin/env python3
"""The example node maps for a checkout without a site: CI and contributors.

A site's node maps stay out of git (``config/nodes.json``, ``config/*.local.json``).
Without them no shape of the recipe resolves, so ``bin/spark`` and the tests that
check real arms cannot run. This writes every shape's missing map from its example,
with the head at the recipe's ``distributed.master_addr`` as on a real site. A map
that exists is the site's own and is kept.

Usage: scripts/example_site.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

NODES_FILE = "config/nodes.json"
EXAMPLES = {
    NODES_FILE: "config/nodes.example.json",
    "config/nodes-ring4.local.json": "config/examples/nodes-ring4.json",
    "config/nodes-switched.local.json": "config/examples/nodes-switched.json",
}


def node_maps(cluster: dict) -> list[str]:
    """The node map of every shape of the recipe."""
    return sorted({recipe.shape_entry(cluster, name).get("nodes_config", NODES_FILE)
                   for name in cluster["shapes"]})


def main() -> int:
    cluster = json.loads((ROOT / "config/cluster.json").read_text())
    maps = node_maps(cluster)
    missing = [name for name in maps if name not in EXAMPLES]
    if missing:
        raise SystemExit(f"no example for {', '.join(missing)}: add one to EXAMPLES in {Path(__file__).name}")
    address = cluster["distributed"]["master_addr"]
    for name in maps:
        target = ROOT / name
        if target.exists():
            print(f"kept {name} (this site's own)")
            continue
        nodes = json.loads((ROOT / EXAMPLES[name]).read_text())
        next(node for node in nodes["nodes"] if node.get("head"))["management_ip"] = address
        target.write_text(json.dumps(nodes, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {name} from {EXAMPLES[name]}, head at {address}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
