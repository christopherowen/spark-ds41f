#!/usr/bin/env python3
"""Write image-cap-32-tp3.json: spark-ds41f-1025's recipe at tp3-triangle with 32 images a prompt.

usage: make_arms.py

An arm's serve_args replace the whole list, so the arm copies the recipe's resolved list and
changes the one entry.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import recipe  # noqa: E402

CAP = 32
FLAG = "--limit-mm-per-prompt"
RECIPE = "releases/spark-ds41f-1025/cluster.json"


def main() -> None:
    resolved = recipe.resolve(json.loads((ROOT / RECIPE).read_text()), "tp3-triangle")
    args = list(resolved["serve_args"])
    args[args.index(FLAG) + 1] = json.dumps({"image": CAP}, separators=(",", ":"))
    arm = {
        "notes": f"The recipe at tp3-triangle with {FLAG} {{\"image\":{CAP}}} (the recipe's is 4) and nothing "
                 "else changed: one boot serves every image count image_cap.py sends.",
        "extends": {"config": RECIPE, "shape": "tp3-triangle"},
        "set": {"serve_args": args},
    }
    (HERE / f"image-cap-{CAP}-tp3.json").write_text(json.dumps(arm, indent=1) + "\n")


if __name__ == "__main__":
    main()
