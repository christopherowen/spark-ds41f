#!/usr/bin/env python3
"""The lab's table for a vision evaluation run: score.py over the arms' answers.

usage: tables_arms.py RUN LABEL [LABEL]   (items from ~/spark-lab/vision-eval/data)
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ITEMS = Path.home() / "spark-lab/vision-eval/data/items.jsonl"

run, labels = sys.argv[1], sys.argv[2:]
answers = [str(ROOT / f"results/private/determinism/{run}/vision-{label}.jsonl") for label in labels]
print(f"vision evaluation {run}: {' against '.join(labels)} (benchmark means in %)", flush=True)
sys.exit(subprocess.run([sys.executable, str(Path(__file__).with_name("score.py")),
                         str(ITEMS), *answers]).returncode)
