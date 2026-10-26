"""Where production's time to first token goes, mix by mix (the inventory's simulated sizes).

Each request's TTFT splits into: frontend (sent to the engine's arrival), waiting for a slot
(--max-num-seqs full), waiting behind another prompt's prefill chunk (the step's budget taken
by a chunk of 1,024 tokens or more), other waiting (the step in flight, late admission), its own
prefill (admission to its last chunk's end, including steps it sat out between chunks) and the
held first token (sink rejection: its first decode step). Writes runs/ttft-split.txt.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibrate  # noqa: E402
import evaluate  # noqa: E402
import stallsim  # noqa: E402

PARTS = ("frontend", "slots", "behind_prefill", "other", "prefill", "held")


def split(shape_name: str, mix: str, seeds: int = 4) -> dict:
    shape = calibrate.load_shapes()[shape_name]
    build, _ = evaluate.MIXES[shape_name][mix]
    rows: dict = {}
    for seed in range(seeds):
        engine = stallsim.Engine(shape, stallsim.Policy(), seed=seed)
        start, end, until = build(engine, random.Random(1000 + seed), shape_name)
        engine.run(until)
        for r in engine.requests:
            if not (start <= r.sent < end) or r.first is None or r.admitted is None or r.prefilled is None:
                continue
            parts = {"frontend": r.arrival - r.sent, "slots": r.waits.get("slots", 0.0),
                     "behind_prefill": r.waits.get("behind_prefill", 0.0),
                     "prefill": r.prefilled - r.admitted, "held": r.first - r.prefilled}
            parts["other"] = max((r.first - r.sent) - sum(parts.values()), 0.0)
            rows.setdefault(r.cls, []).append((r.first - r.sent, parts))
    return rows


def main() -> int:
    lines = ["TTFT split under production scheduling (means in seconds, share of the mean TTFT)"]
    for shape_name, mixes in evaluate.MIXES.items():
        for mix in mixes:
            rows = split(shape_name, mix)
            for cls, items in sorted(rows.items()):
                if cls in ("stream", "agent-primer") or len(items) < 3:
                    continue
                mean = sum(t for t, _ in items) / len(items)
                parts = {k: sum(p[k] for _, p in items) / len(items) for k in PARTS}
                p90 = stallsim.percentile([t for t, _ in items], 0.9)
                cells = "  ".join(f"{k} {parts[k]:.2f} ({parts[k] / mean * 100:.0f}%)" for k in PARTS)
                lines.append(f"{shape_name:<13} {mix:<21} {cls:<11} n={len(items):<5} mean {mean:7.2f}  p90 {p90:7.2f} | {cells}")
    text = "\n".join(lines) + "\n"
    (HERE / "runs/ttft-split.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
