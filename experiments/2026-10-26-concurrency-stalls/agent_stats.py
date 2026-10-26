"""The AgentX-like generator's statistics beside what #574 publishes about the corpus.

Targets (experiments/2026-10-25-agentx/README.md on branch agentx-traffic): a median request
input of about 142K tokens; about half of main-agent turns over 256K; an unbounded cache hits
about 98% of prompt tokens. Writes runs/agent-stats.txt.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calibrate  # noqa: E402
import stallsim  # noqa: E402
import workloads  # noqa: E402

WARM, WINDOW = 900.0, 3600.0


def median(values: list):
    return sorted(values)[len(values) // 2] if values else None


def stats(shape: stallsim.Shape, lanes: int, seed: int = 1) -> str:
    engine = stallsim.Engine(shape, stallsim.Policy(), seed=seed)
    workloads.agentx(engine, random.Random(seed), lanes, 0.0, WARM + WINDOW)
    engine.run(WARM + WINDOW + 900.0)
    requests = [r for r in engine.requests if WARM <= r.sent < WARM + WINDOW and not r.cls.endswith("primer")]
    main = [r.prompt for r in requests if r.cls == "agent"]
    sub = [r.prompt for r in requests if r.cls == "agent-sub"]
    prompt = sum(r.prompt for r in requests)
    ideal = sum(r.ideal_hit for r in requests) / prompt
    hit = sum(r.hit_tokens for r in requests) / prompt
    uncached = sum(r.prompt - r.hit_tokens for r in requests) / WINDOW
    summary = stallsim.summarize(engine, WARM, WARM + WINDOW)
    agent = summary["classes"].get("agent", {})
    over = sum(1 for p in main if p > 256000) / max(len(main), 1)
    return (f"{shape.name} c{lanes}: {len(requests)} requests ({len(main)} main, median {median(main):,}; "
            f"{len(sub)} subagent, median {median(sub) or 0:,}); all median {median([r.prompt for r in requests]):,}; "
            f"main over 256K {over:.2f}; unbounded-cache hit {ideal:.3f}, simulated hit {hit:.3f}; "
            f"uncached {uncached:,.0f} tok/s; output {summary['output_tok_s']:.0f} tok/s; "
            f"main TTFT p50 {agent.get('ttft_p50', 0):.2f} s, p90 {agent.get('ttft_p90', 0):.2f} s")


def main() -> int:
    shapes = calibrate.load_shapes()
    lines = [stats(shapes["tp4-ring"], lanes) for lanes in (4, 8, 16)]
    lines += [stats(shapes["tp3-triangle"], lanes) for lanes in (2, 4)]
    text = "\n".join(lines) + "\n"
    print(text, end="")
    (HERE / "runs/agent-stats.txt").write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
