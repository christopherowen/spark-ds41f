"""Markdown tables for the README from runs/results.json (writes runs/readme-tables.md)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rank  # noqa: E402

FIXES = ["prefill yield", "slots", "prefill yield + slots", "chunks 8K beside decodes", "chunks 4K beside decodes",
         "graphed decode rows*", "held first token off*"]
POLICY = {"tp4-ring": {"prefill yield": "bypass-ft", "slots": "slots-24", "prefill yield + slots": "bypass-ft+slots",
                       "chunks 8K beside decodes": "tiers-8k", "chunks 4K beside decodes": "tiers-4k",
                       "graphed decode rows*": "decode-rows-graphed*", "held first token off*": "sink-off*",
                       "pool 2x": "pool-2x"},
          "tp3-triangle": {"prefill yield": "bypass-ft", "slots": "slots-12", "prefill yield + slots": "bypass-ft+slots",
                           "chunks 4K beside decodes": "tiers-4k", "retention 2,048": "retention-2048",
                           "pool +1.3 GiB": "pool+1.3GiB", "held first token off*": "sink-off*"}}


def s(value, digits=2):
    return "-" if value is None else f"{value:.{digits}f}"


def pct(value):
    return "-" if value is None else f"{(value - 1) * 100:+.0f}%"


def main() -> int:
    results = json.loads((HERE / "runs/results.json").read_text())
    out = []
    for shape, mixes in results.items():
        out.append(f"\n#### {shape}: production\n")
        out.append("| Mix | Users wait on | TTFT p50 | TTFT p99 | Their ITL p99 | Long prompt TTFT p50 | Output tok/s | Recomputed tok/s |")
        out.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for mix, by_policy in mixes.items():
            base = by_policy["production"]
            user, stream, long = rank.KEYS[mix]
            u = base["classes"].get(user, {})
            itl = base["classes"].get(stream, {}).get("itl_p99") if stream else base.get("stall_p50")
            lt = base["classes"].get(long, {}).get("ttft_p50") if long else None
            out.append(f"| {mix} | {user} | {s(u.get('ttft_p50'))} s | {s(u.get('ttft_p99'))} s | "
                       f"{s(itl)} s{' (stall step)' if not stream else ''} | {s(lt, 1) + ' s' if lt else '-'} | "
                       f"{s(base['output_tok_s'], 1)} | {s(base['recompute_tok_s'], 0)} |")
        out.append(f"\n#### {shape}: change against production (user TTFT p50 / p99, output, long-prompt TTFT)\n")
        names = [n for n in POLICY[shape]]
        out.append("| Mix | " + " | ".join(names) + " |")
        out.append("| --- | " + " | ".join("---" for _ in names) + " |")
        for mix, by_policy in mixes.items():
            cells = []
            for name in names:
                policy = POLICY[shape][name]
                if policy not in by_policy:
                    cells.append("-")
                    continue
                m = rank.measures(by_policy[policy], by_policy["production"], mix)
                cell = f"{pct(m['ttft_p50'])} / {pct(m['ttft_p99'])}, out {pct(m['output'])}"
                if m.get("long_ttft") is not None:
                    cell += f", long {pct(m['long_ttft'])}"
                cells.append(cell)
            out.append(f"| {mix} | " + " | ".join(cells) + " |")
    text = "\n".join(out) + "\n"
    (HERE / "runs/readme-tables.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
