#!/usr/bin/env python3
"""The README's tables from results.json (cost.py's output).

usage: report.py [results.json]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHAPES = ("tp4-c1", "tp4-c8", "tp3-c1", "tp3-c8")


def pct(x):
    return f"{100 * x:+.1f}%"


def widths(results):
    for shape in ("tp4-c1", "tp3-c1"):
        info = results["shapes"][shape]
        sources = [k for k in info["widths"][0] if isinstance(info["widths"][0][k], dict)]
        print(f"\n{shape}: overhead {info['overhead_ms']} ms, dense {info['dense_mb']} MB, expert "
              f"{info['expert_mb']} MB, dead row {info['dead_row_ms']} ms, graph ladder to {info['ladder_top']}")
        print("| Rows (graph) | " + " | ".join(f"{s}: experts, step ms, ms a token" for s in sources) + " |")
        print("| --- | " + " | ".join("---" for _ in sources) + " |")
        for row in info["widths"]:
            cells = [f"{row[s]['experts_a_layer']}, {row[s]['step_ms']}, {row[s]['ms_a_token_if_all_accepted']}"
                     for s in sources]
            print(f"| {row['rows']} ({row['padded_to']}) | " + " | ".join(cells) + " |")


def gains(table, key, base="dspark_m0_k0_full", source="shared"):
    return {s: table[key][source][s]["tok_s"] / table[base][source][s]["tok_s"] - 1 for s in SHAPES}


def variant_table(table, keys, sources=("shared", "independent")):
    print("| Variant | tokens a step | suffix share | drafts / accepted a suffix step | wasted rows a token | "
          + " | ".join(SHAPES) + " |")
    print("| --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in SHAPES) + " |")
    for key in keys:
        w = table[key]["walk"]
        g = {src: gains(table, key, source=src) for src in sources if src in table[key]}
        cells = [" / ".join(pct(g[src][s]) for src in g) for s in SHAPES]
        print(f"| {key} | {w['tokens_per_step']} | {w['suffix_token_share']} | {w['drafts_per_suffix_step']} / "
              f"{w['accepted_per_suffix_step']} | {w['wasted_rows_per_token']} | " + " | ".join(cells) + " |")


def main():
    results = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / "results.json"))
    print("routes:", json.dumps(results["inputs"]))
    widths(results)
    for cat, modes in results.get("copy_cases", {}).items():
        table = modes["measured_dspark"]
        base = table["dspark_m0_k0_full"]
        print(f"\n## copy case {cat}: DSpark alone " + ", ".join(
            f"{s} {base['shared'][s]['tok_s']}" for s in SHAPES))
        keys = sorted((k for k in table if not k.startswith("dspark")),
                      key=lambda k: -gains(table, k)["tp4-c1"])
        variant_table(table, keys[:8] + keys[-3:])
    for mode in ("visible", "reasoning_1to1"):
        table = results.get("agent_proxy", {}).get(mode)
        if not table:
            continue
        base = table["dspark_m0_k0_full"]
        print(f"\n## agent proxy, {mode}: DSpark alone " + ", ".join(
            f"{s} {base['shared'][s]['tok_s']}" for s in SHAPES) + f", {base['walk']['tokens_per_step']} tokens a step")
        keys = sorted((k for k in table if not k.startswith("dspark")),
                      key=lambda k: -gains(table, k)["tp4-c1"])
        variant_table(table, keys)


if __name__ == "__main__":
    main()
