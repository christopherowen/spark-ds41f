#!/usr/bin/env python3
"""The README's tables from results.json (cost.py's output).

usage: report.py [results.json]           every variant, by category and workload
       report.py --readme [results.json]  the README's tables
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


README_SOURCES = ("probe", "shared", "independent")
AGENT_ROWS = ("exact_m16_k16_grow", "exact_m16_k16_full", "exact_m24_k16_grow", "exact_m24_k16_full",
              "exact_m8_k16_grow", "exact_m8_k16_full", "exact_m16_k32_grow", "exact_m24_k32_grow",
              "exact_m16_k32_full", "exact_m16_k64_grow", "exact_m24_k64_grow", "exact_m24_k64_full",
              "exact_m8_k64_full", "request_m16_k16_grow", "global_m24_k16_grow", "global_m16_k16_grow",
              "global_m24_k64_grow", "both_m24_k16_grow", "both_m24_k16_full", "both_m16_k16_grow",
              "both_m24_k64_grow")
SHORT_ROWS = ("exact_m16_k16_grow", "exact_m24_k16_grow", "exact_m16_k64_grow", "both_m24_k16_grow")
CASE_ROWS = ("exact_m8_k16_full", "exact_m16_k16_grow", "exact_m24_k16_grow", "exact_m16_k64_grow",
             "exact_m16_k64_full", "request_m16_k16_grow")


def label(key):
    family, m, k, policy = key.split("_")
    return f"{family} m{m[1:]} k{k[1:]} {policy.removesuffix('-nolag')}"


def cell(table, key, shape, sources, digits=1, unit=""):
    values = [gains(table, key, source=src)[shape] for src in sources if src in table[key]]
    return " / ".join(f"{100 * v:+.{digits}f}{unit}".replace("-", "−") for v in values)


def readme(results):
    """The README's tables, gains as measured routes / ordinary-text sharing / independent routes."""
    for shape in ("tp4-c1", "tp3-c1"):
        info = results["shapes"][shape]
        print(f"\n{shape}")
        print("| Rows (graph) | experts a layer | step ms | ms a token if all accepted |")
        print("| --- | --- | --- | --- |")
        for row in info["widths"]:
            src = [s for s in ("probe copy", "json-count +2.4/row", "uniform") if s in row]
            print(f"| {row['rows']} ({row['padded_to']}) | " + " | ".join(
                " / ".join(f"{row[s][field]:.1f}" for s in src)
                for field in ("experts_a_layer", "step_ms", "ms_a_token_if_all_accepted")) + " |")
    print("\ncopy cases (file TP4 c1, TP4 c8, TP3 c1; edit TP4 c1, TP4 c8; tool TP4 c1, TP4 c8)")
    cases = results["copy_cases"]
    for cat in cases:
        base = cases[cat]["measured_dspark"]["dspark_m0_k0_full"]
        print(cat, "DSpark alone", {s: [base[src][s]["tok_s"] for src in README_SOURCES if src in base]
                                    for s in ("tp4-c1", "tp4-c8")})
    for key in CASE_ROWS:
        cells = []
        for cat, shapes in (("copy", ("tp4-c1", "tp4-c8", "tp3-c1")), ("edit", ("tp4-c1", "tp4-c8")),
                            ("tool", ("tp4-c1", "tp4-c8"))):
            table = cases[cat]["measured_dspark"]
            cells += [cell(table, key, s, README_SOURCES, digits=0, unit="%") for s in shapes]
        print(f"| {label(key)} | " + " | ".join(cells) + " |")
    proxy = results["agent_proxy"]
    for mode, rows in (("visible", AGENT_ROWS), ("reasoning_1to1", SHORT_ROWS)):
        table = proxy[mode]
        base = table["dspark_m0_k0_full"]
        print(f"\nagent proxy {mode}: DSpark alone",
              {s: [base[src][s]["tok_s"] for src in README_SOURCES if src in base] for s in SHAPES})
        for key in rows:
            w = table[key]["walk"]
            print(f"| {label(key)} | {w['tokens_per_step']:.2f} | {round(100 * w['suffix_token_share'])}% | "
                  f"{w['drafts_per_suffix_step']:.1f} / {w['accepted_per_suffix_step']:.1f} | "
                  f"{w['wasted_rows_per_token']:.2f} | " + " | ".join(cell(table, key, s, README_SOURCES) for s in SHAPES) + " |")
    print("\nagent proxy without the allocation lag")
    table = proxy["visible"]
    for key in SHORT_ROWS:
        print(f"| {label(key)} | " + " | ".join(cell(table, key + "-nolag", s, README_SOURCES)
                                                for s in ("tp4-c1", "tp4-c8")) + " |")


def main():
    if "--readme" in sys.argv:
        args = [a for a in sys.argv[1:] if a != "--readme"]
        readme(json.load(open(args[0] if args else HERE / "results.json")))
        return
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
