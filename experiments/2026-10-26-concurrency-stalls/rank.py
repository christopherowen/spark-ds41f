"""Rank the fixes from runs/results.json: each mix's key measures against production.

For every policy and mix: output tok/s; TTFT p50 and p99 of the class users wait on in that mix
(short requests: chat or probes; agents: main-agent turns); decode ITL p99 of the streams that
decode through long prefills; the long prompts' own TTFT; recomputed prompt tokens. Then the
geometric mean of each ratio over the mixes where it applies. Writes runs/ranking.txt.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# mix -> (class whose TTFT users feel, class whose ITL shows decode stalls, long-prompt class)
KEYS = {
    "stall": ("probe", None, "doc"),
    "stall-1m": ("probe", None, "doc"),
    "chat-docs": ("chat", "chat", "doc"),
    "agents-c8": ("agent", "agent", None),
    "agents-c16": ("agent", "agent", None),
    "agents-c24": ("agent", "agent", None),
    "agents-c4": ("agent", "agent", None),
    "agents-c8+chat": ("chat", "chat", None),
    "agents-c8+chat-heavy": ("chat", "chat", None),
    "agents-c2+chat": ("chat", "chat", None),
    "soak": ("chat", "chat", "soak"),
    "poisson": ("chat", "chat", None),
}
OVERLOAD = {"agents-c16", "agents-c24", "agents-c8+chat-heavy"}


def ratio(new, old):
    if new is None or old is None or old <= 0 or new <= 0:
        return None
    return new / old


def geomean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def pct(value):
    return "-" if value is None else f"{(value - 1) * 100:+.0f}%"


def measures(row: dict, base: dict, mix: str) -> dict:
    user, stream, long = KEYS[mix]
    u, ub = row["classes"].get(user, {}), base["classes"].get(user, {})
    out = {"output": ratio(row["output_tok_s"], base["output_tok_s"]),
           "ttft_p50": ratio(u.get("ttft_p50"), ub.get("ttft_p50")),
           "ttft_p99": ratio(u.get("ttft_p99"), ub.get("ttft_p99"))}
    if stream:
        s, sb = row["classes"].get(stream, {}), base["classes"].get(stream, {})
        out["itl_p99"] = ratio(s.get("itl_p99"), sb.get("itl_p99"))
    if mix.startswith("stall"):
        out["itl_p99"] = ratio(row.get("stall_p50"), base.get("stall_p50"))
    if long:
        lr, lb = row["classes"].get(long, {}), base["classes"].get(long, {})
        out["long_ttft"] = ratio(lr.get("ttft_p50"), lb.get("ttft_p50"))
    out["livelocked"] = row.get("livelocked", 0)
    return out


def main() -> int:
    results = json.loads((HERE / "runs/results.json").read_text())
    lines = []
    for shape, mixes in results.items():
        policies = list(next(iter(mixes.values())))
        lines.append(f"\n=== {shape}: geometric means over mixes of each measure's ratio to production "
                     f"(sustainable mixes | overload mixes)")
        lines.append(f"{'policy':<24} {'output':>7} {'TTFT p50':>9} {'TTFT p99':>9} {'ITL p99':>8} {'long TTFT':>9}"
                     f" | {'output':>7} {'TTFT p50':>9} {'TTFT p99':>9}  livelocks")
        rows = []
        for policy in policies:
            if policy == "production":
                continue
            groups = {False: [], True: []}
            locks = 0
            for mix, by_policy in mixes.items():
                m = measures(by_policy[policy], by_policy["production"], mix)
                locks += m["livelocked"]
                if m["livelocked"]:
                    continue
                groups[mix in OVERLOAD].append(m)
            ok, over = groups[False], groups[True]
            summary = {k: geomean([m.get(k) for m in ok]) for k in ("output", "ttft_p50", "ttft_p99", "itl_p99", "long_ttft")}
            heavy = {k: geomean([m.get(k) for m in over]) for k in ("output", "ttft_p50", "ttft_p99")}
            rows.append(((1 if locks else 0), summary["ttft_p99"] or 9, policy, summary, heavy, locks))
        for _, _, policy, s, h, locks in sorted(rows, key=lambda r: (r[0], r[1])):
            lines.append(f"{policy:<24} {pct(s['output']):>7} {pct(s['ttft_p50']):>9} {pct(s['ttft_p99']):>9}"
                         f" {pct(s['itl_p99']):>8} {pct(s['long_ttft']):>9} | {pct(h['output']):>7}"
                         f" {pct(h['ttft_p50']):>9} {pct(h['ttft_p99']):>9}  {locks or ''}")
        lines.append("\nper mix (output / user TTFT p50 / p99 / decode ITL p99 or stall step / long TTFT):")
        for policy in policies:
            if policy == "production":
                continue
            cells = []
            for mix, by_policy in mixes.items():
                m = measures(by_policy[policy], by_policy["production"], mix)
                if m["livelocked"]:
                    cells.append(f"{mix}: LIVELOCK")
                    continue
                ci = by_policy[policy].get("intervals", {})

                def band(key):
                    lo_hi = ci.get(key)
                    return f"[{pct(lo_hi[0])},{pct(lo_hi[1])}]" if lo_hi else ""
                cells.append(f"{mix}: out {pct(m['output'])}{band('output')} TTFT p50 {pct(m['ttft_p50'])}"
                             f"{band('ttft_p50')} p99 {pct(m['ttft_p99'])}{band('ttft_p99')} "
                             f"ITL/stall {pct(m.get('itl_p99'))} long {pct(m.get('long_ttft'))}")
            lines.append(f"  {policy}:")
            lines.extend(f"      {cell}" for cell in cells)
    text = "\n".join(lines) + "\n"
    (HERE / "runs/ranking.txt").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
