"""Engine cost grid: identical work timed inside the server.

The server times its own dummy steps (the speculative-decoding cost profile's fixed-shape
steps) at the shapes asked for: verification rows for 1-16 requests at the draft depths
serving runs, at short and long context. Every build does the same work, with no generated
text, draft acceptance or scheduling in it, so the step times are the engine's speed alone.
Needs a server that exposes ``engine_grid`` to the dev-mode collective RPC (vLLM with
``VLLM_SERVER_DEV_MODE=1``).

Prefill chunks are optional (``prefill=True``). Dummy prefill does not track real prefill:
on 2026-10-14 it ran 1.7x faster than a real 16K prefill's kernels and called a build 1%
slower that real cold prefill found level or faster, presumably because identical dummy
tokens route the MoE unlike text. Judge prefill on real prompts.
"""

import json
import math
import urllib.request

from . import prom, stats

SCHEMA = "servebench-engine/1"


def default_shapes(drafts: int = 5, max_context: int | None = None, prefill: bool = False) -> list[dict]:
    """Verification rows (requests x (1 + depth)) at 4K context:
    - 1, 4, 8 and 16 requests at depths 1 and 2, what adaptive verification runs under
      load (about two rows a request at 8-16 streams on 2026-10-14);
    - 1, 2, 4, 8, 12 and 16 requests at the full ``drafts``.
    Then depth 2 for 1, 8 and 16 requests at 64K, 1 and 8 at 256K, and 1 at the server's
    longest context. With ``prefill``, 8K chunks at 0 and 256K context too."""
    def decode(requests: int, depth: int, context: int) -> dict:
        return {"kind": "decode", "num_reqs": requests, "num_tokens": requests * (1 + depth),
                "context_len": context}
    shapes = [decode(r, d, 4096) for d in (1, 2) for r in (1, 4, 8, 16)]
    shapes += [decode(r, drafts, 4096) for r in (1, 2, 4, 8, 12, 16)]
    shapes += [decode(r, 2, 65536) for r in (1, 8, 16)]
    shapes += [decode(r, 2, 262144) for r in (1, 8)]
    if max_context and max_context > 262144:
        shapes.append(decode(1, 2, max_context - 8192))
    if prefill:
        shapes += [{"kind": "prefill", "prefill": True, "num_tokens": 8192, "context_len": c} for c in (0, 262144)]
    return shapes


def label(shape: dict) -> str:
    context = shape.get("context_len", 0)
    depth = f"{context // 1024}K" if context < 1 << 20 else f"{context / (1 << 20):.2g}M"
    if shape.get("prefill"):
        return f"prefill {shape['num_tokens']} @{depth}"
    return f"decode {shape.get('num_reqs', '?')}x{shape['num_tokens']} rows @{depth}"


def max_model_len(base_url: str) -> int | None:
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=60) as response:
        return json.load(response)["data"][0].get("max_model_len")


def _post(url: str, body: dict | None, timeout: float) -> dict | None:
    request = urllib.request.Request(url, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode()
    return json.loads(text) if text.strip() else None


def run(base_url: str, shapes: list[dict], *, replays: int = 100, warmup: int = 2, settle: int = 1,
        timeout: float = 3600.0, require_idle: bool = True) -> dict:
    """Time ``shapes`` on the server and reset its prefix cache afterwards."""
    base = base_url.rstrip("/")
    if require_idle:
        prom.require_idle(base, "the grid")
    spec = {"shapes": [{k: v for k, v in s.items() if k != "kind"} for s in shapes],
            "replays": replays, "warmup": warmup, "settle": settle}
    answer = _post(base + "/collective_rpc",
                   {"method": "engine_grid", "args": [json.dumps(spec)], "timeout": timeout},
                   timeout + 60)
    workers = (answer or {}).get("results") or []
    grids = [w for w in workers if isinstance(w, dict)]
    if not grids:
        raise RuntimeError(f"no engine grid in the reply: {str(answer)[:300]}")
    reset = True
    try:
        _post(base + "/reset_prefix_cache", None, 120)
    except Exception:  # noqa: BLE001 - recorded: later cache hits could read dummy KV
        reset = False
    rank0 = min(grids, key=lambda g: g.get("rank", 0))
    return {"workers": len(workers), "rank": rank0.get("rank"), "prefix_cache_reset": reset,
            "replays": replays, "warmup": warmup, "settle": rank0.get("settle"), "shapes": shapes,
            "grid": rank0["shapes"]}


def summarize(raw: dict) -> list[dict]:
    """Per shape: whole-step milliseconds with its 95% t interval, and the target /
    drafter split where the server timed it."""
    rows = []
    for shape, timed in zip(raw["shapes"], raw["grid"]):
        row = {"label": label(shape), "shape": shape, "step_ms": stats.describe(timed["step_ms"]),
               "step_ms_median": stats.percentiles(timed["step_ms"], (50,)).get("p50"),
               "samples": [round(v, 4) for v in timed["step_ms"]]}
        if timed.get("forward_ms"):
            row["forward_ms"] = stats.describe(timed["forward_ms"])
            row["drafter_ms"] = stats.describe(timed["drafter_ms"])
        rows.append(row)
    return rows


def compare(base: dict, candidate: dict, *, tolerance_pct: float = 0.5, confidence: float = 0.95) -> dict:
    """Shape by shape: step-time change (lower is better) after dropping slow outliers,
    with Welch intervals that hold for all shapes at once (Bonferroni), then the decode
    shapes' geometric mean."""
    by_label = {row["label"]: row for row in base["engine"]}
    pairs, warnings = [], []
    for row in candidate["engine"]:
        other = by_label.get(row["label"])
        if other is None:
            warnings.append(f"{row['label']}: not in the base run")
        else:
            pairs.append((other, row))
    family = 1 - (1 - confidence) / max(len(pairs), 1)
    rows, logs = [], []
    for other, row in pairs:
        a, dropped_a = stats.trim_slow(other["samples"])
        b, dropped_b = stats.trim_slow(row["samples"])
        test = stats.welch(a, b, confidence=family)
        if test is None:
            continue
        rows.append({"metric": row["label"], "base": sum(a) / len(a), "candidate": sum(b) / len(b),
                     "change_pct": test["change_pct"], "ci95_pct": test["ci95_pct"],
                     "trimmed": dropped_a + dropped_b,
                     "verdict": stats.verdict(test["change_pct"], test["ci95_pct"], False, tolerance_pct)})
        if not row["shape"].get("prefill"):
            logs.append((math.log(1 + test["change_pct"] / 100), test["se_pct"] / 100))
    if len(logs) > 1:
        mean = sum(l for l, _ in logs) / len(logs)
        se = math.sqrt(sum(s * s for _, s in logs)) / len(logs)
        change, ci = 100 * (math.exp(mean) - 1), 100 * stats.t_quantile(1 - (1 - confidence) / 2, 1e6) * se
        rows.append({"metric": f"decode geometric mean ({len(logs)})", "base": None, "candidate": None,
                     "change_pct": change, "ci95_pct": ci,
                     "verdict": stats.verdict(change, ci, False, tolerance_pct)})
    note = (f"intervals hold for all {len(pairs)} shapes at once ({100 * confidence:.0f}%, Bonferroni); "
            f"slow outliers (median + 5 MAD) dropped first: {sum(r.get('trimmed', 0) for r in rows)}")
    return {"warnings": warnings, "rows": rows, "note": note}
