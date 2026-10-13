"""Engine cost grid: identical work timed inside the server.

The server times its own dummy steps (the speculative-decoding cost profile's fixed-shape
steps) at the shapes asked for: verification rows for 1-16 requests at short and long
context, single long-context steps, and prefill chunks. Every build does the same work,
with no generated text, draft acceptance or scheduling in it, so the step times are the
engine's speed alone and their intervals are tight. Needs a server that exposes
``engine_grid`` to the dev-mode collective RPC (vLLM with ``VLLM_SERVER_DEV_MODE=1``).
"""

import json
import urllib.request

from . import prom, stats

SCHEMA = "servebench-engine/1"


def default_shapes(drafts: int = 5, max_context: int | None = None) -> list[dict]:
    """Verification rows (requests x (1 + drafts)) for 1-16 requests at 4K and 64K
    context; one and eight requests at 256K; one request at the server's longest
    context; 8K prefill chunks at 0, 64K and 256K context."""
    rows = 1 + drafts
    shapes = []
    for context in (4096, 65536):
        for requests in (1, 2, 4, 8, 12, 16):
            shapes.append({"kind": "decode", "num_reqs": requests, "num_tokens": requests * rows,
                           "context_len": context})
    for requests in (1, 8):
        shapes.append({"kind": "decode", "num_reqs": requests, "num_tokens": requests * rows,
                       "context_len": 262144})
    if max_context and max_context > 262144:
        shapes.append({"kind": "decode", "num_reqs": 1, "num_tokens": rows,
                       "context_len": max_context - 8192})
    for context in (0, 65536, 262144):
        shapes.append({"kind": "prefill", "prefill": True, "num_tokens": 8192, "context_len": context})
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


def run(base_url: str, shapes: list[dict], *, replays: int = 30, warmup: int = 2,
        timeout: float = 3600.0, require_idle: bool = True) -> dict:
    """Time ``shapes`` on the server and reset its prefix cache afterwards."""
    base = base_url.rstrip("/")
    if require_idle:
        prom.require_idle(base, "the grid")
    spec = {"shapes": [{k: v for k, v in s.items() if k != "kind"} for s in shapes],
            "replays": replays, "warmup": warmup}
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
            "replays": replays, "warmup": warmup, "shapes": shapes, "grid": rank0["shapes"]}


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


def compare(base: dict, candidate: dict, *, tolerance_pct: float = 0.5) -> dict:
    """Shape by shape: step-time change with its Welch interval (lower is better)."""
    by_label = {row["label"]: row for row in base["engine"]}
    rows, warnings = [], []
    for row in candidate["engine"]:
        other = by_label.get(row["label"])
        if other is None:
            warnings.append(f"{row['label']}: not in the base run")
            continue
        test = stats.welch(other["samples"], row["samples"])
        if test is None:
            continue
        rows.append({"metric": row["label"], "base": other["step_ms"]["mean"],
                     "candidate": row["step_ms"]["mean"], "change_pct": test["change_pct"],
                     "ci95_pct": test["ci95_pct"],
                     "verdict": stats.verdict(test["change_pct"], test["ci95_pct"], False, tolerance_pct)})
    return {"warnings": warnings, "rows": rows}
