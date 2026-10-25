#!/usr/bin/env python3
"""The price of R rows of one request on real text, on today's build (no patch needed).

usage (a lab extra, run on the head node from the repository root):
  wide_rows_probe.py HEAD_URL [--repeats 12] [--prefix 4096]

A request whose prompt is a cached prefix plus R new tokens runs one step of R rows of one
request, the same forward a verification block of R rows runs (on the prefill plan, with
piecewise graphs). Its time to first token, at max_tokens 1 and one stream, against R:

- copy: the R tokens repeat a span of the prefix (copied text, what a suffix drafter verifies);
- fresh: the R tokens continue the document (ordinary code text);
- random: uniformly random token ids, whose routes are near independent (the upper bound).

The same R costs the same outside the routed experts in all three, so copy against random at
equal R is the routing's sharing in milliseconds; R against R = 1 is the price of the rows.
The prefix is a multiple of the 256-token cache block, so it is cached and the tail never is.
One JSON line per (kind, R) with every repeat's latency.
"""
import argparse
import json
import random
import statistics
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from servebench import prompts  # noqa: E402

MODEL = "deepseek-v4.1-flash"
ROWS = (1, 6, 17, 33, 49, 65)


def post(url: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def complete(url: str, ids: list[int]) -> float:
    t0 = time.perf_counter()
    post(url, "/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0})
    return 1e3 * (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--prefix", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    url = args.url.rstrip("/")
    assert args.prefix % 256 == 0, "the prefix must be whole cache blocks"
    text = prompts.source_text(args.prefix + 2 * max(ROWS) * args.repeats + 512, args.seed)
    ids = post(url, "/tokenize", {"model": MODEL, "prompt": text, "add_special_tokens": False})["tokens"]
    prefix, rest = ids[: args.prefix], ids[args.prefix :]
    rng = random.Random(args.seed)
    for _ in range(3):  # cache the prefix and warm the shapes
        complete(url, prefix + rest[:1])
    failed = False
    for rows in ROWS:
        for kind in ("copy", "fresh", "random"):
            times = []
            for r in range(args.repeats):
                if kind == "copy":
                    start = rng.randrange(0, args.prefix - rows)
                    tail = prefix[start : start + rows]
                elif kind == "fresh":
                    tail = rest[r * rows : (r + 1) * rows]
                else:
                    tail = [rng.randrange(1000, 120000) for _ in range(rows)]  # below the special ids
                try:
                    times.append(round(complete(url, prefix + tail), 2))
                except (OSError, ValueError, KeyError) as error:
                    failed = True
                    print(json.dumps({"kind": kind, "rows": rows, "error": str(error)[:300]}), flush=True)
            if times:
                print(json.dumps({"kind": kind, "rows": rows, "median_ms": statistics.median(times),
                                  "ms": times}), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
