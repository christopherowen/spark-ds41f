#!/usr/bin/env python3
"""#978 live check: a two-turn image chat, whose second turn takes #978's multimodal prefix-cache fast
path (the recipe serves --mm-processor-cache-gb 0), returns the base's text at temperature 0. A lab
extra against a served endpoint; the base arm runs it first and the candidate compares with it.

usage: check_978_image_chat.py URL --label LABEL [--against PATH] [--size 1302] [--max-tokens 192]

- **Image:** a deterministic --size x --size PNG (colour bands, a grid and three filled shapes, written
  with zlib), the same bytes on every arm and run.
- **Conversation:** turn 1 sends the image and asks what it shows; turn 2 sends turn 1 again with its
  reply and asks a follow-up about the same image. Temperature 0, thinking off, --max-tokens. The
  conversation then runs a second time, so turn 2 meets an image already seen twice.
- **Record:** each turn's text, its SHA-256, the usage counts (prompt tokens include the image's
  placeholder expansion, which #978 must keep) and the wall time.
- **--against:** another arm's output of this check (the base's). The run passes when every turn's
  text and prompt token count equal the other arm's; without it (or when it names this arm's own
  output) the run passes when every turn answered. Whether the two conversations agree with each
  other is recorded (`self_consistent`), not required.

Prints one JSON line per turn, then a summary; exits 1 when a turn fails or the comparison differs.
Stdlib only.
"""
import argparse
import base64
import hashlib
import json
import struct
import sys
import time
import urllib.request
import zlib
from pathlib import Path

TURN_1 = "Describe this image: its colours, the shapes in it and where they are."
TURN_2 = "Looking at the same image again: how many filled shapes are there, and which one is largest?"


def png(size: int) -> bytes:
    """Colour bands, a 16-pixel grid and three filled shapes (a square, a disc and a triangle)."""
    rows = []
    for y in range(size):
        row = bytearray(b"\0")  # filter: none
        for x in range(size):
            band = (x * 6) // size
            r, g, b = ((230, 60, 60), (240, 160, 40), (230, 220, 60), (70, 180, 90), (60, 110, 220), (150, 80, 200))[band]
            if x % 16 == 0 or y % 16 == 0:
                r, g, b = r // 2, g // 2, b // 2
            if size // 8 <= x < size // 3 and size // 8 <= y < size // 3:
                r, g, b = 20, 20, 20
            if (x - 2 * size // 3) ** 2 + (y - size // 3) ** 2 < (size // 7) ** 2:
                r, g, b = 250, 250, 250
            if y > size // 2 and abs(x - size // 2) < (y - size // 2) // 2 and y < 7 * size // 8:
                r, g, b = 0, 90, 60
            row += bytes((r, g, b))
        rows.append(bytes(row))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(b"".join(rows), 6))
            + chunk(b"IEND", b""))


def chat(base: str, model: str, messages: list, max_tokens: int, timeout: float) -> dict:
    body = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "chat_template_kwargs": {"thinking": False}}
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.load(response)
    text = reply["choices"][0]["message"].get("content") or ""
    return {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "usage": reply.get("usage"),
            "finish_reason": reply["choices"][0].get("finish_reason"), "seconds": round(time.monotonic() - started, 3)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--label", required=True)
    parser.add_argument("--against", help="another arm's output of this check (JSON lines)")
    parser.add_argument("--size", type=int, default=1302)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    image = png(args.size)
    summary = {"summary": "#978 two-turn image chat", "label": args.label, "passed": False,
               "image_sha256": hashlib.sha256(image).hexdigest(), "image_bytes": len(image)}
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            model = json.load(response)["data"][0]["id"]
    except OSError as error:
        summary["error"] = f"endpoint did not answer: {error}"
        print(json.dumps(summary))
        return 1
    url = "data:image/png;base64," + base64.b64encode(image).decode()
    first = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                          {"type": "text", "text": TURN_1}]}]
    turns, problems = [], []
    for run in (1, 2):
        try:
            one = chat(base, model, first, args.max_tokens, args.timeout)
            second = [*first, {"role": "assistant", "content": one["text"]}, {"role": "user", "content": TURN_2}]
            two = chat(base, model, second, args.max_tokens, args.timeout)
        except (OSError, ValueError, KeyError, IndexError) as error:
            problems.append(f"conversation {run}: {type(error).__name__}: {error}"[:300])
            break
        for turn, reply in ((1, one), (2, two)):
            row = {"label": args.label, "conversation": run, "turn": turn, **reply}
            turns.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    key = lambda row: (row["conversation"], row["turn"])
    if not problems:
        mine = {key(r): (r["sha256"], (r["usage"] or {}).get("prompt_tokens")) for r in turns}
        summary["self_consistent"] = mine[(1, 1)] == mine[(2, 1)] and mine[(1, 2)] == mine[(2, 2)]
        against = Path(args.against) if args.against else None
        if against and against.is_file() and f"-{args.label}." not in against.name:
            rows = [json.loads(line) for line in against.read_text().splitlines() if line.startswith("{")]
            other = {key(r): (r["sha256"], (r.get("usage") or {}).get("prompt_tokens")) for r in rows if "turn" in r}
            summary["against"] = str(against)
            summary["matches_against"] = {f"{c}.{t}": mine.get((c, t)) == other.get((c, t)) for c, t in sorted(mine)}
            problems += [f"conversation {k.split('.')[0]} turn {k.split('.')[1]} differs from {against.name}"
                         for k, same in summary["matches_against"].items() if not same]
        elif against:
            summary["against"] = f"{against} (not compared: missing, or this arm's own output)"
    summary.update(passed=not problems, problems=problems, model=model,
                   turn_seconds={f"{r['conversation']}.{r['turn']}": r["seconds"] for r in turns})
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
