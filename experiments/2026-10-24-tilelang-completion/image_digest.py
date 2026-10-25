#!/usr/bin/env python3
"""Image requests' outputs, as digests two builds can compare, and the texts.

usage: image_digest.py BASE_URL [--tokens T]

Four generated PNGs (gradients with grids of different cell sizes and sizes, so the
encoder cache misses and the vision tower runs for each), each asked for a description
at temperature 0 with thinking off: one at a time, then all four at once. One JSON line a
mode with the SHA-256 digest of the four choices (message, logprobs, finish reason) and the
texts, so a change in the vision tower's numerics can be read as well as detected. Exits 0
either way; the run records.
"""
import argparse
import base64
import hashlib
import json
import struct
import threading
import urllib.request
import uuid
import zlib

# (width, height, grid cell): one tile, a wide and a tall image, one at the tower's cap.
IMAGES = [(224, 224, 32), (896, 448, 64), (448, 1024, 48), (1344, 1344, 96)]
PROMPT = "Describe this image precisely: colours, the grid, and its proportions."


def png(width: int, height: int, cell: int) -> bytes:
    """A colour gradient with a grid of ``cell`` pixels (no imaging library needed)."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            grid = 255 if (x % cell == 0 or y % cell == 0) else 0
            rows += bytes((x * 255 // width, y * 255 // height, grid))

    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b""))


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def describe(self, image: tuple[int, int, int], tokens: int) -> dict:
        url = "data:image/png;base64," + base64.b64encode(png(*image)).decode()
        body = {"model": self.model, "max_tokens": tokens, "temperature": 0,
                "logprobs": True, "cache_salt": uuid.uuid4().hex,
                "chat_template_kwargs": {"thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": PROMPT}]}]}
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.load(response)["choices"][0]
        return {key: choice.get(key) for key in ("message", "logprobs", "finish_reason")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=96)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    alone = [client.describe(image, args.tokens) for image in IMAGES]
    together: list = [None] * len(IMAGES)

    def run(index):
        together[index] = client.describe(IMAGES[index], args.tokens)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(IMAGES))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    digests = {}
    for mode, results in (("alone", alone), ("c4", together)):
        if any(result is None for result in results):
            raise SystemExit(f"{mode}: a request returned nothing")
        digests[mode] = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()
        print(json.dumps({"mode": mode, "sha256": digests[mode],
                          "texts": [r["message"]["content"] for r in results]}), flush=True)
    print(json.dumps({"summary": "image digests", "tokens": args.tokens, "digests": digests}),
          flush=True)


if __name__ == "__main__":
    main()
