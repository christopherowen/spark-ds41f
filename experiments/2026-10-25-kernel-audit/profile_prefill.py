#!/usr/bin/env python3
"""Profile a long prompt's prefill and an image request (the vision tower), one
torch-profiler window apiece (a profiler-enabled arm must be serving).

usage: profile_prefill.py BASE_URL [--tokens N]

The long prompt is about N tokens of numbered sentences (fresh cache salt, a few output
tokens); the image request sends one generated 448x448 PNG with a question. Each is warmed
once, then profiled; one JSON line a scenario in profiling order. The profiled image differs
from the warm-up's (another grid), so the encoder cache misses and the profile holds the
vision tower.
"""
import base64
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib

BASE = sys.argv[1].rstrip("/")
TOKENS = int(sys.argv[sys.argv.index("--tokens") + 1]) if "--tokens" in sys.argv else 16384


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        text = response.read()
        return json.loads(text) if text.startswith(b"{") else None


def png(width=448, height=448, cell=64):
    """A colour gradient with a grid of ``cell`` pixels, as a PNG (no imaging library
    needed)."""
    rows = bytearray()
    for y in range(height):
        rows.append(0)
        for x in range(width):
            grid = 255 if (x % cell == 0 or y % cell == 0) else 0
            rows += bytes((x * 255 // width, y * 255 // height, grid))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(rows)))
            + chunk(b"IEND", b""))


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]
LONG = " ".join(f"Sentence {i} says that the river bends {i % 7} times before the sea." for i in range(TOKENS // 16))


def image_message(cell):
    url = "data:image/png;base64," + base64.b64encode(png(cell=cell)).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                         {"type": "text", "text": "Describe this image in one sentence."}]}]


# name: (warm-up messages, profiled messages)
SCENARIOS = {
    "long-prefill": ([{"role": "user", "content": LONG + "\nHow many sentences mention the sea?"}],) * 2,
    "image": (image_message(64), image_message(56)),
}


def chat(messages):
    return post("/v1/chat/completions", {"model": MODEL, "messages": messages, "max_tokens": 16, "temperature": 0,
                                         "chat_template_kwargs": {"thinking": False},
                                         "cache_salt": uuid.uuid4().hex})


for name, (warm, messages) in SCENARIOS.items():
    try:
        chat(warm)
    except urllib.error.HTTPError as error:
        print(json.dumps({"scenario": name, "status": error.code,
                          "message": error.read().decode(errors="replace")[:300]}), flush=True)
        continue
    post("/start_profile")
    try:
        result = chat(messages)
    finally:
        post("/stop_profile")
        time.sleep(30)
    print(json.dumps({"scenario": name, "status": 200, "prompt_tokens": result["usage"]["prompt_tokens"],
                      "completion_tokens": result["usage"]["completion_tokens"]}), flush=True)
