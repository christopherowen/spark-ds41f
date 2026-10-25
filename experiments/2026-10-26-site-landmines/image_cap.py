#!/usr/bin/env python3
"""Requests with many large images against a serving arm (a lab extras script).

usage: image_cap.py BASE_URL [--nodes dgx1,dgx2,dgx3] [--container spark-ds41f]
                    [--counts 4,8,16,32] [--streams 1,4,8] [--repeats 2] [--floor-gib 4.5]

A step sends STREAMS concurrent chat requests of COUNT images each. Every image is distinct (no
prefix or encoder cache hit) and is the processor's token-max square: 1,302 px, 994 tokens and
8,649 ViT patches, the most the processor encodes from one image (a larger source is shrunk to
it). Each is a PNG of about 0.5 MB, a 1,302 px photo's JPEG size; the server decodes every
encoding to the same 4.8 MiB of pixels. Per step:

- every node's MemAvailable low against the step's start (the head every 0.2 s, the others every
  second), and the head container's process RSS (API server, engine core, workers) every second;
- the KV cache usage peak (``/metrics`` every 0.5 s);
- each request's status, prompt tokens, upload time and TTFT (from the request's start);
- this client's own RSS, which the head's MemAvailable includes.

Steps run in order of images in flight (COUNT x STREAMS). Before a step, its drop is predicted
from the largest drop per image in flight measured so far, times 1.25. A step predicted to take a
node below the floor is not sent. A node falling below the floor during a step ends it: its
requests are cancelled and no larger step is sent. The floor, 4.5 GiB, is dgx1's 3 GiB guard plus
the 1.5 GiB margin the cap is chosen for, so the guard is never approached.

Last, one validation step (``--photos``, 8 at one stream): 12 MP sources (4,032 x 3,024, about
4.5 MB each), which the processor shrinks to the same tokens, so the step's extra API-server
memory over the token-max step is the decode cost the dgx4 bundle measured (bundles/
image-host-memory). It is gated the same way, with that decode cost added.

One JSON line a step, then a summary line. Exits 0 unless a request failed for a reason other
than a cancellation.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import shlex
import socket
import statistics
import struct
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

TOKEN_MAX = (1302, 1302)
PHOTO = (4032, 3024)
# One row in RANDOM_ROW is random bytes; the rest is a gradient that changes every BAND rows:
# about 0.57 MB a token-max PNG and 3.8 MB a 12 MP one, near a photo's JPEG at those sizes.
RANDOM_ROW = 10
BAND = 16
SHIFTS = [bytes((value + shift) % 256 for value in range(256)) for shift in range(256)]
SETTLE_GIB = 0.25
SETTLE_SECONDS = 90


def png(width: int, height: int, seed: int) -> bytes:
    """A distinct image compressed about like a photo's JPEG (no imaging library on the lab host)."""
    base = bytes((x * 255 // width + channel * 85 + seed) % 256 for x in range(width) for channel in range(3))
    rows = []
    for y in range(height):
        rows.append(b"\x00")
        rows.append(os.urandom(width * 3) if y % RANDOM_ROW == seed % RANDOM_ROW
                    else base.translate(SHIFTS[(y // BAND + seed) % 256]))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 6)) + chunk(b"IEND", b""))


def body(model: str, count: int, size: tuple[int, int], seed: int) -> bytes:
    """One chat request's JSON, built as bytes so each image is held once."""
    parts = []
    for index in range(count):
        data = base64.b64encode(png(*size, seed * 1000 + index))
        parts.append(b'{"type":"image_url","image_url":{"url":"data:image/png;base64,' + data + b'"}}')
    text = json.dumps({"type": "text", "text": f"Name the main colour of each of the {count} images, one word each."})
    head = json.dumps({"model": model, "max_tokens": 32, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False}})
    return (head[:-1].encode() + b',"messages":[{"role":"user","content":[' + b",".join(parts)
            + b"," + text.encode() + b"]}]}")


def own_rss_mib() -> float:
    with open("/proc/self/status") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    return 0.0


class Watch:
    """MemAvailable of every node and the head container's process RSS, until stopped."""

    def __init__(self, nodes: list[str], container: str, floor_gib: float):
        self.nodes, self.floor_kib = nodes, floor_gib * 2**20
        self.lock = threading.Lock()
        self.last: dict[str, int] = {}
        self.low: dict[str, int] = {}
        self.rss: dict[str, int] = {}
        self.rss_peak: dict[str, int] = {}
        self.breach: str | None = None
        self.on_breach = lambda: None
        self.table: dict[str, int] = {}
        self.processes = []
        for index, node in enumerate(nodes):
            interval = 0.2 if index == 0 else 1
            self._start(node, f"while :; do awk '/^MemAvailable:/{{print $2}}' /proc/meminfo; sleep {interval}; done",
                        lambda line, node=node: self._memory(node, line))
        self._start(nodes[0], f"while :; do echo T; docker top {shlex.quote(container)} -eo pid,rss,args; sleep 1; done",
                    self._process_line)

    def _start(self, node: str, script: str, handle) -> None:
        process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=5", node,
                                    shlex.join(["bash", "-c", script])],
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.processes.append(process)
        threading.Thread(target=lambda: [handle(line.strip()) for line in process.stdout], daemon=True).start()

    def _memory(self, node: str, line: str) -> None:
        if not line.isdigit():
            return
        value = int(line)
        with self.lock:
            self.last[node] = value
            self.low[node] = min(self.low.get(node, value), value)
            breached = value < self.floor_kib and self.breach is None
            if breached:
                self.breach = f"{node}: MemAvailable {value / 2**20:.2f} GiB below the floor"
        if breached:
            self.on_breach()

    def _process_line(self, line: str) -> None:
        if line == "T":
            with self.lock:
                self.rss = dict(self.table)
                for kind, value in self.rss.items():
                    self.rss_peak[kind] = max(self.rss_peak.get(kind, value), value)
            self.table = {}
            return
        fields = line.split(None, 2)
        if len(fields) < 3 or not fields[1].isdigit():
            return
        command = fields[2]
        kind = ("engine_core" if "EngineCore" in command else "workers" if "Worker" in command
                else "api_server" if ("APIServer" in command or " serve" in command) else "other")
        self.table[kind] = self.table.get(kind, 0) + int(fields[1])

    def ready(self, timeout: float = 30) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if all(node in self.last for node in self.nodes) and self.rss:
                    return
            time.sleep(0.5)
        raise SystemExit("memory telemetry did not start on every node")

    def mark(self) -> tuple[dict[str, int], dict[str, int]]:
        """Start a step: lows and RSS peaks from now; the current values."""
        with self.lock:
            self.low = dict(self.last)
            self.rss_peak = dict(self.rss)
            self.breach = None
            return dict(self.last), dict(self.rss)

    def stop(self) -> None:
        for process in self.processes:
            process.terminate()


class Metrics(threading.Thread):
    """The KV cache usage at a step's start and its peak, from /metrics every 0.5 s. A sample
    that cannot be read is counted, not retried."""

    def __init__(self, base: str):
        super().__init__(daemon=True)
        self.base, self.first, self.peak, self.unread = base, None, None, 0
        self.done = threading.Event()

    def usage(self) -> float | None:
        try:
            with urllib.request.urlopen(self.base + "/metrics", timeout=10) as response:
                for line in response.read().decode().splitlines():
                    if line.startswith("vllm:kv_cache_usage_perc"):
                        return float(line.rsplit(" ", 1)[1])
        except OSError:
            self.unread += 1
        return None

    def run(self) -> None:
        self.first = self.peak = self.usage()
        while not self.done.wait(0.5):
            value = self.usage()
            if value is not None:
                self.peak = value if self.peak is None else max(self.peak, value)


def send(base: str, payload: bytes, connections: list, lock: threading.Lock) -> dict:
    """One streamed chat request: status, upload time, TTFT and prompt tokens."""
    url = urllib.parse.urlparse(base)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=1800)
    with lock:
        connections.append(connection)
    record: dict = {}
    started = time.perf_counter()
    try:
        connection.request("POST", "/v1/chat/completions", body=payload,
                           headers={"Content-Type": "application/json"})
        record["upload_s"] = round(time.perf_counter() - started, 2)
        response = connection.getresponse()
        record["status"] = response.status
        if response.status != 200:
            record["error"] = response.read().decode(errors="replace")[:300]
            return record
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            if "ttft_s" not in record and (delta.get("content") or delta.get("reasoning_content")):
                record["ttft_s"] = round(time.perf_counter() - started, 2)
            if chunk.get("usage"):
                record["prompt_tokens"] = chunk["usage"]["prompt_tokens"]
        record["seconds"] = round(time.perf_counter() - started, 2)
    except (OSError, http.client.HTTPException, ValueError) as error:
        record["error"] = f"{type(error).__name__}: {error}"[:300]
    finally:
        connection.close()
    return record


def gib(kib: int) -> float:
    return round(kib / 2**20, 3)


def run_step(base: str, model: str, watch: Watch, count: int, streams: int, size: tuple[int, int],
             seed: int) -> dict:
    client = own_rss_mib()
    # zlib releases the GIL, so the requests' images compress in parallel.
    with ThreadPoolExecutor(max_workers=streams) as pool:
        payloads = list(pool.map(lambda stream: body(model, count, size, seed * 100 + stream), range(streams)))
    client_with_bodies = own_rss_mib()
    start, rss_start = watch.mark()
    connections, lock = [], threading.Lock()

    def cancel() -> None:
        with lock:
            for connection in connections:
                try:
                    connection.sock and connection.sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    watch.on_breach = cancel
    metrics = Metrics(base)
    metrics.start()
    results: list[dict] = [{} for _ in range(streams)]
    started = time.perf_counter()
    threads = [threading.Thread(target=lambda i=i: results.__setitem__(i, send(base, payloads[i], connections, lock)))
               for i in range(streams)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    seconds = time.perf_counter() - started
    metrics.done.set()
    metrics.join()
    watch.on_breach = lambda: None
    with watch.lock:
        low, rss_peak, breach = dict(watch.low), dict(watch.rss_peak), watch.breach
    ttfts = [r["ttft_s"] for r in results if "ttft_s" in r]
    return {
        "images": count, "streams": streams, "in_flight": count * streams, "size": f"{size[0]}x{size[1]}",
        "upload_mib_per_request": round(len(payloads[0]) / 2**20, 2), "seconds": round(seconds, 1),
        "start_gib": {node: gib(value) for node, value in start.items()},
        "low_gib": {node: gib(value) for node, value in low.items()},
        "drop_gib": {node: gib(start[node] - low[node]) for node in start},
        "head_rss_start_mib": {kind: round(value / 1024) for kind, value in rss_start.items()},
        "head_rss_peak_mib": {kind: round(value / 1024) for kind, value in rss_peak.items()},
        "kv_usage_start": metrics.first, "kv_usage_peak": metrics.peak, "kv_samples_unread": metrics.unread,
        "client_rss_mib": {"before": client, "with_bodies": client_with_bodies},
        "ttft_s": {"min": min(ttfts), "median": statistics.median(ttfts), "max": max(ttfts)} if ttfts else None,
        "prompt_tokens": sorted({r.get("prompt_tokens") for r in results if r.get("prompt_tokens")}),
        "statuses": sorted({r.get("status", 0) for r in results}),
        "errors": [r["error"] for r in results if "error" in r][:4],
        "halted": breach,
    }


def settle(watch: Watch, start: dict[str, int]) -> float:
    """Wait for the head to come back within SETTLE_GIB of the step's start."""
    head = watch.nodes[0]
    deadline = time.monotonic() + SETTLE_SECONDS
    while time.monotonic() < deadline:
        with watch.lock:
            if watch.last[head] >= start[head] - SETTLE_GIB * 2**20:
                break
        time.sleep(1)
    with watch.lock:
        return gib(watch.last[head])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("url")
    parser.add_argument("--nodes", default="dgx1,dgx2,dgx3", help="the head first")
    parser.add_argument("--container", default="spark-ds41f")
    parser.add_argument("--counts", default="4,8,16,32")
    parser.add_argument("--streams", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--floor-gib", type=float, default=4.5)
    parser.add_argument("--photos", type=int, default=8, help="images in the 12 MP validation step (0: none)")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.load(response)["data"][0]["id"]
    nodes = args.nodes.split(",")
    watch = Watch(nodes, args.container, args.floor_gib)
    watch.ready()
    plan = sorted(((count, streams) for count in map(int, args.counts.split(","))
                   for streams in map(int, args.streams.split(","))), key=lambda step: (step[0] * step[1], step[1]))
    # Largest drop per image in flight so far, per node (GiB).
    per_image: dict[str, float] = {}
    halted, failed, seed = None, False, 0

    def predicted_low(in_flight: int, extra_gib: float = 0.0) -> dict[str, float]:
        with watch.lock:
            now = {node: watch.last[node] / 2**20 for node in nodes}
        return {node: round(now[node] - 1.25 * in_flight * (per_image.get(node, 0.0) + extra_gib), 3)
                for node in nodes}

    warmup = run_step(base, model, watch, 1, 1, TOKEN_MAX, seed)
    print(json.dumps({"step": "warmup", **warmup}), flush=True)
    if warmup["halted"]:
        halted = "warmup"
    settle(watch, {node: int(value * 2**20) for node, value in warmup["start_gib"].items()})
    steps = [(count, streams, TOKEN_MAX, 0.0) for count, streams in plan]
    measured: list[dict] = []
    if args.photos:
        # The 12 MP decode on top of the token-max step: RGB pixels plus one conversion copy.
        steps.append((args.photos, 1, PHOTO, PHOTO[0] * PHOTO[1] * 6 / 2**30))
    for count, streams, size, extra in steps:
        for repeat in range(args.repeats if size == TOKEN_MAX else 1):
            seed += 1
            entry = {"step": f"{count}x{streams}" + ("" if size == TOKEN_MAX else "-photo"), "repeat": repeat}
            if halted is not None:
                print(json.dumps({**entry, "skipped": f"after {halted}"}), flush=True)
                continue
            # Before any step has run, the prediction is the nodes' current reading.
            prediction = predicted_low(count * streams, extra)
            if min(prediction.values()) < args.floor_gib:
                print(json.dumps({**entry, "skipped": "predicted below the floor", "predicted_low_gib": prediction}),
                      flush=True)
                continue
            record = run_step(base, model, watch, count, streams, size, seed)
            record["predicted_low_gib"] = prediction
            record["settled_gib"] = settle(watch, {node: int(value * 2**20) for node, value in record["start_gib"].items()})
            print(json.dumps({**entry, **record}), flush=True)
            measured.append(record)
            for node, drop in record["drop_gib"].items():
                per_image[node] = max(per_image.get(node, 0.0), max(drop, 0.0) / (count * streams))
            if record["halted"]:
                halted = entry["step"]
            elif record["errors"] or record["statuses"] != [200]:
                failed = True
    watch.stop()
    # Per image count at the token-max size: the streams it ran at and the head's lowest reading.
    by_count: dict[int, dict] = {}
    for record in measured:
        if record["size"] != f"{TOKEN_MAX[0]}x{TOKEN_MAX[1]}":
            continue
        entry = by_count.setdefault(record["images"], {"streams": [], "head_low_gib": None, "ttft_max_s": None})
        entry["streams"] = sorted({*entry["streams"], record["streams"]})
        low = record["low_gib"][nodes[0]]
        entry["head_low_gib"] = low if entry["head_low_gib"] is None else min(entry["head_low_gib"], low)
        if record["ttft_s"]:
            entry["ttft_max_s"] = max(entry["ttft_max_s"] or 0, record["ttft_s"]["max"])
    print(json.dumps({"summary": True, "floor_gib": args.floor_gib, "halted_at": halted, "by_count": by_count,
                      "per_image_drop_gib": {node: round(value, 4) for node, value in per_image.items()}}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
