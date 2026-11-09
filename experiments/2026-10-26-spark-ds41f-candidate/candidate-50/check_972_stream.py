#!/usr/bin/env python3
"""#972 live check: a client disconnect mid-stream aborts the engine request, SSE keep-alive
comment frames arrive about every 15 s on a slow stream, and servebench's stream parsing
still works with those frames. A lab extra against a served endpoint, on an idle server.

usage: check_972_stream.py URL [--container NAME] [--interval 15] [--prompt-words N]
                           [--min-frames 2] [--parts disconnect,keepalive]

Parts, in order (each prints one JSON line):
- `disconnect`: waits until the server runs and queues nothing, then streams a chat request
  that cannot finish early (`ignore_eos`, 8192 tokens, temperature 0, thinking off), reads
  32 content chunks and closes the socket. It then reads URL/metrics every 0.25 s.
  Required:
  - `vllm:generation_tokens_total` stops growing (unchanged for 3 s) within 30 s;
  - the request generated fewer tokens than it asked for;
  - the server still answers a short request.
  Recorded beside: the seconds to quiet, the tokens generated after the close, the change in
  `vllm:request_success_total{finished_reason="abort"}`, and with --container the serving
  container's "Aborted request" log lines since the start (they need request logging).
- `keepalive`: one slow stream through servebench's own collector (servebench.client.stream_chat,
  as every bench step uses it), its bytes recorded on the way in. The slow part is the prefill
  of a fresh prompt of N random words (default 500,000, about 0.5-0.65M tokens, under the
  recipe's 1M window), so no output is sent for tens of seconds. Required:
  - at least --min-frames keep-alive comments before the first data frame;
  - the first one --interval s after the request (up to 10 s later: the server's timer starts
    once it has tokenized the prompt), then one every --interval s (2 s tolerance);
  - servebench's record has no error, a first output after the last of those comments, the
    usage counts, and one event for every data frame that carries output.
  A prefill shorter than min-frames x interval cannot show the cadence; the part then fails
  with `slow_enough` false (raise --prompt-words).

Prints a summary line ("passed") last; exits 1 when a required part fails or the endpoint does
not answer, 0 otherwise. Imports servebench from this checkout; otherwise stdlib only.
"""
import argparse
import datetime
import http.client
import json
import random
import socket
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from servebench import client as servebench_client  # noqa: E402

GENERATED = "vllm:generation_tokens_total"
WORDS = ("time year people way day man thing woman life child world school state family student group "
         "country problem hand part place case week company system program question work government number "
         "night point home water room mother area money story fact month lot right study book eye job word "
         "business issue side kind head house service friend father power hour game line end member law car "
         "city community name president team minute idea kid body information back parent face others level "
         "office door health person art war history party result change morning reason research girl guy "
         "moment air teacher force education river garden window paper music market table summer bridge "
         "valley engine signal harbor island forest letter season planet").split()


def metrics(base: str) -> tuple[dict[str, float], float]:
    """Each series summed over its labels, and the abort-finished request count."""
    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        lines = response.read().decode().splitlines()
    totals, aborted = {}, 0.0
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        totals[name] = totals.get(name, 0.0) + value
        if name == "vllm:request_success_total" and 'finished_reason="abort"' in line:
            aborted += value
    return totals, aborted


def busy(totals: dict[str, float]) -> float:
    return totals.get("vllm:num_requests_running", 0.0) + totals.get("vllm:num_requests_waiting", 0.0)


def wait_idle(base: str, seconds: float) -> str | None:
    """How the server was seen idle: its running and waiting gauges at zero, or its token
    counters unchanged for 10 s (the gauges are refreshed only by an engine step, so an abort
    with nothing else running leaves them stale until the next request). None: busy throughout."""
    deadline = time.monotonic() + seconds
    still, since = None, time.monotonic()
    while time.monotonic() < deadline:
        totals = metrics(base)[0]
        if not busy(totals):
            return "gauges"
        counters = (totals.get(GENERATED), totals.get("vllm:prompt_tokens_total"))
        if counters != still:
            still, since = counters, time.monotonic()
        elif time.monotonic() - since >= 10:
            return "counters still"
        time.sleep(1)
    return None


def chat(base: str, body: dict, timeout: float = 300) -> dict:
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def aborted_log_lines(container: str, since: str) -> int | str:
    process = subprocess.run(["docker", "logs", "--since", since, container], capture_output=True, text=True)
    if process.returncode:
        return f"docker logs failed: {process.stderr.strip()[:160]}"
    return sum("Aborted request" in line for line in (process.stdout + process.stderr).splitlines())


def disconnect(base: str, model: str, container: str | None) -> dict:
    result = {"part": "disconnect", "passed": False, "problems": []}
    since = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result["idle"] = wait_idle(base, 120)
    if result["idle"] is None:
        result["problems"].append("the server did not go idle within 120 s")
        return result
    totals, aborted_before = metrics(base)
    generated_before = totals.get(GENERATED, 0.0)
    max_tokens = 8192
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "ignore_eos": True, "stream": True,
            "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "Count from 1 to 100000, one number per line."}]}
    url = urllib.parse.urlsplit(base)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=300)
    connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    chunks = 0
    while chunks < 32:
        line = response.readline()
        if not line:
            result["problems"].append(f"the stream ended after {chunks} chunks, before the disconnect")
            return result
        if line.startswith(b"data: ") and not line.startswith(b"data: [DONE]"):
            for choice in json.loads(line[6:]).get("choices", []):
                if (choice.get("delta") or {}).get("content"):
                    chunks += 1
    result["running_before_close"] = metrics(base)[0].get("vllm:num_requests_running")
    closed = time.monotonic()
    generated_at_close = metrics(base)[0].get(GENERATED, 0.0)
    try:
        connection.sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    connection.close()
    # Quiet: the generated-token counter unchanged for 3 s. A request the abort missed keeps decoding.
    last, last_change, quiet_at = generated_at_close, closed, None
    while time.monotonic() - closed < 30:
        time.sleep(0.25)
        value = metrics(base)[0].get(GENERATED, 0.0)
        now = time.monotonic()
        if value != last:
            last, last_change = value, now
        elif now - last_change >= 3:
            quiet_at = last_change
            break
    totals, aborted_after = metrics(base)
    generated = totals.get(GENERATED, 0.0) - generated_before
    result.update(chunks_read=chunks, generated_tokens=int(generated),
                  tokens_after_close=int(totals.get(GENERATED, 0.0) - generated_at_close),
                  seconds_to_quiet=round(quiet_at - closed, 3) if quiet_at is not None else None,
                  running_after=totals.get("vllm:num_requests_running"),
                  abort_finished_delta=aborted_after - aborted_before)
    if quiet_at is None:
        result["problems"].append("generation did not stop within 30 s of the disconnect")
    if generated >= max_tokens:
        result["problems"].append(f"the request ran to its {max_tokens} tokens")
    try:
        reply = chat(base, {"model": model, "temperature": 0, "max_tokens": 8, "reasoning_effort": "none",
                            "messages": [{"role": "user", "content": "Say OK."}]})
        result["after"] = reply["choices"][0]["message"].get("content")
    except (OSError, ValueError, KeyError, IndexError) as error:
        result["problems"].append(f"the server did not answer after the disconnect: {error}")
    if container:
        result["aborted_log_lines"] = aborted_log_lines(container, since)
    result["passed"] = not result["problems"]
    return result


class Tee:
    """An urlopen response that records each line it yields, with its arrival time."""

    def __init__(self, response, lines: list):
        self.response, self.lines = response, lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.response.close()

    def __iter__(self):
        for raw in self.response:
            self.lines.append((time.monotonic(), raw))
            yield raw


def keepalive(base: str, model: str, interval: float, words: int, min_frames: int) -> dict:
    result = {"part": "keepalive", "passed": False, "problems": [], "interval": interval}
    result["idle"] = wait_idle(base, 120)
    if result["idle"] is None:
        result["problems"].append("the server did not go idle within 120 s")
        return result
    rng = random.Random(time.time_ns())
    text = f"Note {rng.getrandbits(64):016x}. " + " ".join(rng.choice(WORDS) for _ in range(words))
    payload = {"model": model, "temperature": 0, "max_tokens": 16, "reasoning_effort": "none",
               "messages": [{"role": "user", "content": text + "\n\nReply with the single word OK."}]}
    lines: list = []
    opened = urllib.request.urlopen
    urllib.request.urlopen = lambda *a, **k: Tee(opened(*a, **k), lines)
    try:
        record = servebench_client.stream_chat(base, payload, clock=time.monotonic)
    finally:
        urllib.request.urlopen = opened
    sent = record["sent"]
    frames = [at - sent for at, raw in lines if raw.strip().startswith(b":")]
    data = []
    for at, raw in lines:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        if any(isinstance((c.get("delta") or {}).get(key), str) and (c.get("delta") or {}).get(key)
               for c in event.get("choices") or [] for key in ("content", "reasoning_content", "reasoning")) \
                or any(c.get("token_ids") for c in event.get("choices") or []):
            data.append(at - sent)
    first = data[0] if data else None
    before = [f for f in frames if first is None or f < first]
    gaps = [round(b - a, 3) for a, b in zip([0.0, *before], before)]
    slow_enough = first is not None and first > min_frames * interval + 2
    result.update(prompt_tokens=record["prompt_tokens"], completion_tokens=record["completion_tokens"],
                  first_output_s=round(first, 3) if first is not None else None,
                  frames_before_output=[round(f, 3) for f in before], gaps=gaps,
                  frames_after_output=len(frames) - len(before), slow_enough=slow_enough,
                  servebench={"error": record["error"], "events": len(record["events"]),
                              "first_s": round(record["first"] - sent, 3) if record["first"] else None,
                              "finish_reason": record["finish_reason"], "token_times": record["token_times"]})
    if not slow_enough:
        result["problems"].append(f"first output at {first} s: too soon for {min_frames} frames of {interval} s")
    if len(before) < min_frames:
        result["problems"].append(f"{len(before)} keep-alive frames before the first output, expected {min_frames}")
    # The server's timer starts once it has rendered and tokenized the prompt, a few seconds
    # after the request was sent for this size; the frames after the first keep its cadence.
    if gaps and not interval - 1 <= gaps[0] <= interval + 10:
        result["problems"].append(f"the first keep-alive frame came {gaps[0]} s after the request, not {interval} s")
    if any(abs(gap - interval) > 2 for gap in gaps[1:]):
        result["problems"].append(f"keep-alive gaps {gaps[1:]} are not {interval} s")
    if record["error"] is not None:
        result["problems"].append(f"servebench: {record['error']}")
    if record["first"] is None or (before and record["first"] - sent <= before[-1]):
        result["problems"].append("servebench took a keep-alive frame for output")
    if record["prompt_tokens"] is None or not record["completion_tokens"]:
        result["problems"].append("servebench read no usage")
    if len(record["events"]) != len(data):
        result["problems"].append(f"servebench counted {len(record['events'])} output events of {len(data)}")
    result["passed"] = not result["problems"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--model")
    parser.add_argument("--container", help="the serving container on this host, for its abort log lines")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--prompt-words", type=int, default=500_000)
    parser.add_argument("--min-frames", type=int, default=2)
    parser.add_argument("--parts", default="disconnect,keepalive")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    label = "#972 disconnect abort and keep-alive"
    try:
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            model = args.model or json.load(response)["data"][0]["id"]
    except OSError as error:
        print(json.dumps({"summary": label, "passed": False, "error": f"endpoint did not answer: {error}"}))
        return 1
    parts = {}
    for part in args.parts.split(","):
        try:
            if part == "disconnect":
                parts[part] = disconnect(base, model, args.container)
            elif part == "keepalive":
                parts[part] = keepalive(base, model, args.interval, args.prompt_words, args.min_frames)
            else:
                raise SystemExit(f"unknown part {part}")
        except (OSError, ValueError, KeyError) as error:
            parts[part] = {"part": part, "passed": False, "problems": [f"{type(error).__name__}: {error}"]}
        print(json.dumps(parts[part]), flush=True)
    passed = all(p["passed"] for p in parts.values())
    print(json.dumps({"summary": label, "passed": passed, "model": model,
                      "parts": {name: p["passed"] for name, p in parts.items()},
                      "failing": [f"{name}: {problem}" for name, p in parts.items() for problem in p["problems"]]}),
          flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
