#!/usr/bin/env python3
"""A long-context agent soak for host-side memory growth (#439).

usage (a lab extra, run on the head node from the repository root):
  soak.py HEAD_URL [--minutes 75] [--sessions 4] [--chat-streams 8] [--seed 0]

#439 reports the API server growing 1.68 -> 2.44 GiB and the engine core 1.43 -> 1.97 GiB
over three hours of ordinary 1M-context use. This drives the host side the way coding agents
do, harder: ``sessions`` conversations each resend their whole history every turn (the API
server parses, renders and tokenizes it again; the engine core hashes and schedules it),
grow by a real-text tool result of ``--turn-tokens`` per turn, and start over with a fresh
document once they pass ``--max-context``. ``chat-streams`` short requests run beside them.
Every ``--report-seconds`` it prints one JSON line of counters; the memory itself is read
on the nodes by sampler.py.
"""
import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from servebench import prompts  # noqa: E402

MODEL = "deepseek-v4.1-flash"


def post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


class Counters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.values = {"agent_turns": 0, "agent_resets": 0, "agent_prompt_tokens": 0, "agent_cached_tokens": 0,
                       "chat_requests": 0, "completion_tokens": 0, "errors": 0, "largest_prompt": 0}
        self.last_error = ""

    def add(self, **changes) -> None:
        with self.lock:
            for name, value in changes.items():
                if name == "largest_prompt":
                    self.values[name] = max(self.values[name], value)
                else:
                    self.values[name] += value

    def error(self, text: str) -> None:
        with self.lock:
            self.values["errors"] += 1
            self.last_error = text[:300]

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.values, "last_error": self.last_error}


def agent(url: str, index: int, args, counters: Counters, deadline: float) -> None:
    rng = random.Random(args.seed * 1000 + index)
    generation = 0
    while time.time() < deadline:
        generation += 1
        seed = args.seed * 100000 + index * 1000 + generation
        messages = [{"role": "user", "content": "You are reviewing a codebase. Here is the first file.\n\n"
                     + prompts.source_text(args.first_tokens, seed) + "\n\nSummarize what it does in two sentences."}]
        context = args.first_tokens
        turn = 0
        while time.time() < deadline and context < args.max_context:
            body = {"model": MODEL, "messages": messages, "max_tokens": args.output_tokens,
                    "chat_template_kwargs": {"thinking": False}}
            try:
                reply = post(url, body, timeout=1800)
            except (urllib.error.URLError, OSError, ValueError) as error:
                counters.error(f"agent {index}: {error}")
                time.sleep(5)
                break
            usage = reply.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            prompt_tokens = usage.get("prompt_tokens", 0)
            counters.add(agent_turns=1, agent_prompt_tokens=prompt_tokens,
                         agent_cached_tokens=details.get("cached_tokens") or 0,
                         completion_tokens=usage.get("completion_tokens", 0), largest_prompt=prompt_tokens)
            content = reply["choices"][0]["message"].get("content") or ""
            turn += 1
            tool = prompts.source_text(rng.randint(args.turn_tokens // 2, args.turn_tokens * 3 // 2), seed * 100 + turn)
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": f"Tool output (read_file, turn {turn}):\n\n{tool}\n\n"
                                                     "What changed in your understanding? Answer briefly."}]
            context = prompt_tokens + args.turn_tokens + args.output_tokens
        counters.add(agent_resets=1)


def chat(url: str, index: int, args, counters: Counters, deadline: float) -> None:
    pool = prompts.load("default")
    rng = random.Random(args.seed * 7 + index)
    while time.time() < deadline:
        body = {"model": MODEL, "messages": [{"role": "user", "content": rng.choice(pool)["text"]}],
                "max_tokens": args.chat_tokens}
        try:
            reply = post(url, body, timeout=600)
            counters.add(chat_requests=1, completion_tokens=(reply.get("usage") or {}).get("completion_tokens", 0))
        except (urllib.error.URLError, OSError, ValueError) as error:
            counters.error(f"chat {index}: {error}")
            time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--minutes", type=float, default=75)
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--chat-streams", type=int, default=8)
    parser.add_argument("--first-tokens", type=int, default=65536)
    parser.add_argument("--turn-tokens", type=int, default=32768)
    parser.add_argument("--max-context", type=int, default=960000)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--chat-tokens", type=int, default=384)
    parser.add_argument("--report-seconds", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    start = time.time()
    deadline = start + args.minutes * 60
    counters = Counters()
    threads = [threading.Thread(target=agent, args=(url, i, args, counters, deadline), daemon=True)
               for i in range(args.sessions)]
    threads += [threading.Thread(target=chat, args=(url, i, args, counters, deadline), daemon=True)
                for i in range(args.chat_streams)]
    print(json.dumps({"event": "start", "t": start, "args": vars(args)}), flush=True)
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        time.sleep(min(args.report_seconds, max(1.0, deadline + 5 - time.time())))
        print(json.dumps({"event": "progress", "t": time.time(), "minutes": round((time.time() - start) / 60, 1),
                          **counters.snapshot()}), flush=True)
        if time.time() > deadline + 1800:  # a request stuck past its own timeout
            print(json.dumps({"event": "abandoned", "alive": sum(t.is_alive() for t in threads)}), flush=True)
            break
    summary = counters.snapshot()
    print(json.dumps({"event": "done", "t": time.time(), **summary}), flush=True)
    sys.exit(1 if summary["agent_turns"] == 0 or summary["errors"] > summary["agent_turns"] // 10 else 0)


if __name__ == "__main__":
    main()
