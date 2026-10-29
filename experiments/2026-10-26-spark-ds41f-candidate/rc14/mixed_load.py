#!/usr/bin/env python3
"""Long-context agents beside short chats, the load of spark-ds41f #677.

usage: mixed_load.py BASE_URL [--seconds S] [--agents N] [--chats N] [--no-resend]

Each agent resends its whole history plus about --fresh new salted tokens a turn,
starting at --agent-start and starting over past --agent-max; each chat stream asks short
salted questions one after another, and (unless --no-resend) asks each question twice: the
second is cached up to its last token, so the scheduler admits it with one token left,
padded to the verification width with placeholder drafts. Every request has max_tokens
256 at temperature 0.

One JSON line: requests, failures (the first's time and error), the largest agent prompt
and output tok/s. Exits 1 when any request failed: the engine died or refused it.
"""
import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request

WORDS = ("time year people way day man thing woman life child world school state family "
         "student group country problem hand part place case week company system program "
         "question work government number night point home water room mother area money "
         "story fact month lot right study book eye job word business issue side kind head "
         "house service friend father power hour game line end member law car city name "
         "team minute idea kid body information back parent face others level office door "
         "health person art war history party result change morning reason research girl "
         "guy moment air teacher force education").split()


def text(rng: random.Random, words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(words))


class Load:
    def __init__(self, url: str, deadline: float):
        self.url, self.deadline, self.start = url.rstrip("/"), deadline, time.time()
        self.model = json.load(urllib.request.urlopen(f"{self.url}/v1/models", timeout=30))["data"][0]["id"]
        self.lock = threading.Lock()
        self.requests = self.output_tokens = self.max_prompt = 0
        self.failures: list[dict] = []

    def ask(self, messages: list[dict]) -> str | None:
        body = json.dumps({"model": self.model, "messages": messages, "max_tokens": 256,
                           "temperature": 0}).encode()
        request = urllib.request.Request(f"{self.url}/v1/chat/completions", body,
                                         {"Content-Type": "application/json"})
        try:
            reply = json.load(urllib.request.urlopen(request, timeout=1800))
        except (urllib.error.URLError, OSError, ValueError) as error:
            with self.lock:
                self.failures.append({"t_s": round(time.time() - self.start, 1), "error": str(error)[:300]})
            return None
        usage = reply.get("usage", {})
        with self.lock:
            self.requests += 1
            self.output_tokens += usage.get("completion_tokens", 0)
            self.max_prompt = max(self.max_prompt, usage.get("prompt_tokens", 0))
        return reply["choices"][0]["message"]["content"] or ""

    def running(self) -> bool:
        return time.time() < self.deadline and not self.failures


def agent(load: Load, seed: int, start: int, fresh: int, most: int) -> None:
    rng = random.Random(seed)
    history: list[dict] = []
    words = start
    while load.running():
        if words >= most or not history:
            history, words = [{"role": "user", "content": f"[{rng.random()}] " + text(rng, start)}], start
        else:
            history.append({"role": "user", "content": f"[{rng.random()}] " + text(rng, fresh)})
            words += fresh
        answer = load.ask(history)
        if answer is None:
            return
        history.append({"role": "assistant", "content": answer})


def chat(load: Load, seed: int, resend: bool) -> None:
    rng = random.Random(seed)
    while load.running():
        question = [{"role": "user", "content": f"[{rng.random()}] Summarize: " + text(rng, 60)}]
        for _ in range(2 if resend else 1):
            if load.ask(question) is None:
                return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--chats", type=int, default=8)
    parser.add_argument("--agent-start", type=int, default=50_000, help="words (about a token each)")
    parser.add_argument("--fresh", type=int, default=20_000, help="new words an agent turn")
    parser.add_argument("--agent-max", type=int, default=300_000, help="words before an agent starts over")
    parser.add_argument("--no-resend", dest="resend", action="store_false")
    args = parser.parse_args()
    load = Load(args.base_url, time.time() + args.seconds)
    threads = [threading.Thread(target=agent, args=(load, i, args.agent_start, args.fresh, args.agent_max))
               for i in range(args.agents)]
    threads += [threading.Thread(target=chat, args=(load, 1000 + i, args.resend)) for i in range(args.chats)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.time() - load.start
    print(json.dumps({"summary": "mixed load", "seconds": round(elapsed, 1), "resend": args.resend,
                      "requests": load.requests, "failed": len(load.failures),
                      "first_failure": load.failures[0] if load.failures else None,
                      "max_prompt_tokens": load.max_prompt,
                      "output_tok_s": round(load.output_tokens / elapsed, 1)}), flush=True)
    return 1 if load.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
