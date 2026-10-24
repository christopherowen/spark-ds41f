#!/usr/bin/env python3
"""Prompt logprobs on the serving stack: the request that killed candidate 4's engine, the
logprobs it returns, and the temperature-0 outputs around it.

usage: prompt_logprobs_check.py BASE_URL --label LABEL [--reference-labels A,B]
                                [--prompt-logprobs N] [--tokens T] [--seed S]

The probe is the request of run tilelang-ports-controls (2026-10-24 22:34 CET): a chat
completion of controls_digest.py's first prompt with thinking off, temperature 0,
``logprobs`` and ``prompt_logprobs`` N. The other seven prompts run beside it. Steps:

- others: the seven prompts one at a time;
- probe: the probe without prompt logprobs;
- probe-prompt-logprobs: the probe with them;
- echo: the probe with ``echo``, which asks for prompt logprobs too (``top_logprobs`` of
  them, 0 by default);
- together: the seven and the probe with prompt logprobs, all at once;
- teacher: a completion of the probe's prompt tokens followed by its reply tokens, with
  prompt logprobs and one token: its logprobs score the probe's prompt again and each
  token the probe chose, as the probe's own logprobs did.

A reference arm (one of ``--reference-labels``) has no fix: it sends the probe and the
others without prompt logprobs (the engine dies on them), so its digests are what the
fix arm must reproduce.

Each step prints one JSON line; the summary line holds the digests of the seven, of the
probe's reply, the prompt-logprob checks and ``pass``. Hard checks, all in this arm:
every request returns HTTP 200; one prompt logprob entry per prompt token, the first
null, each holding its prompt token and the top N with ranks 1..N; the probe's reply
the same with and without prompt logprobs, and alone or beside the seven; the echo
request answered with the probe's tokens; the seven's replies the same alone and beside
the probe. The teacher's agreement (ranks and largest
logprob gaps) is recorded, not checked: it scores the tokens in another forward shape.
Exits 1 when a hard check fails, so the lab keeps the engine logs.
"""
import argparse
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
import uuid

# controls_digest.py's prompts (token_determinism.py's eight); the first is the probe.
PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
    "Give a detailed recipe for lentil soup with timings and substitutions.",
    "Summarize the causes and consequences of the printing press in Europe.",
    "Explain the difference between supervised and unsupervised learning to a student.",
    "Plan a three-day hiking trip in the Alps with packing advice.",
]


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return {"status": response.status, "body": json.load(response)}
        except urllib.error.HTTPError as error:
            return {"status": error.code, "body": error.read().decode(errors="replace")}
        except (urllib.error.URLError, ConnectionError) as error:
            # A dead engine's server closes or refuses the connection.
            return {"status": 0, "body": repr(error)}

    def chat(self, prompt: str, tokens: int, seed: int, prompt_logprobs: int | None,
             **extra) -> dict:
        # controls_digest.py's request, with token ids returned.
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "seed": seed, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "logprobs": True, "temperature": 0.0,
                "chat_template_kwargs": {"thinking": False}, "return_token_ids": True,
                **extra}
        if prompt_logprobs:
            body["prompt_logprobs"] = prompt_logprobs
        return self.post("/v1/chat/completions", body)


def reply(result: dict) -> dict | None:
    """The sampled reply: message, logprobs, finish reason and token ids."""
    if result["status"] != 200:
        return None
    choice = result["body"]["choices"][0]
    return {key: choice.get(key) for key in ("message", "logprobs", "finish_reason", "token_ids")}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def together(calls: list) -> list:
    results: list = [None] * len(calls)

    def run(index):
        results[index] = calls[index]()

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


def check_prompt_logprobs(entries, prompt_ids: list[int], n: int) -> list[str]:
    """The problems of a prompt-logprob list: one entry per prompt token, the first
    null, each holding its prompt token and the top n with ranks 1..n."""
    if entries is None:
        return ["no prompt_logprobs"]
    problems = []
    if len(entries) != len(prompt_ids):
        problems.append(f"{len(entries)} entries for {len(prompt_ids)} prompt tokens")
    if entries and entries[0] is not None:
        problems.append("the first entry is not null")
    for position, (entry, token) in enumerate(zip(entries[1:], prompt_ids[1:]), 1):
        if entry is None or str(token) not in entry:
            problems.append(f"position {position}: prompt token {token} missing")
            continue
        # The top n, and the prompt token at its own rank when it is not among them.
        top = sorted(lp["rank"] for t, lp in entry.items()
                     if not (t == str(token) and lp["rank"] > n))
        if top != list(range(1, n + 1)):
            problems.append(f"position {position}: top ranks {top}")
    return problems[:10]


def teacher_agreement(probe_entries, probe_reply: dict, teacher_entries) -> dict:
    """How the teacher's prompt logprobs score the probe's prompt and its chosen tokens,
    against the probe's prompt logprobs and its reply's logprobs. At temperature 0 a
    chosen token is the top one unless sink rejection replaced it."""
    prompt_len = len(probe_entries)
    prompt_gap, prompt_rank_mismatches = 0.0, 0
    for mine, theirs in zip(probe_entries[1:], teacher_entries[1:prompt_len]):
        for token, lp in mine.items():
            if token in theirs:
                prompt_gap = max(prompt_gap, abs(lp["logprob"] - theirs[token]["logprob"]))
                prompt_rank_mismatches += lp["rank"] != theirs[token]["rank"]
    reply_gap, not_top, scored = 0.0, [], 0
    content = probe_reply["logprobs"]["content"]
    for i, (token, item) in enumerate(zip(probe_reply["token_ids"], content)):
        entry = teacher_entries[prompt_len + i]
        if entry is None or str(token) not in entry:
            continue
        scored += 1
        reply_gap = max(reply_gap, abs(item["logprob"] - entry[str(token)]["logprob"]))
        if entry[str(token)]["rank"] != 1:
            not_top.append(i)
    return {"prompt_max_gap": round(prompt_gap, 5), "prompt_rank_mismatches": prompt_rank_mismatches,
            "reply_tokens_scored": scored, "reply_max_gap": round(reply_gap, 5),
            "reply_positions_not_top": not_top}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference-labels", default="")
    parser.add_argument("--prompt-logprobs", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    seeds = [args.seed + i for i in range(len(PROMPTS))]
    probe, others = (PROMPTS[0], seeds[0]), list(zip(PROMPTS[1:], seeds[1:]))
    reference = args.label in args.reference_labels.split(",")
    n = None if reference else args.prompt_logprobs
    failures: list[str] = []

    def emit(record: dict) -> None:
        print(json.dumps(record), flush=True)

    def require(condition: bool, what: str) -> None:
        if not condition:
            failures.append(what)

    alone = [client.chat(p, args.tokens, s, None) for p, s in others]
    require(all(r["status"] == 200 for r in alone), "others: a request failed")
    others_digest = digest([reply(r) for r in alone])
    emit({"step": "others", "statuses": [r["status"] for r in alone], "sha256": others_digest})

    plain = client.chat(probe[0], args.tokens, probe[1], None)
    require(plain["status"] == 200, "probe: the request failed")
    probe_digest = digest(reply(plain))
    emit({"step": "probe", "status": plain["status"], "sha256": probe_digest})

    summary = {"summary": "prompt logprobs", "label": args.label, "reference": reference,
               "prompt_logprobs": n, "tokens": args.tokens, "seed": args.seed,
               "others_sha256": others_digest, "probe_sha256": probe_digest}
    calls = [lambda p=p, s=s: client.chat(p, args.tokens, s, None) for p, s in others]
    calls.append(lambda: client.chat(probe[0], args.tokens, probe[1], n))
    if not reference:
        with_lp = client.chat(probe[0], args.tokens, probe[1], n)
        status = with_lp["status"]
        emit({"step": "probe-prompt-logprobs", "status": status,
              "body": None if status == 200 else with_lp["body"][:2000]})
        require(status == 200, "probe with prompt logprobs: the request failed")
        if status == 200:
            body = with_lp["body"]
            prompt_ids, entries = body["prompt_token_ids"], body["prompt_logprobs"]
            problems = check_prompt_logprobs(entries, prompt_ids, n)
            require(not problems, f"probe prompt logprobs: {problems}")
            require(digest(reply(with_lp)) == probe_digest,
                    "probe: prompt logprobs changed its reply")
            summary.update(prompt_tokens=len(prompt_ids),
                           prompt_logprobs_sha256=digest(entries),
                           prompt_logprob_problems=problems)
        echo = client.chat(probe[0], args.tokens, probe[1], None, echo=True)
        emit({"step": "echo", "status": echo["status"],
              "body": None if echo["status"] == 200 else echo["body"][:2000]})
        require(echo["status"] == 200, "echo: the request failed")
        if echo["status"] == 200 and plain["status"] == 200:
            require(reply(echo)["token_ids"] == reply(plain)["token_ids"],
                    "echo: the probe's tokens changed")
    batch = together(calls)
    statuses = [r["status"] for r in batch]
    require(all(s == 200 for s in statuses), "together: a request failed")
    batch_others = digest([reply(r) for r in batch[:-1]])
    batch_probe = digest(reply(batch[-1]))
    emit({"step": "together", "statuses": statuses, "others_sha256": batch_others,
          "probe_sha256": batch_probe})
    require(batch_others == others_digest, "together: the others' replies changed")
    require(batch_probe == probe_digest, "together: the probe's reply changed")
    summary.update(together_others_sha256=batch_others, together_probe_sha256=batch_probe)
    if not reference and statuses[-1] == 200 and "prompt_logprobs_sha256" in summary:
        entries = batch[-1]["body"]["prompt_logprobs"]
        summary["together_prompt_logprobs_sha256"] = digest(entries)
        problems = check_prompt_logprobs(entries, batch[-1]["body"]["prompt_token_ids"], n)
        require(not problems, f"together: probe prompt logprobs: {problems}")

        probe_reply = reply(with_lp)
        teacher = client.post("/v1/completions", {
            "model": client.model, "prompt": prompt_ids + probe_reply["token_ids"],
            "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": n,
            "cache_salt": uuid.uuid4().hex})
        require(teacher["status"] == 200, "teacher: the request failed")
        if teacher["status"] == 200:
            agreement = teacher_agreement(with_lp["body"]["prompt_logprobs"], probe_reply,
                                          teacher["body"]["choices"][0]["prompt_logprobs"])
            emit({"step": "teacher", **agreement})
            summary["teacher"] = agreement
    summary.update(failures=failures, **{"pass": not failures})
    emit(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
