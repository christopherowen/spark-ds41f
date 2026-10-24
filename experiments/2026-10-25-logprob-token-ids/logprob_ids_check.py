#!/usr/bin/env python3
"""Custom logprob token ids on the serving stack: the request that returned HTTP 500, the
logprobs it returns, and the temperature-0 outputs around it.

usage: logprob_ids_check.py BASE_URL --label LABEL [--reference-labels A,B]
                            [--bug-labels A] [--tokens T] [--seed S]

The probe is controls_digest.py's ``logprob-ids`` scenario (run tilelang-ports-serving-3):
a chat completion of its first prompt with thinking off, temperature 0, ``logprobs`` and
``logprob_token_ids`` [100, 1000, 3000, 4000, 5000]. Every request returns token ids, and
logprob tokens as ``token_id:N``. Steps:

- others: the seven other prompts one at a time, without logprobs (greedy verification);
- probe: the probe with ``logprobs`` only (top 0: each chosen token's logprob);
- probe-ids: the probe with the ids;
- completion-ids: the same through /v1/completions, on the probe's prompt tokens;
- together: the seven and the probe with the ids, all at once (the seven then verify with
  the rejection sampler, beside a request that wants logprobs);
- teacher: completions of the probe's prompt followed by its first P reply tokens, one
  token each, with the ids: each scores the ids at reply position P where a step without
  drafts samples (the sampler), as the probe's verified steps did (the rejection sampler).

A reference arm (one of ``--reference-labels``) has no fix: it runs the seven and the
probe, alone and together, without the ids, so its digests are what the fix arm must
reproduce. A bug arm (one of ``--bug-labels``, a reference arm) also sends the probe with
the ids once and records its status (HTTP 500 without the fix), then the probe again,
which must still return its reply: the engine survives.

Each step prints one JSON line; the summary line holds the digests and ``pass``. Hard
checks, in a fix arm: every request returns HTTP 200; the probe with the ids returns one
logprob entry per token, each listing the five ids; its tokens and each chosen token's
logprob are the probe's without the ids, alone and beside the seven; the completion lists
the five ids for every token; the seven's replies are the same alone and beside the probe.
The completion's tokens, the ids' logprobs alone against beside the seven, and the
teacher's largest gap are recorded, not checked. Exits 1 when a hard check fails, so the
lab keeps the engine logs.
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
# controls_digest.py's LOGPROB_IDS.
IDS = [100, 1000, 3000, 4000, 5000]
TEACHER_POSITIONS = [0, 1, 2, 3, 15, 31, 63, 95]


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
            return {"status": 0, "body": repr(error)}

    def chat(self, prompt: str, tokens: int, seed: int, logprobs: bool,
             ids: list[int] | None = None) -> dict:
        # controls_digest.py's request, with token ids returned.
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "seed": seed, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "temperature": 0.0,
                "chat_template_kwargs": {"thinking": False}, "return_token_ids": True,
                "return_tokens_as_token_ids": True, "logprobs": logprobs}
        if ids:
            body["logprob_token_ids"] = ids
        return self.post("/v1/chat/completions", body)

    def complete(self, prompt_ids: list[int], tokens: int, seed: int) -> dict:
        return self.post("/v1/completions", {
            "model": self.model, "prompt": prompt_ids, "max_tokens": tokens, "seed": seed,
            "ignore_eos": True, "cache_salt": uuid.uuid4().hex, "temperature": 0.0,
            "logprobs": 0, "logprob_token_ids": IDS, "return_tokens_as_token_ids": True,
            "return_token_ids": True})


def reply(result: dict) -> dict | None:
    """The reply's tokens: message, finish reason and token ids."""
    if result["status"] != 200:
        return None
    choice = result["body"]["choices"][0]
    return {key: choice.get(key) for key in ("message", "finish_reason", "token_ids")}


def chosen(result: dict) -> list | None:
    """Each chosen token and its logprob, from a chat reply's logprobs."""
    if result["status"] != 200 or not result["body"]["choices"][0].get("logprobs"):
        return None
    return [[item["token"], item["logprob"]]
            for item in result["body"]["choices"][0]["logprobs"]["content"]]


def id_logprobs(entries: list[dict]) -> list[list]:
    """Per token, the five ids' logprobs (None when an id is missing)."""
    return [[entry.get(f"token_id:{token_id}") for token_id in IDS] for entry in entries]


def chat_entries(result: dict) -> list[dict]:
    """Per token of a chat reply, its listed logprobs as {token: logprob}."""
    return [{top["token"]: top["logprob"] for top in item["top_logprobs"]}
            for item in result["body"]["choices"][0]["logprobs"]["content"]]


def check_entries(entries: list[dict], token_ids: list[int]) -> list[str]:
    """The problems of a reply's logprob entries: one per token, each listing every id."""
    problems = []
    if len(entries) != len(token_ids):
        problems.append(f"{len(entries)} logprob entries for {len(token_ids)} tokens")
    for position, entry in enumerate(entries):
        missing = [t for t in IDS if f"token_id:{t}" not in entry]
        if missing:
            problems.append(f"position {position}: ids {missing} missing")
        elif any(entry[f"token_id:{t}"] > 0 for t in IDS):
            problems.append(f"position {position}: a positive logprob")
    return problems[:10]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--label", required=True)
    parser.add_argument("--reference-labels", default="")
    parser.add_argument("--bug-labels", default="")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    seeds = [args.seed + i for i in range(len(PROMPTS))]
    probe, others = (PROMPTS[0], seeds[0]), list(zip(PROMPTS[1:], seeds[1:]))
    reference = args.label in args.reference_labels.split(",")
    bug = args.label in args.bug_labels.split(",")
    failures: list[str] = []

    def emit(record: dict) -> None:
        print(json.dumps(record), flush=True)

    def require(condition: bool, what: str) -> None:
        if not condition:
            failures.append(what)

    alone = [client.chat(p, args.tokens, s, False) for p, s in others]
    require(all(r["status"] == 200 for r in alone), "others: a request failed")
    others_digest = digest([reply(r) for r in alone])
    emit({"step": "others", "statuses": [r["status"] for r in alone], "sha256": others_digest})

    plain = client.chat(probe[0], args.tokens, probe[1], True)
    require(plain["status"] == 200, "probe: the request failed")
    probe_digest, probe_logprobs = digest(reply(plain)), digest(chosen(plain))
    emit({"step": "probe", "status": plain["status"], "sha256": probe_digest,
          "logprobs_sha256": probe_logprobs})
    summary = {"summary": "logprob token ids", "label": args.label, "reference": reference,
               "ids": IDS, "tokens": args.tokens, "seed": args.seed,
               "others_sha256": others_digest, "probe_sha256": probe_digest,
               "probe_logprobs_sha256": probe_logprobs}

    if bug:
        crash = client.chat(probe[0], args.tokens, probe[1], True, IDS)
        again = client.chat(probe[0], args.tokens, probe[1], True)
        emit({"step": "probe-ids (no fix)", "status": crash["status"],
              "body": None if crash["status"] == 200 else crash["body"][:2000],
              "probe_again_status": again["status"]})
        require(again["status"] == 200 and digest(reply(again)) == probe_digest,
                "probe after the request with ids: the engine did not serve it the same")
        summary.update(bug_status=crash["status"], bug_entries=None if crash["status"] != 200
                       else len(crash["body"]["choices"][0]["logprobs"]["content"]))

    with_ids = None
    if not reference:
        with_ids = client.chat(probe[0], args.tokens, probe[1], True, IDS)
        status = with_ids["status"]
        emit({"step": "probe-ids", "status": status,
              "body": None if status == 200 else with_ids["body"][:2000]})
        require(status == 200, "probe with ids: the request failed")
        if status == 200:
            entries = chat_entries(with_ids)
            token_ids = reply(with_ids)["token_ids"]
            problems = check_entries(entries, token_ids)
            require(not problems, f"probe with ids: {problems}")
            require(digest(reply(with_ids)) == probe_digest, "probe: the ids changed its reply")
            require(digest(chosen(with_ids)) == probe_logprobs,
                    "probe: the ids changed its chosen tokens' logprobs")
            summary.update(entries=len(entries), entry_problems=problems,
                           ids_logprobs_sha256=digest(id_logprobs(entries)))

            prompt_ids = with_ids["body"]["prompt_token_ids"]
            completion = client.complete(prompt_ids, args.tokens, probe[1])
            ok = completion["status"] == 200
            emit({"step": "completion-ids", "status": completion["status"],
                  "body": None if ok else completion["body"][:2000]})
            require(ok, "completion with ids: the request failed")
            if ok:
                choice = completion["body"]["choices"][0]
                problems = check_entries(choice["logprobs"]["top_logprobs"], choice["token_ids"])
                require(not problems, f"completion with ids: {problems}")
                summary.update(completion_entries=len(choice["logprobs"]["top_logprobs"]),
                               completion_problems=problems,
                               completion_tokens_equal=choice["token_ids"] == token_ids)

    calls = [lambda p=p, s=s: client.chat(p, args.tokens, s, False) for p, s in others]
    calls.append(lambda: client.chat(probe[0], args.tokens, probe[1], True,
                                     None if reference else IDS))
    batch = together(calls)
    statuses = [r["status"] for r in batch]
    require(all(s == 200 for s in statuses), "together: a request failed")
    batch_others, batch_probe = digest([reply(r) for r in batch[:-1]]), digest(reply(batch[-1]))
    batch_logprobs = digest(chosen(batch[-1]))
    emit({"step": "together", "statuses": statuses, "others_sha256": batch_others,
          "probe_sha256": batch_probe, "probe_logprobs_sha256": batch_logprobs})
    require(batch_others == others_digest, "together: the others' replies changed")
    require(batch_probe == probe_digest, "together: the probe's reply changed")
    require(batch_logprobs == probe_logprobs,
            "together: the probe's chosen tokens' logprobs changed")
    summary.update(together_others_sha256=batch_others, together_probe_sha256=batch_probe,
                   together_probe_logprobs_sha256=batch_logprobs)

    if with_ids is not None and with_ids["status"] == 200 and statuses[-1] == 200:
        entries = chat_entries(batch[-1])
        problems = check_entries(entries, reply(batch[-1])["token_ids"])
        require(not problems, f"together: probe with ids: {problems}")
        summary["together_ids_logprobs_equal"] = (
            digest(id_logprobs(entries)) == summary["ids_logprobs_sha256"])

        reply_ids = reply(with_ids)["token_ids"]
        probe_entries = chat_entries(with_ids)
        gaps, teacher_failed = [], []
        for position in TEACHER_POSITIONS:
            if position >= len(reply_ids):
                continue
            result = client.complete(prompt_ids + reply_ids[:position], 1, probe[1])
            if result["status"] != 200:
                teacher_failed.append(position)
                continue
            theirs = result["body"]["choices"][0]["logprobs"]["top_logprobs"][0]
            mine = probe_entries[position]
            if check_entries([theirs], [0]):
                teacher_failed.append(position)
                continue
            gaps.append(max(abs(mine[f"token_id:{t}"] - theirs[f"token_id:{t}"]) for t in IDS))
        require(not teacher_failed, f"teacher: requests failed at {teacher_failed}")
        teacher = {"positions": [p for p in TEACHER_POSITIONS if p < len(reply_ids)],
                   "max_gap": round(max(gaps), 6) if gaps else None,
                   "gaps": [round(g, 6) for g in gaps]}
        emit({"step": "teacher", **teacher})
        summary["teacher"] = teacher
    summary.update(failures=failures, **{"pass": not failures})
    emit(summary)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
