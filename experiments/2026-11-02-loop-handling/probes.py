"""The known loop triggers of triggers.json, sent to a server and checked.

Each trigger is sent once per seed through /v1/chat/completions. The reply's
reasoning and answer are tokenized by the server (/tokenize), and the probe
records the finish reason, the longest run of one token and the distinct-token
ratio, then checks the trigger's expectation:

- clean: finishes by itself (finish_reason stop) and no run reaches 24 tokens;
- bounded: may degenerate, but never runs away: no run longer than max_run, and
  a reply that degenerated does not end on finish_reason length;
- kept: legitimate repetition is never cut by the loop handling: finish_reason is
  not repetition. The model may still run past what was asked (the 2026-11-02 soak
  wrote 'hello' 281 times when asked for 200) and end on length.

Read-only towards the server: chat completions and /tokenize only. Run it in a
lab window like the sink probes (experiments/2026-10-19-sink-rejection-v2).

Usage: python3 probes.py BASE_URL [--only ID,ID] [--periodic] [--repo DIR]
Prints one JSON line per reply, then {"summary": ...}; exits 1 if any fails.
"""

import argparse
import json
import pathlib
import sys
import urllib.request

MODEL = "deepseek-v4.1-flash"
HERE = pathlib.Path(__file__).resolve().parent
DEGENERATE_RUN = 24


def post(base: str, path: str, body: dict, timeout: int = 1800) -> dict:
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def tokenize(base: str, text: str) -> list[int]:
    if not text:
        return []
    return post(base, "/tokenize", {"model": MODEL, "prompt": text,
                                    "add_special_tokens": False})["tokens"]


def analyze(ids: list[int]) -> dict:
    """The longest run of one token, its token, and the distinct-token ratio."""
    longest, token, run = 0, None, 0
    for i, tok in enumerate(ids):
        run = run + 1 if i and tok == ids[i - 1] else 1
        if run > longest:
            longest, token = run, tok
    return {"tokens": len(ids), "longest_run": longest, "run_token": token,
            "distinct_ratio": round(len(set(ids)) / len(ids), 3) if ids else None}


def verdict(trigger: dict, finish_reason: str, stats: dict) -> tuple[bool, str]:
    run = stats["longest_run"]
    expect = trigger["expect"]
    if expect == "clean":
        ok = finish_reason == "stop" and run < DEGENERATE_RUN
        return ok, "" if ok else f"finish_reason {finish_reason}, longest run {run}"
    if expect == "kept":
        ok = finish_reason != "repetition"
        return ok, "" if ok else "legitimate repetition was stopped as a loop"
    if expect == "bounded":
        if run > trigger.get("max_run", 600):
            return False, f"run of {run} tokens passed max_run"
        if run >= DEGENERATE_RUN and finish_reason == "length":
            return False, f"degenerated (run {run}) and ran to max_tokens"
        return True, ""
    raise SystemExit(f"{trigger['id']}: unknown expectation {expect}")


def repository_context(base: str, repo: pathlib.Path, target: int) -> str:
    """Distinct repository documents, in a fixed order, up to target tokens."""
    paths = [repo / "README.md", repo / "TODO.md", repo / "AGENTS.md"]
    paths += sorted((repo / "docs").glob("*.md"))
    paths += sorted((repo / "experiments").glob("*/README.md"))
    parts, total = [], 0
    for path in paths:
        if not path.is_file():
            continue
        text = f"# {path.relative_to(repo)}\n\n{path.read_text()}\n"
        n = len(tokenize(base, text))
        if total + n > target:
            break
        parts.append(text)
        total += n
    if total < target * 0.9:
        raise SystemExit(f"repository context reached only {total} of {target} tokens")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--only", help="comma-separated trigger ids")
    parser.add_argument("--periodic", action="store_true", help="include the long-context triggers")
    parser.add_argument("--repo", default=str(HERE.parents[1]), help="checkout for the long context")
    parser.add_argument("--port", type=int, help="the arm's port, if not BASE_URL's (lab passes :8000)")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.port:
        base = base.rsplit(":", 1)[0] + f":{args.port}"

    triggers = json.loads((HERE / "triggers.json").read_text())["triggers"]
    only = set(args.only.split(",")) if args.only else None
    contexts: dict[int, str] = {}
    failures = []
    count = 0
    for trigger in triggers:
        if only is not None and trigger["id"] not in only:
            continue
        if trigger.get("periodic") and not args.periodic and only is None:
            continue
        messages = list(trigger["messages"])
        if trigger.get("context_tokens"):
            target = trigger["context_tokens"]
            if target not in contexts:
                contexts[target] = repository_context(base, pathlib.Path(args.repo), target)
            messages[0] = {"role": "user",
                           "content": contexts[target] + "\n\n" + messages[0]["content"]}
        kwargs = {"thinking": trigger["thinking"]}
        if trigger.get("effort"):
            kwargs["reasoning_effort"] = trigger["effort"]
        for seed in trigger["seeds"]:
            response = post(base, "/v1/chat/completions", {
                "model": MODEL, "messages": messages, "max_tokens": trigger["max_tokens"],
                "temperature": trigger["temperature"], "seed": seed,
                "chat_template_kwargs": kwargs})
            choice = response["choices"][0]
            message = choice["message"]
            reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
            content = message.get("content") or ""
            ids = tokenize(base, reasoning) + tokenize(base, content)
            stats = analyze(ids)
            ok, why = verdict(trigger, choice["finish_reason"], stats)
            row = {"id": trigger["id"], "seed": seed, "expect": trigger["expect"], "ok": ok,
                   "why": why, "finish_reason": choice["finish_reason"],
                   "completion_tokens": (response.get("usage") or {}).get("completion_tokens"),
                   "reasoning_chars": len(reasoning), "content_chars": len(content), **stats,
                   "content_head": content[:200]}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            count += 1
            if not ok:
                failures.append(f"{trigger['id']}/{seed}: {why}")
    print(json.dumps({"summary": {"replies": count, "failed": len(failures), "failures": failures}}))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
