#!/usr/bin/env python3
"""DSpark against a suffix drafter on the same copy-heavy output.

usage (a lab extra, run on the head node from the repository root):
  copy_probe.py HEAD_URL [--prompts 12] [--seed 0]

A suffix drafter proposes the tokens that followed the latest earlier occurrence of
the last ``m`` tokens; on agent traffic about a third of output tokens could come
from such copies (experiments/2026-10-23-next-release README, "Draft acceptance").
Whether that pays depends on what DSpark already accepts there. Each category runs
alone at one stream (temperature 0, thinking off): DSpark's tokens a step come from
the server's spec-decode counters before and after it, and a suffix drafter is
simulated on the exact prompt and output token ids the server returns. Categories:
prose (servebench's default prompts), copy (reproduce a source file), edit (the
file again with a comment above every ``def``), tool (an edit tool call whose
old_string is a function copied from the context). One JSON line per category.
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from servebench import prom, prompts  # noqa: E402

MODEL = "deepseek-v4.1-flash"
COUNTERS = ("vllm:spec_decode_num_drafts_total", "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_accepted_tokens_total")
EDIT_TOOL = {"type": "function", "function": {
    "name": "edit_file", "description": "Replace old_string with new_string in a file.",
    "parameters": {"type": "object", "required": ["path", "old_string", "new_string"], "properties": {
        "path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}}}}}


def counters(url: str) -> dict:
    text = urllib.request.urlopen(f"{url}/metrics", timeout=10).read().decode()
    values = prom.parse(text)
    out = {name: sum(v for k, v in values.items() if k == name or k.startswith(name + "{")) for name in COUNTERS}
    out["per_pos"] = {k.split('position="')[1].split('"')[0]: v for k, v in values.items()
                      if k.startswith("vllm:spec_decode_num_accepted_tokens_per_pos_total{")}
    return out


def chat(url: str, messages: list, max_tokens: int, tools=None) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "return_token_ids": True, "chat_template_kwargs": {"thinking": False}}
    if tools:
        body.update(tools=tools, tool_choice="required")
    request = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.loads(response.read())


def suffix_steps(prompt: list[int], output: list[int], m: int, k: int) -> list[int]:
    """Accepted lengths of the steps a suffix drafter proposes on ``output`` (-1 where
    it proposes nothing), stepping by accepted + 1 as a verifier would. A proposal
    copies only tokens that exist when it is made."""
    seq = prompt + output
    latest: dict[tuple, int] = {}
    indexed = m - 1  # the last n-gram end indexed
    steps, i = [], 0
    while i < len(output):
        position = len(prompt) + i
        while indexed < position - 1:  # every earlier occurrence, not the current suffix
            indexed += 1
            latest[tuple(seq[indexed - m:indexed])] = indexed
        source = latest.get(tuple(seq[position - m:position])) if position >= m else None
        if source is None:
            steps.append(-1)
            i += 1
            continue
        accepted = 0
        while (accepted < k and i + accepted < len(output) and source + accepted < position
               and seq[source + accepted] == output[i + accepted]):
            accepted += 1
        steps.append(accepted)
        i += accepted + 1
    return steps


def simulate(samples: list[tuple[list[int], list[int]]], dspark_tps: float, m: int, k: int) -> dict:
    """Tokens a step for a hybrid that takes every suffix proposal and lets DSpark do
    the rest at its measured rate, and for one that takes a proposal only when it
    beats DSpark (an oracle bound)."""
    tokens = sum(len(o) for _, o in samples)
    hybrid_steps = best_steps = 0.0
    proposals = accepted_total = 0
    for prompt, output in samples:
        rest_h = rest_b = 0
        for accepted in suffix_steps(prompt, output, m, k):
            if accepted < 0:
                rest_h += 1
                rest_b += 1
                continue
            proposals += 1
            accepted_total += accepted
            hybrid_steps += 1
            if accepted + 1 > dspark_tps:
                best_steps += 1
            else:
                rest_b += accepted + 1
        hybrid_steps += rest_h / dspark_tps
        best_steps += rest_b / dspark_tps
    return {"m": m, "k": k, "proposals": proposals,
            "mean_accepted": round(accepted_total / proposals, 2) if proposals else None,
            "hybrid_tokens_per_step": round(tokens / hybrid_steps, 3) if hybrid_steps else None,
            "best_tokens_per_step": round(tokens / best_steps, 3) if best_steps else None}


def cases(name: str, count: int, seed: int):
    if name == "prose":
        for p in prompts.load("default")[:count]:
            yield [{"role": "user", "content": p["text"]}], 384, None
        return
    for i in range(count):
        code = prompts.source_text(1500, seed * 1000 + i).split("\n", 1)[1]
        code = code[:code.rfind("\n")]
        fence = f"```python\n{code}\n```"
        if name == "copy":
            text = f"Here is a Python file:\n\n{fence}\n\nReproduce the file exactly as it is, with no commentary."
            yield [{"role": "user", "content": text}], 2048, None
        elif name == "edit":
            text = (f"Here is a Python file:\n\n{fence}\n\nOutput the complete file again, unchanged except for a "
                    "line `# reviewed` (at the same indentation) above every line that starts a `def`.")
            yield [{"role": "user", "content": text}], 2400, None
        else:
            text = (f"The file lib/module.py contains:\n\n{fence}\n\nRename the first function or method defined in "
                    "it by adding the suffix `_v2` to its name. Use edit_file; old_string must be that whole "
                    "function, copied exactly, and new_string the same function with only its name changed.")
            yield [{"role": "user", "content": text}], 2048, [EDIT_TOOL]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--prompts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    url = args.url.rstrip("/")
    failed = False
    for name in ("prose", "copy", "edit", "tool"):
        before = counters(url)
        samples, errors = [], 0
        for messages, max_tokens, tools in cases(name, args.prompts, args.seed):
            try:
                reply = chat(url, messages, max_tokens, tools)
                samples.append((reply["prompt_token_ids"], reply["choices"][0]["token_ids"]))
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors += 1
                print(json.dumps({"category": name, "error": str(error)[:300]}), flush=True)
        after = counters(url)
        drafts = after[COUNTERS[0]] - before[COUNTERS[0]]
        accepted = after[COUNTERS[2]] - before[COUNTERS[2]]
        dspark = (accepted + drafts) / drafts if drafts else None
        per_pos = {p: round((after["per_pos"][p] - before["per_pos"].get(p, 0)) / drafts, 3)
                   for p in sorted(after["per_pos"])} if drafts else {}
        record = {"category": name, "requests": len(samples), "errors": errors,
                  "output_tokens": sum(len(o) for _, o in samples), "drafts": drafts,
                  "dspark_tokens_per_step": round(dspark, 3) if dspark else None,
                  "dspark_acceptance_by_position": per_pos}
        if dspark:
            record["suffix"] = [simulate(samples, dspark, m, k) for m, k in ((8, 16), (16, 16), (4, 16), (8, 8))]
        failed |= errors > len(samples) // 4 or not samples
        print(json.dumps(record), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
