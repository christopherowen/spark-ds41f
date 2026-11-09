#!/usr/bin/env python3
"""#955 live check: a tool call whose arguments quote DSML markup returns complete arguments,
and DSML the model quotes stays text. A lab extra against a served endpoint.

usage: check_955_quoted_dsml.py URL [--model NAME] [--timeout S]

The requests are #955's own capture cases (capture_fixtures.py in
experiments/2026-11-06-deepseek-v41-dsml-provenance at the PR head 3a854ba440e1), sent to
URL/v1/chat/completions at temperature 0, unforced, with return_token_ids, whole and streamed:
- thinking off (reasoning_effort "none", 600 tokens): n1_file_of_closers, n2_python_line,
  ok_two_params (control), q876_tools, q876_no_tools;
- thinking on (effort left out, 6000 tokens): th_n2_python_line.

A case passes, whole and streamed alike, when:
- n1, n2, th_n2, ok_two_params: exactly one `write` call, with JSON arguments, the intended
  path, and content that keeps every quoted line (1107 cut n1's content to "DSML closers:\\n"
  and n2's to nothing or `TOOL_END = "`);
- q876_no_tools, q876_tools: no tool call, and the reply keeps the quoted DSML (either
  spelling) and FINISHED (1107 cut q876_no_tools's content to the opening code fence).
Recorded beside, not required: whether the arguments equal the PR's replay exactly
(`exact`), whether q876_no_tools keeps the sample verbatim, and where the DSML token
(128825) sits in the output token IDs.

Prints one JSON line per case and mode, then a summary line ("passed"); exits 1 when a case
fails or the endpoint does not answer, 0 otherwise. Stdlib only.
"""
import argparse
import json
import sys
import time
import urllib.request

D = "｜DSML｜"
DSML_TOKEN = 128825
WRITE = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SAMPLE = (f"<{D} calls>\n<{D} invoke name=\"get_weather\">\n"
          f"<{D} parameter name=\"city\" string=\"true\">Boston</{D} parameter>\n</{D} invoke>\n</{D} calls>")
QUOTE = ("Show me an example of DeepSeek's tool-call markup. Put exactly this sample in a code block, do not call any "
         "tool, and end your reply with the word FINISHED.\n\n" + SAMPLE)
N2 = f'Use the write tool to create consts.py containing exactly one line: TOOL_END = "</{D} calls>"'
# name: (tools, prompt, reasoning effort: "none" = thinking off, None = left out)
CASES = {
    "n1_file_of_closers": (WRITE, "Use the write tool to create notes.md whose content is exactly these three lines:\n"
                           f"DSML closers:\n</{D} parameter>\n</{D} calls>", "none"),
    "n2_python_line": (WRITE, N2, "none"),
    "ok_two_params": (WRITE, "Use the write tool to create hello.md containing exactly: hello world", "none"),
    "q876_tools": (WEATHER, QUOTE, "none"),
    "q876_no_tools": (None, QUOTE, "none"),
    "th_n2_python_line": (WRITE, N2, None),
}
# What each write call must keep: its path and every quoted line of its content.
WRITES = {
    "n1_file_of_closers": ("notes.md", ["DSML closers:", f"</{D} parameter>", f"</{D} calls>"]),
    "n2_python_line": ("consts.py", [f'TOOL_END = "</{D} calls>"']),
    "th_n2_python_line": ("consts.py", [f'TOOL_END = "</{D} calls>"']),
    "ok_two_params": ("hello.md", ["hello world"]),
}
# The PR's replay of the patched parser over 1030's tokens (runs/replay-patched.jsonl).
INTENDED = {
    "n1_file_of_closers": [["write", {"path": "notes.md", "content": f"DSML closers:\n</{D} parameter>\n</{D} calls>\n"}]],
    "n2_python_line": [["write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'}]],
    "th_n2_python_line": [["write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'}]],
    "ok_two_params": [["write", {"path": "hello.md", "content": "hello world"}]],
}


def model_name(base: str, timeout: float) -> str:
    with urllib.request.urlopen(base + "/v1/models", timeout=timeout) as response:
        return json.load(response)["data"][0]["id"]


def run(base: str, model: str, tools, prompt: str, effort, stream: bool, timeout: float) -> dict:
    body = {"model": model, "temperature": 0, "max_tokens": 600 if effort == "none" else 6000,
            "return_token_ids": True, "messages": [{"role": "user", "content": prompt}]}
    if effort is not None:
        body["reasoning_effort"] = effort
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
    request = urllib.request.Request(base + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not stream:
            choice = json.load(response)["choices"][0]
            message = choice["message"]
            return {"content": message.get("content") or "",
                    "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
                    "calls": [[t["function"]["name"], t["function"]["arguments"]] for t in message.get("tool_calls") or []],
                    "token_ids": choice.get("token_ids") or [], "finish_reason": choice.get("finish_reason")}
        content, reasoning, finish, parts, ids = "", "", None, {}, []
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            for choice in event.get("choices", []):
                ids += choice.get("token_ids") or []
                delta = choice.get("delta") or {}
                content += delta.get("content") or ""
                reasoning += delta.get("reasoning_content") or delta.get("reasoning") or ""
                finish = choice.get("finish_reason") or finish
                for call in delta.get("tool_calls") or []:
                    part = parts.setdefault(call.get("index", 0), {"name": "", "arguments": ""})
                    function = call.get("function") or {}
                    part["name"] += function.get("name") or ""
                    part["arguments"] += function.get("arguments") or ""
        return {"content": content, "reasoning": reasoning, "token_ids": ids, "finish_reason": finish,
                "calls": [[p["name"], p["arguments"]] for _, p in sorted(parts.items())]}


def judge(name: str, reply: dict) -> tuple[bool, list[str], list]:
    """Whether the reply passes, why not, and its calls with parsed arguments."""
    problems, parsed = [], []
    for function, arguments in reply["calls"]:
        try:
            parsed.append([function, json.loads(arguments or "{}")])
        except ValueError:
            parsed.append([function, {"_unparsed": arguments}])
            problems.append(f"{function} arguments are not JSON: {arguments[:120]!r}")
    if name in WRITES:
        path, lines = WRITES[name]
        if len(parsed) != 1 or parsed[0][0] != "write":
            problems.append(f"expected one write call, got {[c[0] for c in parsed]}")
        else:
            arguments = parsed[0][1]
            if arguments.get("path") != path:
                problems.append(f"path {arguments.get('path')!r}, expected {path!r}")
            content = arguments.get("content")
            if not isinstance(content, str):
                problems.append("content argument missing")
            else:
                problems += [f"content lost {line!r}" for line in lines if line not in content]
    else:
        if parsed:
            problems.append(f"expected no tool call, got {[c[0] for c in parsed]}")
        # The model quotes either spelling (1030 wrote V4's <｜DSML｜tool_calls> in q876_tools).
        for kept in (D, "FINISHED"):
            if kept not in reply["content"]:
                problems.append(f"reply lost {kept!r}")
    return not problems, problems, parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--model")
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    base = args.url.rstrip("/")
    try:
        model = args.model or model_name(base, 60)
    except OSError as error:
        print(json.dumps({"summary": "#955 quoted DSML stays text", "passed": False,
                          "error": f"endpoint did not answer: {error}"}))
        return 1
    results, failing, exact = {}, [], {}
    for name, (tools, prompt, effort) in CASES.items():
        for mode in ("whole", "stream"):
            started = time.monotonic()
            try:
                reply = run(base, model, tools, prompt, effort, mode == "stream", args.timeout)
            except (OSError, ValueError, KeyError, IndexError) as error:
                row = {"case": name, "mode": mode, "passed": False, "problems": [f"{type(error).__name__}: {error}"]}
            else:
                passed, problems, parsed = judge(name, reply)
                ids = reply["token_ids"]
                row = {"case": name, "mode": mode, "passed": passed, "problems": problems, "calls": parsed,
                       "content": reply["content"][:400], "reasoning_chars": len(reply["reasoning"]),
                       "finish_reason": reply["finish_reason"], "output_tokens": len(ids),
                       "dsml_token_positions": [i for i, t in enumerate(ids) if t == DSML_TOKEN]}
                if name in INTENDED:
                    row["exact"] = parsed == INTENDED[name]
                    exact[f"{name}/{mode}"] = row["exact"]
                elif name == "q876_no_tools":
                    row["sample_verbatim"] = SAMPLE in reply["content"]
            row["seconds"] = round(time.monotonic() - started, 3)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            results.setdefault(name, {})[mode] = row["passed"]
            if not row["passed"]:
                failing.append(f"{name}/{mode}")
    passed = not failing
    print(json.dumps({"summary": "#955 quoted DSML stays text", "passed": passed, "model": model,
                      "cases": results, "failing": failing, "exact": exact}, ensure_ascii=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
