"""check the DSML parser overlay on the live engine. Read-only; temperature 0, thinking off.
Usage (on the head):
  python3 overlay_check.py clients              # requests as agent and chat clients send them (no return_token_ids)
  python3 overlay_check.py compare OLD.json NEW.json   # a fresh capture_fixtures.py run against an earlier one
`clients` checks what the model wrote, whole and streamed. `compare` first checks the output token IDs are the same,
then judges the live parse: quoted cases must come out as intended; for clean ones with the same token IDs, reasoning,
content and tool calls must equal the earlier capture (a parser regression would show there)."""
import json, sys, urllib.request

BASE = "http://127.0.0.1:8000/v1"
D = "｜DSML｜"
WRITE = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
SAMPLE = (f"<{D} calls>\n<{D} invoke name=\"get_weather\">\n"
          f"<{D} parameter name=\"city\" string=\"true\">Boston</{D} parameter>\n</{D} invoke>\n</{D} calls>")
QUOTE = ("Show me an example of DeepSeek's tool-call markup. Put exactly this sample in a code block, do not call any "
         "tool, and end your reply with the word FINISHED.\n\n" + SAMPLE)
INTENDED = {
    "n1_file_of_closers": [["write", {"path": "notes.md", "content": f"DSML closers:\n</{D} parameter>\n</{D} calls>\n"}]],
    "n2_python_line": [["write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'}]],
    "th_n2_python_line": [["write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'}]],
    "ok_two_params": [["write", {"path": "hello.md", "content": "hello world"}]],
}
CLIENT_CASES = {
    "n1_file_of_closers": (WRITE, "Use the write tool to create notes.md whose content is exactly these three lines:\n"
                           f"DSML closers:\n</{D} parameter>\n</{D} calls>"),
    "n2_python_line": (WRITE, f'Use the write tool to create consts.py containing exactly one line: TOOL_END = "</{D} calls>"'),
    "ok_two_params": (WRITE, "Use the write tool to create hello.md containing exactly: hello world"),
    "q876_no_tools": (None, QUOTE),
}


def parsed(calls):
    out = []
    for name, args in calls:
        try:
            out.append([name, json.loads(args or "{}")])
        except ValueError:
            out.append([name, {"_unparsed": args}])
    return out


def judge(name, content, calls):
    if name in INTENDED:
        return parsed(calls) == INTENDED[name]
    if name.startswith("q876_no_tools"):
        return not calls and SAMPLE.split("\n")[-1] in (content or "") and "FINISHED" in (content or "")
    return None


def call(tools, prompt, stream, **extra):
    body = {"model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": 600, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
    body.update(extra)
    if body.get("reasoning_effort") is None:
        body.pop("reasoning_effort")
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            m = json.load(r)["choices"][0]["message"]
            return m.get("content") or "", [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
        content, parts = "", {}
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            for c in json.loads(line[6:]).get("choices", []):
                d = c.get("delta") or {}
                content += d.get("content") or ""
                for t in d.get("tool_calls") or []:
                    p = parts.setdefault(t.get("index", 0), {"name": "", "arguments": ""})
                    f = t.get("function") or {}
                    p["name"] += f.get("name") or ""
                    p["arguments"] += f.get("arguments") or ""
        return content, [(p["name"], p["arguments"]) for _, p in sorted(parts.items())]


if sys.argv[1] == "clients":
    for name, (tools, prompt) in CLIENT_CASES.items():
        for stream in (False, True):
            content, calls = call(tools, prompt, stream)
            print(json.dumps({"case": name, "run": "stream" if stream else "whole", "ok": judge(name, content, calls),
                              "calls": parsed(calls), "content": content[:160]}, ensure_ascii=False), flush=True)
elif sys.argv[1] == "agents":
    # Shapes agent clients send: tool history in the prompt (real DSML tokens), parallel calls, required/named choice.
    WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    hello = "Use the write tool to create hello.md containing exactly: hello world"
    history = [{"role": "user", "content": hello},
               {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {
                   "name": "write", "arguments": json.dumps({"path": "hello.md", "content": "hello world"})}}]},
               {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
               {"role": "user", "content": CLIENT_CASES["n2_python_line"][1]}]
    oslo = "What is the weather in Oslo right now?"
    cases = [
        ("multiturn_n2", WRITE, lambda c, k: parsed(k) == INTENDED["n2_python_line"], {"messages": history}),
        ("multiturn_n2_max", WRITE, lambda c, k: parsed(k) == INTENDED["n2_python_line"],
         {"messages": history, "reasoning_effort": "max", "max_tokens": 6000}),
        ("parallel", WEATHER, lambda c, k: sorted(a.get("city", "") for n, a in parsed(k) if n == "get_weather") == ["Oslo", "Paris"],
         {"messages": [{"role": "user", "content": "What is the weather in Oslo and in Paris right now? Check both."}]}),
        ("required", WEATHER, lambda c, k: parsed(k) == [["get_weather", {"city": "Oslo"}]],
         {"messages": [{"role": "user", "content": oslo}], "tool_choice": "required"}),
        ("named", WEATHER, lambda c, k: parsed(k) == [["get_weather", {"city": "Oslo"}]],
         {"messages": [{"role": "user", "content": oslo}],
          "tool_choice": {"type": "function", "function": {"name": "get_weather"}}}),
    ]
    for name, tools, check, extra in cases:
        for stream in (False, True):
            content, calls = call(tools, "", stream, **extra)
            print(json.dumps({"case": name, "run": "stream" if stream else "whole", "ok": check(content, calls),
                              "calls": parsed(calls), "content": content[:120]}, ensure_ascii=False), flush=True)
elif sys.argv[1] == "compare":
    old = json.load(open(sys.argv[2], encoding="utf-8"))
    new = json.load(open(sys.argv[3], encoding="utf-8"))
    for name, fx in new.items():
        before = old.get(name) or {}
        same_ids = fx.get("token_ids") == before.get("token_ids")
        base = name[:-len("_stream")] if name.endswith("_stream") else name
        ok = judge(base, fx.get("content"), fx.get("tool_calls") or [])
        if ok is None:  # clean case: with the same tokens the parse must equal the earlier one
            ok = (same_ids and (fx.get("content") or "") == (before.get("content") or "")
                  and (fx.get("reasoning") or "") == (before.get("reasoning") or "")
                  and [list(c) for c in fx.get("tool_calls") or []] == [list(c) for c in before.get("tool_calls") or []])
        print(json.dumps({"case": name, "same_token_ids": same_ids, "ok": ok,
                          "calls": parsed(fx.get("tool_calls") or []), "content": (fx.get("content") or "")[:120],
                          "content_before": (before.get("content") or "")[:120]}, ensure_ascii=False), flush=True)
