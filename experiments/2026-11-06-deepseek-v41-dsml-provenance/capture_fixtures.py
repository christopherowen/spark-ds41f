"""Capture replay fixtures from a live deepseek_v41 server: each request, its prompt and output token IDs, and
what the parser returned, whole and streamed. Read-only; temperature 0, thinking off, unforced.
Usage (on the head, against 127.0.0.1:8000): python3 capture_fixtures.py <out.json>"""
import json, sys, urllib.request

BASE = "http://127.0.0.1:8000/v1"
D = "｜DSML｜"
WRITE = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
WEATHER = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
SAMPLE = (f"<{D} calls>\n<{D} invoke name=\"get_weather\">\n"
          f"<{D} parameter name=\"city\" string=\"true\">Boston</{D} parameter>\n</{D} invoke>\n</{D} calls>")
QUOTE = ("Show me an example of DeepSeek's tool-call markup. Put exactly this sample in a code block, do not call any "
         "tool, and end your reply with the word FINISHED.\n\n" + SAMPLE)
CASES = {
    "n1_file_of_closers": (WRITE, "Use the write tool to create notes.md whose content is exactly these three lines:\n"
                           f"DSML closers:\n</{D} parameter>\n</{D} calls>"),
    "n2_python_line": (WRITE, f'Use the write tool to create consts.py containing exactly one line: TOOL_END = "</{D} calls>"'),
    "ok_two_params": (WRITE, "Use the write tool to create hello.md containing exactly: hello world"),
    "q876_tools": (WEATHER, QUOTE),
    "q876_no_tools": (None, QUOTE),
}


def run(tools, prompt, stream):
    body = {"model": "deepseek-v4.1-flash", "temperature": 0, "reasoning_effort": "none", "max_tokens": 600,
            "return_token_ids": True, "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            o = json.load(r)
            c = o["choices"][0]
            m = c["message"]
            return {"request": body, "prompt_token_ids": o.get("prompt_token_ids"), "token_ids": c.get("token_ids"),
                    "content": m.get("content"), "reasoning": m.get("reasoning_content") or m.get("reasoning"),
                    "tool_calls": [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []],
                    "finish_reason": c.get("finish_reason")}
        content, finish, parts, ids, prompt_ids = "", None, {}, [], None
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            o = json.loads(line[6:])
            prompt_ids = prompt_ids or o.get("prompt_token_ids")
            for c in o.get("choices", []):
                ids += c.get("token_ids") or []
                d = c.get("delta") or {}
                content += d.get("content") or ""
                finish = c.get("finish_reason") or finish
                for t in d.get("tool_calls") or []:
                    p = parts.setdefault(t.get("index", 0), {"name": "", "arguments": ""})
                    f = t.get("function") or {}
                    p["name"] += f.get("name") or ""
                    p["arguments"] += f.get("arguments") or ""
        return {"request": body, "prompt_token_ids": prompt_ids, "token_ids": ids, "content": content,
                "tool_calls": [(p["name"], p["arguments"]) for _, p in sorted(parts.items())], "finish_reason": finish}


out = {}
for name, (tools, prompt) in CASES.items():
    for stream in (False, True):
        r = run(tools, prompt, stream)
        out[f"{name}{'_stream' if stream else ''}"] = r
        print(name, "stream" if stream else "whole", "| finish", r["finish_reason"], "| tokens", len(r["token_ids"] or []),
              "| 128825 at", [i for i, t in enumerate(r["token_ids"] or []) if t == 128825],
              "| calls", json.dumps(r["tool_calls"], ensure_ascii=False)[:160],
              "| content", json.dumps(r["content"] or "", ensure_ascii=False)[:80], flush=True)
json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
