"""legitimate repetitive output, the case 1107's default loop detection (0656/0674/0679) must leave
alone. Tool calls that write long repetitive files, and one repetitive plain answer; thinking off and at max effort,
whole and streamed. Read-only against :8000; temperature 0.
Usage (on the head): python3 loop_probe.py <out.jsonl>
Per run: finish_reason, completion tokens, and whether the written content is complete (counted exactly)."""
import json, re, sys, time, urllib.request

ZERO = re.compile(r"(?<![0-9.])0(?![0-9.])")

BASE = "http://127.0.0.1:8000/v1"
WRITE = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
CASES = {
    # A 256-zero C array, every zero written out (0679's notes: about 780 tokens).
    "zeros256": (WRITE, "Use the write tool to create zeros.c containing exactly one statement: a C array named z of 256 "
                 "ints, every element written out explicitly as 0, separated by \", \", like `int z[256] = {0, 0, 0};` "
                 "but with all 256 zeros. Do not use loops, macros or shorthand.",
                 lambda c, a: len(ZERO.findall(a.get("content", "").split("{", 1)[-1])) == 256),
    # A 16x16 identity-free matrix of ones (about 790 tokens).
    "ones16x16": (WRITE, "Use the write tool to create ones.txt: 16 lines, each line exactly sixteen 1s separated by "
                  "single spaces. Nothing else in the file.",
                  lambda c, a: [l.split() for l in a.get("content", "").strip().splitlines()] == [["1"] * 16] * 16),
    # A long table with identical rows.
    "table120": (WRITE, "Use the write tool to create table.md: a markdown table with header `| id | status |`, the "
                 "separator row `|---|---|`, then 120 rows numbered 1 to 120, each with status `ok`, like `| 7 | ok |`.",
                 lambda c, a: sum(1 for l in a.get("content", "").splitlines() if l.strip().endswith("| ok |")) == 120),
    # No tool: the same zero array as the visible answer.
    "zeros256_answer": (None, "Write out a C array named z of 256 ints with every element written explicitly as 0, "
                        "separated by \", \", in one code block. No loops, macros or shorthand, and no other text.",
                        lambda c, a: len(ZERO.findall(c.split("{", 1)[-1])) == 256),
}


def run(tools, prompt, stream, effort):
    body = {"model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": 4000 if effort == "none" else 12000,
            "reasoning_effort": effort, "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    t0 = time.time()
    req = urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        if not stream:
            o = json.load(r)
            c = o["choices"][0]
            m = c["message"]
            calls = [(t["function"]["name"], t["function"]["arguments"]) for t in m.get("tool_calls") or []]
            u = o["usage"]
            return (m.get("content") or "", calls, c.get("finish_reason"), u["completion_tokens"], time.time() - t0,
                    m.get("reasoning_content") or m.get("reasoning") or "", (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))
        content, reasoning, parts, finish, usage = "", "", {}, None, None
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            o = json.loads(line[6:])
            usage = o.get("usage") or usage
            for c in o.get("choices", []):
                d = c.get("delta") or {}
                content += d.get("content") or ""
                reasoning += d.get("reasoning_content") or d.get("reasoning") or ""
                finish = c.get("finish_reason") or finish
                for t in d.get("tool_calls") or []:
                    p = parts.setdefault(t.get("index", 0), {"name": "", "arguments": ""})
                    f = t.get("function") or {}
                    p["name"] += f.get("name") or ""
                    p["arguments"] += f.get("arguments") or ""
        calls = [(p["name"], p["arguments"]) for _, p in sorted(parts.items())]
        u = usage or {}
        return (content, calls, finish, u.get("completion_tokens"), time.time() - t0, reasoning,
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens"))


with open(sys.argv[1], "w", encoding="utf-8") as out:
    for effort in ("none", "max"):
        for name, (tools, prompt, check) in CASES.items():
            for stream in (False, True):
                content, calls, finish, ntok, secs, reasoning, rtok = run(tools, prompt, stream, effort)
                try:
                    args = json.loads(calls[0][1]) if calls else {}
                except ValueError:
                    args = {"_unparsed": calls[0][1]}
                row = {"case": name, "effort": effort, "run": "stream" if stream else "whole",
                       "complete": bool(check(content, args)), "finish_reason": finish, "completion_tokens": ntok,
                       "seconds": round(secs, 1), "calls": [c[0] for c in calls],
                       "content_len": len(args.get("content", content)), "reasoning_chars": len(reasoning),
                       "reasoning_tokens": rtok,
                       "zeros": len(ZERO.findall((args.get("content", content) or "").split("{", 1)[-1])),
                       "written": (args.get("content", content) or "")[:1500], "args_raw": (calls[0][1][:300] if calls else ""),
                       "reasoning_tail": reasoning[-400:]}
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(json.dumps(row), flush=True)
