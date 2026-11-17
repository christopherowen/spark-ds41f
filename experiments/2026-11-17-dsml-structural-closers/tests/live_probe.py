"""Live probe: does a served DSV4.1 lane return a tool argument holding DSML closers intact?

python3 live_probe.py <base_url_with_/v1> [model]   (stdlib only)
"""
import json
import sys
import time
import urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else None
TARGET = ("DSV4.1 tool calls end with </｜DSML｜ calls>, each argument ends with </｜DSML｜ parameter>, "
          "and each invoke ends with </｜DSML｜ invoke>. Keep these tags literal.")
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a text file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=300)


def model_id():
    with urllib.request.urlopen(BASE + "/models", timeout=20) as r:
        return json.load(r)["data"][0]["id"]


def body(prompt, stream):
    b = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "tools": TOOLS, "tool_choice": "auto",
         "temperature": 0, "max_tokens": 600, "stream": stream,
         "chat_template_kwargs": {"thinking": False}}
    return b


def run(prompt, stream):
    t = time.time()
    r = post("/chat/completions", body(prompt, stream), stream)
    if not stream:
        d = json.load(r)
        msg = d["choices"][0]["message"]
        calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (msg.get("tool_calls") or [])]
        return calls, msg.get("content"), d["choices"][0].get("finish_reason"), round(time.time() - t, 1)
    calls, content, finish = {}, "", None
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        ch = json.loads(line[5:])["choices"]
        if not ch:
            continue
        delta = ch[0].get("delta") or {}
        finish = ch[0].get("finish_reason") or finish
        if delta.get("content"):
            content += delta["content"]
        for tc in delta.get("tool_calls") or []:
            s = calls.setdefault(tc.get("index", 0), {"name": None, "args": ""})
            f = tc.get("function") or {}
            if f.get("name"):
                s["name"] = f["name"]
            if f.get("arguments"):
                s["args"] += f["arguments"]
    return [(calls[k]["name"], calls[k]["args"]) for k in sorted(calls)], content, finish, round(time.time() - t, 1)


MODEL = MODEL or model_id()
PROMPTS = {
    "control": "Use the write_file tool once: path hello.txt, content: hello world",
    "dsml": "Use the write_file tool exactly once. path: dsml-notes.md. The content must be exactly the following "
            "text, byte for byte, with no changes:\n" + TARGET,
}
ok_all = True
seen = {}
for name, prompt in PROMPTS.items():
    for stream in (False, True):
        try:
            calls, content, finish, secs = run(prompt, stream)
        except Exception as e:  # noqa
            print(json.dumps({"case": name, "stream": stream, "error": str(e)}))
            ok_all = False
            continue
        rec = {"case": name, "stream": stream, "finish": finish, "s": secs, "n_calls": len(calls)}
        if calls:
            nm, args = calls[0]
            try:
                a = json.loads(args)
                rec["json_ok"] = True
                rec["content_arg"] = a.get("content")
                if name == "dsml":
                    rec["has_all_closers"] = all(t in (a.get("content") or "") for t in
                                                 ("</｜DSML｜ calls>", "</｜DSML｜ parameter>", "</｜DSML｜ invoke>"))
                    rec["exact"] = a.get("content") == TARGET
                    c_ = a.get("content") or ""
                    rec["kept_past_closer"] = "</｜DSML｜" in c_ and c_.rstrip().endswith("literal.")
            except Exception:  # noqa
                rec["json_ok"] = False
                rec["raw_args"] = args
        rec["text"] = (content or "")[:160]
        ok = bool(calls) and rec.get("json_ok") and (name == "control" or rec.get("kept_past_closer"))
        seen.setdefault(name, []).append(rec.get("content_arg"))
        if stream and len(set(map(str, seen[name]))) != 1:
            rec["stream_differs_from_nonstream"] = True
            ok = False
        rec["PASS"] = bool(ok)
        ok_all &= bool(ok)
        print(json.dumps(rec, ensure_ascii=False))
print(json.dumps({"model": MODEL, "base": BASE, "ALL_PASS": ok_all}))
