"""Replay DeepSeek V4.1 replies captured from a live server through the deepseek_v41 parsers, as the server runs them.

Usage: python3 replay_fixtures.py <model dir> <fixtures.json>...
Each fixture holds a request, the prompt and output token IDs, and what the server returned, whole and streamed.
As in the chat handler, the parser is built per request with the request's tools and its effective chat-template
kwargs. A whole reply is ``parse(text, request, enable_auto_tools=True, model_output_token_ids=ids)`` with the IDs
as generated (end-of-sentence included) and the text decoded with ``skip_special_tokens=True``. A stream feeds
``parse_delta`` at 1, 7 and all tokens a delta, the text decoded cumulatively the same way, with an incomplete
UTF-8 tail held back. One JSON line per case and run; ``ok`` checks what the model wrote, and for clean replies
that reasoning, content and raw arguments equal what the live server returned.
"""
import json
import sys

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager
from vllm.tokenizers import get_tokenizer

tokenizer = get_tokenizer(sys.argv[1], tokenizer_mode="deepseek_v41")
fixtures = {}
for path in sys.argv[2:]:
    fixtures.update(json.load(open(path, encoding="utf-8")))
D = "｜DSML｜"
PARSER_CLS = ParserManager.get_parser(
    tool_parser_name="deepseek_v41", reasoning_parser_name="deepseek_v41", enable_auto_tools=True
)
# What the model wrote. It ended both files with a newline, inside the value, before the genuine closer.
INTENDED = {
    "n1_file_of_closers": [("write", {"path": "notes.md", "content": f"DSML closers:\n</{D} parameter>\n</{D} calls>\n"})],
    "n2_python_line": [("write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'})],
    "th_n2_python_line": [("write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'})],
    "ok_two_params": [("write", {"path": "hello.md", "content": "hello world"})],
}
# Replies without quoted DSML must come out exactly as the live server returned them.
EQUIVALENT = ("ok_two_params", "q876_tools", "th_plain", "th_tool", "th_tool_max")


def request(fx):
    body = fx["request"]
    return ChatCompletionRequest(
        model="m", messages=body["messages"], tools=body.get("tools"), reasoning_effort=body.get("reasoning_effort")
    )


def parser(req):
    kwargs = req.build_chat_params(None, "auto").with_defaults({}).chat_template_kwargs
    return PARSER_CLS(tokenizer, req.tools, chat_template_kwargs=kwargs)


def calls_of(calls):
    out = []
    for name, args in calls:
        try:
            out.append((name, json.loads(args or "{}"), args))
        except ValueError:
            out.append((name, {"_unparsed": args}, args))
    return out


def whole(fx):
    req, ids = request(fx), fx["token_ids"]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    reasoning, content, tool_calls = parser(req).parse(text, req, enable_auto_tools=True, model_output_token_ids=ids)
    return reasoning or "", content or "", calls_of([(c.name, c.arguments) for c in tool_calls or []])


def streamed(fx, chunk):
    req, ids = request(fx), fx["token_ids"]
    p = parser(req)
    emitted, reasoning, content, parts = "", "", "", {}
    for start in range(0, len(ids), chunk):
        batch = ids[start:start + chunk]
        last = start + chunk >= len(ids)
        text = tokenizer.decode(ids[:start + len(batch)], skip_special_tokens=True)
        safe = text if last else text.rstrip("�")
        delta, emitted = safe[len(emitted):], safe
        d = p.parse_delta(delta, batch, req, prompt_token_ids=fx["prompt_token_ids"] if start == 0 else None,
                          finished=last)
        if d is None:
            continue
        reasoning += d.reasoning or ""
        content += d.content or ""
        for t in d.tool_calls or []:
            slot = parts.setdefault(t.index, {"name": "", "arguments": ""})
            if t.function:
                slot["name"] += t.function.name or ""
                slot["arguments"] += t.function.arguments or ""
    return reasoning, content, calls_of([(s["name"], s["arguments"]) for _, s in sorted(parts.items())])


summary = {}
for name, fx in fixtures.items():
    if name.endswith("_stream"):
        continue
    full = tokenizer.decode(fx["token_ids"], skip_special_tokens=True)
    runs = [("whole", whole(fx))] + [(f"stream/{c}", streamed(fx, c)) for c in (1, 7, 10000)]
    for run, (reasoning, content, calls) in runs:
        parsed = [(n, a) for n, a, _ in calls]
        if name in INTENDED:
            ok = parsed == INTENDED[name]
        elif name.startswith("q876"):
            ok = parsed == [] and content == full
        else:
            ok = True
        row = {"case": name, "run": run, "ok": ok, "calls": parsed, "reasoning": reasoning[:80], "content": content[:120]}
        if name in EQUIVALENT:
            live = fixtures[name if run == "whole" else name + "_stream"]
            same = (reasoning == (live.get("reasoning") or "") and content == (live["content"] or "")
                    and [(n, raw) for n, _, raw in calls] == [tuple(c) for c in live["tool_calls"]])
            row["equal_to_live"] = same
            row["ok"] = ok = ok and same
        summary.setdefault(name, []).append(ok)
        print(json.dumps(row, ensure_ascii=False), flush=True)
print(json.dumps({"summary": {k: all(v) for k, v in summary.items()}}))
