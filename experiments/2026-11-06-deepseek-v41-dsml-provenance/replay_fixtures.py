"""Replay DeepSeek V4.1 replies captured from a live server through the deepseek_v41 parsers.

Usage: python3 replay_fixtures.py <fixtures.json> <model dir>
Each fixture holds a request, the prompt and output token IDs, and what the server returned. The replay feeds the
output token IDs through ParserManager's deepseek_v41 tool and reasoning parsers: whole (``parse`` with the token IDs)
and streamed (``parse_delta`` at several chunk sizes, the text decoded cumulatively and an incomplete UTF-8 tail held
back, as the server's detokenizer does). It prints one JSON line per case and run, with what was intended.
"""
import json
import sys

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser.parser_manager import ParserManager

try:
    from vllm.tokenizers import get_tokenizer

    def load_tokenizer(path):
        return get_tokenizer(path, tokenizer_mode="deepseek_v41")
except ImportError:  # pragma: no cover
    from transformers import AutoTokenizer

    def load_tokenizer(path):
        return AutoTokenizer.from_pretrained(path)

fixtures = json.load(open(sys.argv[1], encoding="utf-8"))
tokenizer = load_tokenizer(sys.argv[2])
D = "｜DSML｜"
# What the model wrote. It ended both files with a newline, inside the value, before the genuine closer.
INTENDED = {
    "n1_file_of_closers": [("write", {"path": "notes.md", "content": f"DSML closers:\n</{D} parameter>\n</{D} calls>\n"})],
    "n2_python_line": [("write", {"path": "consts.py", "content": f'TOOL_END = "</{D} calls>"\n'})],
    "ok_two_params": [("write", {"path": "hello.md", "content": "hello world"})],
}
# Replies without quoted DSML must come out exactly as the live server returned them (content and raw arguments).
EQUIVALENT = ("ok_two_params", "q876_tools")


def parser():
    cls = ParserManager.get_parser(
        tool_parser_name="deepseek_v41", reasoning_parser_name="deepseek_v41", enable_auto_tools=True
    )
    return cls(tokenizer, chat_template_kwargs={"thinking": False})


def request(fx):
    body = fx["request"]
    return ChatCompletionRequest(model="m", messages=body["messages"], tools=body.get("tools"))


def calls_of(calls):
    out = []
    for name, args in calls:
        try:
            out.append((name, json.loads(args or "{}"), args))
        except ValueError:
            out.append((name, {"_unparsed": args}, args))
    return out


def whole(fx):
    ids = [t for t in fx["token_ids"] if t != tokenizer.eos_token_id]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    _, content, tool_calls = parser().parse(text, request(fx), enable_auto_tools=True, model_output_token_ids=ids)
    return content or "", calls_of([(c.name, c.arguments) for c in tool_calls or []])


def streamed(fx, chunk):
    p, req = parser(), request(fx)
    ids = [t for t in fx["token_ids"] if t != tokenizer.eos_token_id]
    emitted, content, parts = "", "", {}
    for start in range(0, len(ids), chunk):
        batch = ids[start:start + chunk]
        text = tokenizer.decode(ids[:start + len(batch)], skip_special_tokens=False)
        last = start + chunk >= len(ids)
        safe = text if last else text.rstrip("�")
        delta, emitted = safe[len(emitted):], safe
        d = p.parse_delta(delta, batch, req, prompt_token_ids=fx["prompt_token_ids"] if start == 0 else None,
                          finished=last)
        if d is None:
            continue
        content += d.content or ""
        for t in d.tool_calls or []:
            slot = parts.setdefault(t.index, {"name": "", "arguments": ""})
            if t.function:
                slot["name"] += t.function.name or ""
                slot["arguments"] += t.function.arguments or ""
    return content, calls_of([(s["name"], s["arguments"]) for _, s in sorted(parts.items())])


summary = {}
for name, fx in fixtures.items():
    if name.endswith("_stream"):
        continue
    ids = [t for t in fx["token_ids"] if t != tokenizer.eos_token_id]
    full = tokenizer.decode(ids, skip_special_tokens=False)
    want_calls = INTENDED.get(name, [])
    want_content = None if name in INTENDED else full
    for run, (content, calls) in [("whole", whole(fx))] + [(f"stream/{c}", streamed(fx, c)) for c in (1, 7, 10000)]:
        ok = [(n, a) for n, a, _ in calls] == want_calls and (want_content is None or content == want_content)
        row = {"case": name, "run": run, "ok": ok, "calls": [(n, a) for n, a, _ in calls], "content": content[:160]}
        if name in EQUIVALENT:
            live = fixtures[name if run == "whole" else name + "_stream"]
            same = (content == (live["content"] or "")
                    and [(n, raw) for n, _, raw in calls] == [tuple(c) for c in live["tool_calls"]])
            row["equal_to_live"] = same
            ok = ok and same
            row["ok"] = ok
        summary.setdefault(name, []).append(ok)
        print(json.dumps(row, ensure_ascii=False), flush=True)
print(json.dumps({"summary": {k: all(v) for k, v in summary.items()}}))
