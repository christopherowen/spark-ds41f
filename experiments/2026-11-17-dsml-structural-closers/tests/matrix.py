"""DSML structural-closer test matrix for the deepseek_v41 overlay (candidate-60 tree).

Run inside the candidate-60 image (CPU only, no GPU, no server). Prints one JSON line per
case. Mode: VLLM_DSV41_STRUCTURAL_DSML=0 -> stock, =1 -> patched (the overlay file is bind
mounted over the image's vllm/parser/deepseek_v41.py). Compare the runs with compare.py.

Adapted from the 2026-11-06 r6x kit for the candidate-60 tree:
  * tokenizer comes from MODEL_DIR (default the checkpoint path used by the gate kit);
  * a preflight line on stderr records which vllm and which parser file are in use;
  * the 25 cases, the modes and the imports are otherwise unchanged.
"""
import hashlib
import json
import os
import sys

from transformers import AutoTokenizer
import vllm.parser.deepseek_v41 as v41mod
from vllm.tool_parsers.deepseekv41_engine_tool_parser import DeepSeekV41EngineToolParser
from vllm.parser.engine.registered_adapters import DeepSeekV41ParserReasoningAdapter

try:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
except ImportError:  # older layout (Sparks fork)
    from vllm.entrypoints.openai.protocol import ChatCompletionRequest

MODEL_DIR = os.environ.get("MODEL_DIR", "/models/DeepSeek-V4.1-Flash-1017")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
MODE = os.environ.get("VLLM_DSV41_STRUCTURAL_DSML", "1")
OVERLAY = getattr(v41mod, "OVERLAY_ID", "stock-file")


def preflight():
    import vllm

    path = getattr(v41mod, "__file__", "?")
    try:
        with open(path, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        digest = "unreadable"
    print(
        json.dumps(
            {
                "preflight": {
                    "mode": MODE,
                    "overlay_id": OVERLAY,
                    "structural_active": getattr(v41mod, "_STRUCTURAL_ACTIVE", None),
                    "vllm": getattr(vllm, "__file__", "?"),
                    "parser_file": path,
                    "parser_sha256": digest,
                    "serving_parser": hasattr(v41mod, "DeepSeekV41ServingParser"),
                    "model_dir": MODEL_DIR,
                    "python": sys.version.split()[0],
                }
            }
        ),
        file=sys.stderr,
        flush=True,
    )


preflight()

tools = [
    {"type": "function", "function": {"name": "write_file", "description": "write a file",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                    "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "set_cfg", "description": "set config",
     "parameters": {"type": "object", "properties": {"n": {"type": "integer"}, "cfg": {"type": "object"},
                                                     "tag": {"type": "string"}}, "required": ["n"]}}},
]

S, E = "<｜DSML｜ calls>", "</｜DSML｜ calls>"
IS, IE = '<｜DSML｜ invoke name="', "</｜DSML｜ invoke>"
PS, PE = "<｜DSML｜ parameter", "</｜DSML｜ parameter>"


def p(name, val, s="true"):
    attr = f' string="{s}"' if s is not None else ""
    return f'{PS} name="{name}"{attr}>{val}{PE}\n'


def wf(path, content):
    return f'{IS}write_file">\n{p("path", path)}{p("content", content)}{IE}\n'


CASES = {
    "A_control": f"Writing it.\n\n{S}\n{wf('a.txt', 'hello world')}{E}",
    "B_value_holds_calls_closer": f"Writing it.\n\n{S}\n{wf('dsml.md', 'The block ends with ' + E + ' on its own line.')}{E}",
    "C_value_holds_param_closer": f"Writing it.\n\n{S}\n{wf('dsml.md', 'A param ends with ' + PE + ' then more.')}{E}",
    "D_value_holds_invoke_closer": f"Writing it.\n\n{S}\n{wf('dsml.md', 'An invoke ends with ' + IE + ' then more.')}{E}",
    "E_two_calls_first_holds_closer": f"{S}\n{wf('one.md', 'see ' + E + ' here')}{wf('two.txt', 'second')}{E}",
    "E2_all_three_closers_multiline": f"{S}\n{wf('doc.md', 'line1 ' + PE + chr(10) + 'line2 ' + IE + chr(10) + 'line3 ' + E + chr(10) + 'end')}{E}",
    "F_orphan_closers_only": f"{IE}\n{E}",
    "G_json_answer_with_tools": '{"city": "Paris", "temp_c": 21}',
    "H_parallel_calls": f"{S}\n{wf('a.txt', 'one')}{wf('b.txt', 'two')}{E}",
    "I_string_false_values": f'{S}\n{IS}set_cfg">\n{p("n", "5", "false")}{p("cfg", json.dumps({"a": [1, 2], "b": None}), "false")}{IE}\n{E}',
    "J_string_false_json_holding_invoke_closer": f'{S}\n{IS}set_cfg">\n{p("n", "7", "false")}{p("cfg", json.dumps({"note": "a " + IE + " b"}, ensure_ascii=False), "false")}{IE}\n{E}',
    "K_missing_string_attr": f'{S}\n{IS}set_cfg">\n{p("n", "3", None)}{p("tag", "plain", None)}{IE}\n{E}',
    "L_omit_invoke_end": f"{S}\n{IS}write_file\">\n{p('path', 'a.txt')}{p('content', 'hi')}{E}",
    "M_omit_param_close_before_next_head": f'{S}\n{IS}write_file\">\n{PS} name="path" string="true">a.txt\n{p("content", "hi")}{IE}\n{E}',
    "N_omit_param_close_then_invoke_end": f'{S}\n{IS}write_file\">\n{p("path", "a.txt")}{PS} name="content" string="true">hello\n{IE}\n{E}',
    "N2_omit_param_close_then_invoke_end_eos": f'{S}\n{IS}write_file\">\n{p("path", "a.txt")}{PS} name="content" string="true">hello\n{IE}',
    "O_truncated_mid_value": f'{S}\n{IS}write_file">\n{p("path", "a.txt")}{PS} name="content" string="true">partial tex',
    "O2_truncated_after_literal_closer": f'{S}\n{IS}write_file">\n{p("path", "a.txt")}{PS} name="content" string="true">see {E} and then tru',
    "Q_ambiguous_value_holds_closer_plus_head": f"{S}\n{wf('x.md', 'a' + PE + chr(10) + PS + ' name=' + chr(34) + 'z' + chr(34) + ' string=' + chr(34) + 'true' + chr(34) + '>b')}{E}",
    "R_value_ends_with_literal_param_closer": f"{S}\n{wf('x.md', 'the tag is ' + PE)}{E}",
    "T_text_after_block": f"{S}\n{wf('a.txt', 'x')}{E}\ntrailing words",
    "U_json_body": f'{S}\n{IS}write_file">\n{json.dumps({"path": "j.txt", "content": "json body"})}\n{IE}\n{E}',
    "U2_json_body_holds_invoke_closer": f'{S}\n{IS}write_file">\n{json.dumps({"path": "j.txt", "content": "has " + IE + " inside"}, ensure_ascii=False)}\n{IE}\n{E}',
}

REASONING_CASES = {
    "P_reasoning_mentions_closers": f"<think>I could write {E} or {IE} in prose here.</think>Plain answer, no tools.",
    "P2_reasoning_then_call": f"<think>plan: use {PE} maybe</think>{S}\n{wf('a.txt', 'ok')}{E}",
}


def req(stream=False):
    return ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], tools=tools, stream=stream)


def parsed(args):
    try:
        return json.loads(args) if args else None, True
    except Exception:  # noqa
        return args, False


def run_nonstream(text):
    r = req()
    tp = DeepSeekV41EngineToolParser(tok, r.tools, chat_template_kwargs={"thinking": False})
    out = tp.extract_tool_calls(text, r)
    calls = []
    for c in out.tool_calls:
        a, ok = parsed(c.function.arguments)
        calls.append({"name": c.function.name, "args": a, "json_ok": ok})
    return {"tools_called": out.tools_called, "calls": calls, "content": out.content}


def run_stream(text, chunk):
    r = req(stream=True)
    tp = DeepSeekV41EngineToolParser(tok, r.tools, chat_template_kwargs={"thinking": False})
    ids = tok.encode(text, add_special_tokens=False)
    prev_text, prev_ids, content, calls = "", [], "", {}

    def absorb(d):
        nonlocal content
        if d is None:
            return
        if d.content:
            content += d.content
        for tc in d.tool_calls or []:
            c = calls.setdefault(tc.index, {"name": None, "args": ""})
            if tc.function and tc.function.name:
                c["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                c["args"] += tc.function.arguments

    for i in range(0, len(ids), chunk):
        cur_ids = ids[: i + chunk]
        cur_text = tok.decode(cur_ids)
        delta = cur_text[len(prev_text):]
        absorb(tp.extract_tool_calls_streaming(prev_text, cur_text, delta, prev_ids, cur_ids, ids[i:i + chunk], r))
        prev_text, prev_ids = cur_text, cur_ids
    if hasattr(tp, "finish_streaming"):
        absorb(tp.finish_streaming())
    out = []
    for k in sorted(calls):
        a, ok = parsed(calls[k]["args"])
        out.append({"name": calls[k]["name"], "args": a, "json_ok": ok})
    return {"calls": out, "content": content}


def run_reasoning(text):
    r = req()
    rp = DeepSeekV41ParserReasoningAdapter(tok, chat_template_kwargs={"thinking": True})
    reasoning, content = rp.extract_reasoning(text, r)
    return {"reasoning": reasoning, "content": content}


for name, text in CASES.items():
    rec = {"case": name, "mode": MODE, "overlay": OVERLAY}
    try:
        rec["nonstream"] = run_nonstream(text)
    except Exception as e:  # noqa
        rec["nonstream"] = {"error": f"{type(e).__name__}: {e}"}
    for chunk in (1, 3, 7, 10**6):
        try:
            rec[f"stream{chunk if chunk < 10**6 else 'All'}"] = run_stream(text, chunk)
        except Exception as e:  # noqa
            rec[f"stream{chunk if chunk < 10**6 else 'All'}"] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(rec, ensure_ascii=False), flush=True)

for name, text in REASONING_CASES.items():
    rec = {"case": name, "mode": MODE, "overlay": OVERLAY}
    try:
        rec["reasoning"] = run_reasoning(text)
    except Exception as e:  # noqa
        rec["reasoning"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        rec["nonstream"] = run_nonstream(text.split("</think>", 1)[1])
    except Exception as e:  # noqa
        rec["nonstream"] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(rec, ensure_ascii=False), flush=True)
