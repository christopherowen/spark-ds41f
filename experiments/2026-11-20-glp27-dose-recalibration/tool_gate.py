#!/usr/bin/env python3
"""Steered tool-calling qualification gate for a GLP arm.

A GLP vector's published calibration gates refusal suites only; nothing in it
exercises tool calling, and a dose can pass every refusal gate while derailing
the DSML tool channel (GLP-27 at its published alpha 3.5 did exactly that in
production agent traffic). This gate is the missing check: run it against a
booted arm before the arm is qualified, and again before any dose increase.

It runs 76 scenarios against one served model name — single calls, parallel
calls, selection from 16 tools, schema stress (nested objects, arrays, enums,
booleans, arrays of objects, deep nesting), multi-turn chains with canned tool
results, tool_choice modes (required / none / named, plus no-call controls),
streaming, temperature-0.7 repeats, and agent-style (terminal/file/search)
calls — and scores each mechanically: call presence, name multiset, argument
values, JSON validity, chain completion.

Two verdicts are reported:

- the strict score (scenarios passed / 76), comparable across runs;
- the gate verdict, which fails only on *derailment signatures*: no call
  where one is required (the vanish: finish=length with the budget burned, a
  silent no-call stop, a preamble with no call), malformed JSON arguments,
  empty arguments, a dropped expected argument, an exact duplicate call, or
  DSML markup leaked into an argument string or the content. A scenario can
  fail strictly without derailing (the model makes a second, different, valid
  call where exactly one was demanded; a semantic value miss) — those are
  quality misses, reported, but they do not fail the gate: the gate exists to
  catch a steered name whose tool channel is broken, not to grade taste.

Usage:
    tool_gate.py [--base URL] [--model NAME] [--control NAME] [--out PATH]

--base defaults to $TOOL_GATE_BASE or http://127.0.0.1:8000/v1; --model to
$TOOL_GATE_MODEL or deepseek-v4.1-flash-weightless. --control NAME also runs
a 20-scenario subset against a second (usually stock) name as a control.
Exit code: 0 = gate passes, 1 = a derailment signature was found, 2 = the
endpoint could not be exercised. Stdlib only.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error
from collections import Counter

BASE = os.environ.get("TOOL_GATE_BASE", "http://127.0.0.1:8000/v1")
WL = os.environ.get("TOOL_GATE_MODEL", "deepseek-v4.1-flash-weightless")
STOCK = "deepseek-v4.1-flash"

def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

S = {"type": "string"}; N = {"type": "number"}; I = {"type": "integer"}; B = {"type": "boolean"}

TOOLS = {
 "get_weather": fn("get_weather", "Get current weather for a city", {"city": S}, ["city"]),
 "get_time": fn("get_time", "Get current local time for a city", {"city": S}, ["city"]),
 "add": fn("add", "Add two numbers", {"a": N, "b": N}, ["a", "b"]),
 "multiply": fn("multiply", "Multiply two numbers", {"a": N, "b": N}, ["a", "b"]),
 "convert_units": fn("convert_units", "Convert a value between units", {"value": N, "from_unit": S, "to_unit": S}, ["value", "from_unit", "to_unit"]),
 "get_stock_price": fn("get_stock_price", "Get the latest stock price for a ticker", {"ticker": S}, ["ticker"]),
 "search_news": fn("search_news", "Search recent news articles", {"query": S, "max_results": I}, ["query"]),
 "send_email": fn("send_email", "Send an email", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
 "create_event": fn("create_event", "Create a calendar event", {"title": S, "date": S, "time": S}, ["title", "date"]),
 "translate": fn("translate", "Translate text into a target language", {"text": S, "target_language": S}, ["text", "target_language"]),
 "define_word": fn("define_word", "Look up the dictionary definition of a word", {"word": S}, ["word"]),
 "flight_status": fn("flight_status", "Get the status of a flight", {"flight_number": S}, ["flight_number"]),
 "distance": fn("distance", "Get the distance between two places", {"from_place": S, "to_place": S}, ["from_place", "to_place"]),
 "book_table": fn("book_table", "Book a restaurant table", {"restaurant": S, "party_size": I, "date": S}, ["restaurant", "party_size", "date"]),
 "set_timer": fn("set_timer", "Set a countdown timer", {"minutes": I, "label": S}, ["minutes"]),
 "play_music": fn("play_music", "Play a song or playlist", {"query": S}, ["query"]),
}
def T(*names): return [TOOLS[n] for n in names]

# Hermes-style tools
HERMES_SYS = ("You are Hermes, an agent with tool access. Use tools to complete tasks. "
              "When a task needs a shell command, file access, or a search, call the matching tool.")
H_TOOLS = [
 fn("terminal", "Run a shell command", {"command": S}, ["command"]),
 fn("read_file", "Read a file's contents", {"path": S}, ["path"]),
 fn("write_file", "Write content to a file", {"path": S, "content": S}, ["path", "content"]),
 fn("search_files", "Search file contents for a pattern", {"pattern": S, "path": S}, ["pattern"]),
 fn("patch", "Apply a text replacement in a file", {"path": S, "old_text": S, "new_text": S}, ["path", "old_text", "new_text"]),
 fn("web_search", "Search the web", {"query": S}, ["query"]),
]

# scenario: dict(kind, prompt, tools, expect=[(name, {arg: expected-or-type})...] or "none",
#                tool_choice, temp, stream, system, chain=[canned results...])
SC = []
def single(prompt, tools, expect, **kw): SC.append(dict(kind="single", prompt=prompt, tools=tools, expect=expect, **kw))
def parallel(prompt, tools, expect, **kw): SC.append(dict(kind="parallel", prompt=prompt, tools=tools, expect=expect, **kw))
def select(prompt, tools, expect, **kw): SC.append(dict(kind="select", prompt=prompt, tools=tools, expect=expect, **kw))
def schema(prompt, tools, expect, **kw): SC.append(dict(kind="schema", prompt=prompt, tools=tools, expect=expect, **kw))

single("What is the weather in Tokyo right now?", T("get_weather"), [("get_weather", {"city": "Tokyo"})])
single("What's the local time in Paris?", T("get_time"), [("get_time", {"city": "Paris"})])
single("What is 17 plus 25?", T("add"), [("add", {"a": 17, "b": 25})])
single("Multiply 6 by 9.", T("multiply"), [("multiply", {"a": 6, "b": 9})])
single("Convert 100 kilometers to miles.", T("convert_units"), [("convert_units", {"value": 100, "from_unit": "kilometers", "to_unit": "miles"})])
single("What is Apple's stock price? The ticker is AAPL.", T("get_stock_price"), [("get_stock_price", {"ticker": "AAPL"})])
single("Search the news for the latest on fusion energy, up to 5 results.", T("search_news"), [("search_news", {"query": "fusion energy"})])
single("Send an email to bob@example.com with subject 'Lunch' and body 'See you at noon.'", T("send_email"),
       [("send_email", {"to": "bob@example.com", "subject": "Lunch"})])
single("Create a calendar event titled 'Dentist' on 2026-12-01 at 09:30.", T("create_event"),
       [("create_event", {"title": "Dentist", "date": "2026-12-01"})])
single("Translate 'good morning' into French.", T("translate"), [("translate", {"text": "good morning", "target_language": "French"})])
single("What does the word 'serendipity' mean?", T("define_word"), [("define_word", {"word": "serendipity"})])
single("Is flight DL2924 on time?", T("flight_status"), [("flight_status", {"flight_number": "DL2924"})])
single("How far is it from Denver to Boulder?", T("distance"), [("distance", {"from_place": "Denver", "to_place": "Boulder"})])
single("Book a table for 4 at Luigi's on 2026-11-25.", T("book_table"), [("book_table", {"restaurant": "Luigi's", "party_size": 4})])
single("Set a timer for 20 minutes.", T("set_timer"), [("set_timer", {"minutes": 20})])

parallel("What's the weather in Tokyo and the local time in Tokyo?", T("get_weather", "get_time"),
         [("get_weather", {"city": "Tokyo"}), ("get_time", {"city": "Tokyo"})])
parallel("Add 3 and 4, and also multiply 5 by 6.", T("add", "multiply"),
         [("add", {"a": 3, "b": 4}), ("multiply", {"a": 5, "b": 6})])
parallel("Get the weather in London, Paris, and Berlin.", T("get_weather"),
         [("get_weather", {"city": "London"}), ("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Berlin"})])
parallel("Check AAPL's stock price and search news about Apple.", T("get_stock_price", "search_news"),
         [("get_stock_price", {"ticker": "AAPL"}), ("search_news", {"query": "Apple"})])
parallel("What's the time in Sydney and the weather in Sydney, and how far is Sydney from Melbourne?",
         T("get_time", "get_weather", "distance"),
         [("get_time", {"city": "Sydney"}), ("get_weather", {"city": "Sydney"}), ("distance", {"from_place": "Sydney", "to_place": "Melbourne"})])
parallel("Translate 'thank you' into German and define the word 'gratitude'.", T("translate", "define_word"),
         [("translate", {"target_language": "German"}), ("define_word", {"word": "gratitude"})])
parallel("Set a 10 minute timer labelled 'tea' and play some jazz.", T("set_timer", "play_music"),
         [("set_timer", {"minutes": 10}), ("play_music", {"query": "jazz"})])
parallel("What is 100 plus 200 and 1000 plus 2000?", T("add"),
         [("add", {"a": 100, "b": 200}), ("add", {"a": 1000, "b": 2000})])
parallel("Get weather for Oslo and check flight DL10 status.", T("get_weather", "flight_status"),
         [("get_weather", {"city": "Oslo"}), ("flight_status", {"flight_number": "DL10"})])
parallel("Convert 5 miles to kilometers and 10 miles to kilometers.", T("convert_units"),
         [("convert_units", {"value": 5}), ("convert_units", {"value": 10})])

ALL16 = T(*TOOLS.keys())
select("I need the current weather in Chicago.", ALL16, [("get_weather", {"city": "Chicago"})])
select("Please add 42 and 58 for me.", ALL16, [("add", {"a": 42, "b": 58})])
select("Look up what 'ephemeral' means.", ALL16, [("define_word", {"word": "ephemeral"})])
select("Has flight UA123 departed yet?", ALL16, [("flight_status", {"flight_number": "UA123"})])
select("Email alice@example.com: subject 'Report', body 'Attached is the report.'", ALL16,
       [("send_email", {"to": "alice@example.com", "subject": "Report"})])
select("How long does it take to drive from Austin to Dallas? Just give me the distance.", ALL16,
       [("distance", {"from_place": "Austin", "to_place": "Dallas"})])
select("Play the song 'Imagine' by John Lennon.", ALL16, [("play_music", {"query": "Imagine"})])
select("Book a table for 2 at Sushi Den on 2026-12-05.", ALL16, [("book_table", {"party_size": 2, "date": "2026-12-05"})])
select("Search news about electric vehicles.", ALL16, [("search_news", {"query": "electric vehicles"})])
select("What time is it in Tokyo right now?", ALL16, [("get_time", {"city": "Tokyo"})])

# schema-stress tools
NESTED = fn("create_user", "Create a user profile", {
    "name": S, "address": {"type": "object", "properties": {"street": S, "city": S, "zip": S}, "required": ["street", "city"]},
    "age": I}, ["name", "address"])
ARRAY = fn("add_tags", "Add tags to a post", {"post_id": I, "tags": {"type": "array", "items": {"type": "string"}}}, ["post_id", "tags"])
ENUM = fn("set_priority", "Set task priority", {"task": S, "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]}}, ["task", "priority"])
BOOL = fn("configure", "Configure a service", {"service": S, "enabled": B, "retries": I}, ["service", "enabled"])
ARROBJ = fn("add_line_items", "Add line items to an invoice", {"invoice_id": S,
    "items": {"type": "array", "items": {"type": "object", "properties": {"sku": S, "qty": I}, "required": ["sku", "qty"]}}}, ["invoice_id", "items"])
NUMS = fn("stats", "Compute statistics", {"values": {"type": "array", "items": {"type": "number"}}, "operation": S}, ["values", "operation"])
OPT = fn("remind", "Create a reminder", {"text": S, "at": S, "repeat": S}, ["text"])
DEEP = fn("deploy", "Deploy a service", {"service": S, "config": {"type": "object", "properties": {
    "env": {"type": "object", "properties": {"region": S, "replicas": I}, "required": ["region"]},
    "debug": B}, "required": ["env"]}}, ["service", "config"])

schema("Create a user named Jane Doe, address 12 Oak St, Springfield, zip 12345, age 34.", [NESTED],
       [("create_user", {"name": "Jane Doe"})])
schema("Add tags 'python', 'ai', and 'tools' to post 42.", [ARRAY], [("add_tags", {"post_id": 42})])
schema("Set the priority of task 'Fix login bug' to high.", [ENUM], [("set_priority", {"task": "Fix login bug", "priority": "high"})])
schema("Configure the cache service: enabled true, retries 3.", [BOOL], [("configure", {"service": "cache", "enabled": True})])
schema("Add line items to invoice INV-9: 2 of SKU A1 and 1 of SKU B2.", [ARROBJ], [("add_line_items", {"invoice_id": "INV-9"})])
schema("Compute the mean of 1.5, 2.5, and 3.0.", [NUMS], [("stats", {"operation": "mean"})])
schema("Remind me to 'call mom' at 18:00.", [OPT], [("remind", {"text": "call mom"})])
schema("Deploy the api service with config env region us-east, replicas 2, debug false.", [DEEP],
       [("deploy", {"service": "api"})])

# chains: (first prompt, tools, first expect, [ (canned result, next expect or "final") ... ])
CHAINS = [
 dict(prompt="What's the weather in Tokyo?", tools=T("get_weather", "get_time"),
      first=[("get_weather", {"city": "Tokyo"})],
      steps=[('{"temp_c": 18, "condition": "rain"}', [("get_time", {"city": "Tokyo"})], "And what time is it there?"),
             ('{"time": "21:04"}', "final", None)]),
 dict(prompt="Add 2 and 3.", tools=T("add", "multiply"),
      first=[("add", {"a": 2, "b": 3})],
      steps=[('{"result": 5}', [("multiply", {"a": 5, "b": 4})], "Now multiply that result by 4."),
             ('{"result": 20}', "final", None)]),
 dict(prompt="Search news about Mars missions.", tools=T("search_news", "define_word"),
      first=[("search_news", {"query": "Mars"})],
      steps=[('{"articles": [{"title": "Mars sample return delayed"}]}', "final", None)]),
 dict(prompt="Get the stock price of MSFT.", tools=T("get_stock_price", "convert_units"),
      first=[("get_stock_price", {"ticker": "MSFT"})],
      steps=[('{"price": 510.25, "currency": "USD"}', "final", None)]),
 dict(prompt="What time is it in London?", tools=T("get_time", "get_weather"),
      first=[("get_time", {"city": "London"})],
      steps=[('{"time": "13:00"}', [("get_weather", {"city": "London"})], "And the weather there?"),
             ('{"temp_c": 11}', "final", None)]),
 dict(prompt="Translate 'hello' into Spanish.", tools=T("translate"),
      first=[("translate", {"text": "hello", "target_language": "Spanish"})],
      steps=[('{"translation": "hola"}', "final", None)]),
 dict(prompt="How far from Paris to Lyon?", tools=T("distance", "convert_units"),
      first=[("distance", {"from_place": "Paris", "to_place": "Lyon"})],
      steps=[('{"distance_km": 465}', [("convert_units", {"value": 465, "to_unit": "miles"})], "Convert that to miles."),
             ('{"value": 289}', "final", None)]),
 dict(prompt="Define 'quintessential'.", tools=T("define_word", "search_news"),
      first=[("define_word", {"word": "quintessential"})],
      steps=[('{"definition": "representing the perfect example"}', "final", None)]),
]

MODES = [
 dict(kind="mode-required", prompt="What is the weather in Tokyo?", tools=T("get_weather", "get_time"),
      expect=[("get_weather", {"city": "Tokyo"})], tool_choice="required"),
 dict(kind="mode-none", prompt="What is the weather in Tokyo?", tools=T("get_weather"),
      expect="none", tool_choice="none"),
 dict(kind="mode-named", prompt="Tell me something about Tokyo.", tools=T("get_weather", "get_time"),
      expect=[("get_time", {"city": "Tokyo"})], tool_choice={"type": "function", "function": {"name": "get_time"}}),
 dict(kind="mode-nocall", prompt="Write a haiku about autumn leaves.", tools=T("get_weather", "get_time"),
      expect="none", tool_choice=None),
 dict(kind="mode-nocall2", prompt="Explain photosynthesis in two sentences.", tools=ALL16,
      expect="none", tool_choice=None),
 dict(kind="mode-required2", prompt="Add 1 and 2, and get the weather in Rome.", tools=T("add", "get_weather"),
      expect=[("add", {"a": 1, "b": 2}), ("get_weather", {"city": "Rome"})], tool_choice="required"),
]

STREAM_PROMPTS = [
 ("What is the weather in Tokyo right now?", T("get_weather"), [("get_weather", {"city": "Tokyo"})]),
 ("Add 11 and 31.", T("add"), [("add", {"a": 11, "b": 31})]),
 ("Get the weather in Madrid and the time in Madrid.", T("get_weather", "get_time"),
  [("get_weather", {"city": "Madrid"}), ("get_time", {"city": "Madrid"})]),
 ("Define the word 'luminous'.", ALL16, [("define_word", {"word": "luminous"})]),
 ("Is flight DL10 on time?", ALL16, [("flight_status", {"flight_number": "DL10"})]),
]

T07_IDX = [0, 2, 4, 15, 17, 22, 30, 40]  # indices into SC for temp-0.7 repeats

HERMES_SC = [
 ("Check the disk usage on this machine.", [("terminal", {"command": "df"})]),
 ("Read the file /etc/hostname and tell me what it says.", [("read_file", {"path": "/etc/hostname"})]),
 ("Search the project in /home/user/app for the string 'TODO'.", [("search_files", {"pattern": "TODO"})]),
 ("Write 'hello world' to the file /tmp/greeting.txt.", [("write_file", {"path": "/tmp/greeting.txt"})]),
 ("In /home/user/app/config.py, replace 'DEBUG = True' with 'DEBUG = False'.", [("patch", {"path": "/home/user/app/config.py"})]),
 ("Search the web for the vLLM documentation on tool calling.", [("web_search", {"query": "vLLM"})]),
]

# ---------- client ----------
def chat(model, messages, tools=None, tool_choice="omit", max_tokens=1024, temperature=0.0, stream=False, timeout=600):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if tools is not None: body["tools"] = tools
    if tool_choice != "omit" and tool_choice is not None: body["tool_choice"] = tool_choice
    if stream: body["stream"] = True
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r), time.time() - t0
    # streaming: accumulate
    calls = {}; content = ""; finish = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"): continue
            payload = line[5:].strip()
            if payload == "[DONE]": break
            try: d = json.loads(payload)
            except Exception: continue
            ch = d["choices"][0]
            if ch.get("finish_reason"): finish = ch["finish_reason"]
            delta = ch.get("delta") or {}
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                i = tc.get("index", 0)
                slot = calls.setdefault(i, {"name": "", "arguments": ""})
                f = tc.get("function") or {}
                slot["name"] += f.get("name") or ""
                slot["arguments"] += f.get("arguments") or ""
    msg = {"content": content, "tool_calls": [{"function": {"name": calls[i]["name"], "arguments": calls[i]["arguments"]}} for i in sorted(calls)] or None}
    return {"choices": [{"finish_reason": finish, "message": msg}], "usage": {}}, time.time() - t0

def norm(v):
    if isinstance(v, str): return v.strip().lower()
    return v

def check_calls(msg, expect):
    """expect: 'none' or list of (name, {arg: expected}). Returns (pass, detail)."""
    calls = msg.get("tool_calls") or []
    got = []
    for c in calls:
        f = c["function"]
        try: args = json.loads(f["arguments"])
        except Exception: args = None
        got.append((f["name"], args, f["arguments"]))
    if expect == "none":
        return (len(got) == 0), f"expected no calls, got {[g[0] for g in got]}"
    if len(got) != len(expect):
        return False, f"expected {len(expect)} calls {[e[0] for e in expect]}, got {len(got)}: {[g[0] for g in got]}"
    exp_names = Counter(e[0] for e in expect); got_names = Counter(g[0] for g in got)
    if exp_names != got_names:
        return False, f"name multiset mismatch: expected {dict(exp_names)}, got {dict(got_names)}"
    used = [False] * len(got)
    for name, eargs in expect:
        ok = False
        for i, (gname, gargs, raw) in enumerate(got):
            if used[i] or gname != name: continue
            if gargs is None: continue
            good = True
            for k, ev in eargs.items():
                gv = gargs.get(k)
                if gv is None: good = False; break
                if isinstance(ev, str):
                    if ev.lower() not in str(gv).lower(): good = False; break
                elif isinstance(ev, bool):
                    if gv is not ev: good = False; break
                elif isinstance(ev, (int, float)):
                    try:
                        if abs(float(gv) - float(ev)) > 1e-9: good = False; break
                    except Exception: good = False; break
            if good: ok = True; used[i] = True; break
        if not ok: return False, f"no matching call for {name} {eargs}; got {[(g[0], g[1]) for g in got]}"
    return True, "ok"

def run_simple(model, sc, idx, stream=False, temp=None):
    msgs = []
    if sc.get("system"): msgs.append({"role": "system", "content": sc["system"]})
    msgs.append({"role": "user", "content": sc["prompt"]})
    tc = sc.get("tool_choice", "omit")
    try:
        d, dt = chat(model, msgs, tools=sc["tools"], tool_choice=tc,
                     temperature=sc.get("temp", 0.0) if temp is None else temp, stream=stream)
        ch = d["choices"][0]; msg = ch["message"]
        ok, detail = check_calls(msg, sc["expect"])
        return {"idx": idx, "kind": sc["kind"] + ("-stream" if stream else "") + (f"-t{temp}" if temp else ""),
                "pass": ok, "detail": detail, "finish": ch["finish_reason"], "expect": sc["expect"],
                "content_head": (msg.get("content") or "")[:100],
                "raw_calls": [{"name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in (msg.get("tool_calls") or [])],
                "usage": d.get("usage"), "secs": round(dt, 1)}
    except Exception as e:
        return {"idx": idx, "kind": sc["kind"], "pass": False, "detail": f"ERROR {e}", "finish": None,
                "content_head": "", "raw_calls": [], "usage": None, "secs": 0}

def run_chain(model, ci, chain):
    msgs = [{"role": "user", "content": chain["prompt"]}]
    log = []
    all_calls = []
    def collect(msg):
        all_calls.extend({"name": c["function"]["name"], "arguments": c["function"]["arguments"]}
                         for c in (msg.get("tool_calls") or []))
    try:
        d, dt = chat(model, msgs, tools=chain["tools"])
        msg = d["choices"][0]["message"]
        collect(msg)
        ok, detail = check_calls(msg, chain["first"])
        log.append({"step": 0, "pass": ok, "detail": detail})
        if not ok: return {"idx": ci, "kind": "chain", "pass": False, "detail": f"step0: {detail}", "steps": log,
                           "raw_calls": all_calls, "expect": chain["first"]}
        msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls")})
        for si, (result, nxt, followup) in enumerate(chain["steps"]):
            call = (msg.get("tool_calls") or [{}])[0]
            msgs.append({"role": "tool", "tool_call_id": call.get("id", f"call{si}"), "content": result})
            if followup: msgs.append({"role": "user", "content": followup})
            d, dt = chat(model, msgs, tools=chain["tools"])
            msg = d["choices"][0]["message"]
            collect(msg)
            if nxt == "final":
                ok = not (msg.get("tool_calls"))
                detail = "final prose, no further call" if ok else f"unexpected extra call: {[c['function']['name'] for c in msg['tool_calls']]}"
                log.append({"step": si + 1, "pass": ok, "detail": detail,
                            "content_head": (msg.get("content") or "")[:100]})
                return {"idx": ci, "kind": "chain", "pass": ok, "detail": detail, "steps": log,
                        "raw_calls": all_calls, "expect": chain["first"]}
            ok, detail = check_calls(msg, nxt)
            log.append({"step": si + 1, "pass": ok, "detail": detail})
            if not ok: return {"idx": ci, "kind": "chain", "pass": False, "detail": f"step{si+1}: {detail}", "steps": log,
                               "raw_calls": all_calls, "expect": chain["first"]}
            msgs.append({"role": "assistant", "content": msg.get("content"), "tool_calls": msg.get("tool_calls")})
        return {"idx": ci, "kind": "chain", "pass": True, "detail": "ok", "steps": log,
                "raw_calls": all_calls, "expect": chain["first"]}
    except Exception as e:
        return {"idx": ci, "kind": "chain", "pass": False, "detail": f"ERROR {e}", "steps": log,
                "raw_calls": all_calls, "expect": chain["first"]}

DSML_MARKERS = ("DSML", "｜", "<|")

def derailments(r):
    """The derailment signatures in one scenario result (see the header).

    Fires only on positive mechanical evidence in the recorded response; a
    strict failure with no signature here is a quality miss, not a derailed
    tool channel.
    """
    sigs = []
    detail = r.get("detail") or ""
    if detail.startswith("ERROR"):
        return ["request error (the endpoint could not be exercised)"]
    expect = r.get("expect")
    content = r.get("content_head") or ""
    if any(m in content for m in DSML_MARKERS):
        sigs.append("DSML markup leaked into content")
    if "raw_calls" in r:
        calls = r.get("raw_calls") or []
        parsed = []
        for c in calls:
            raw = c.get("arguments") or ""
            if any(m in raw for m in DSML_MARKERS):
                sigs.append("DSML markup leaked into argument string")
            try:
                parsed.append((c.get("name"), json.loads(raw)))
            except Exception:
                parsed.append((c.get("name"), None))
                sigs.append("malformed JSON arguments")
        if isinstance(expect, list) and expect:
            if not calls:
                vanish = " (finish=length, budget burned)" if r.get("finish") == "length" else ""
                sigs.append(f"no call emitted{vanish}")
            else:
                for name, eargs in expect:
                    same = [a for (n, a) in parsed if n == name and a is not None]
                    if same and all(a == {} for a in same) and eargs:
                        sigs.append(f"empty arguments on {name}")
                    elif same and not any(all(k in a for k in eargs) for a in same):
                        sigs.append(f"dropped expected argument on {name}")
                counts = Counter((c.get("name"), c.get("arguments")) for c in calls)
                expected_names = Counter(e[0] for e in expect)
                for (name, _raw), n in counts.items():
                    # Within one response only: a chain's calls span steps, and
                    # the same call may legitimately recur across them.
                    if n > 1 and n > expected_names.get(name, 0) and not r["kind"].startswith("chain"):
                        sigs.append(f"exact duplicate call to {name}")
    if r["kind"].startswith("chain"):
        for step in r.get("steps") or []:
            if "got 0" in (step.get("detail") or ""):
                sigs.append("no call emitted (chain step)")
                break
    return sorted(set(sigs))

def run_suite(model, label):
    results = []
    for i, sc in enumerate(SC):
        r = run_simple(model, sc, i); results.append(r)
        print(f"[{label}] {r['kind']} #{i} {'PASS' if r['pass'] else 'FAIL'} {r['detail'][:80]}", flush=True)
    for j, (prompt, tools, expect) in enumerate(STREAM_PROMPTS):
        r = run_simple(model, dict(kind="stream", prompt=prompt, tools=tools, expect=expect), 100 + j, stream=True)
        results.append(r); print(f"[{label}] stream #{j} {'PASS' if r['pass'] else 'FAIL'} {r['detail'][:80]}", flush=True)
    for j, i in enumerate(T07_IDX):
        r = run_simple(model, SC[i], 200 + i, temp=0.7)
        results.append(r); print(f"[{label}] t07 #{i} {'PASS' if r['pass'] else 'FAIL'} {r['detail'][:80]}", flush=True)
    for j, (prompt, expect) in enumerate(HERMES_SC):
        sc = dict(kind="hermes", prompt=prompt, tools=H_TOOLS, expect=expect, system=HERMES_SYS)
        r = run_simple(model, sc, 300 + j)
        results.append(r); print(f"[{label}] hermes #{j} {'PASS' if r['pass'] else 'FAIL'} {r['detail'][:80]}", flush=True)
    for ci, chain in enumerate(CHAINS):
        r = run_chain(model, 400 + ci, chain)
        results.append(r); print(f"[{label}] chain #{ci} {'PASS' if r['pass'] else 'FAIL'} {r['detail'][:80]}", flush=True)
    for mi, m in enumerate(MODES):
        r = run_simple(model, m, 500 + mi)
        results.append(r); print(f"[{label}] {m['kind']} {'PASS' if r['pass'] else 'FAIL'} {r['detail'][:80]}", flush=True)
    return results

def run_control(model, label):
    results = []
    subset = [0, 1, 2, 5, 15, 16, 22, 25, 30, 35, 40, 42]
    for i in subset:
        r = run_simple(model, SC[i], i); results.append(r)
        print(f"[{label}] control #{i} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    for ci in [0, 1, 4]:
        r = run_chain(model, 400 + ci, CHAINS[ci]); results.append(r)
        print(f"[{label}] control-chain #{ci} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    for j, (prompt, expect) in enumerate(HERMES_SC[:3]):
        sc = dict(kind="hermes", prompt=prompt, tools=H_TOOLS, expect=expect, system=HERMES_SYS)
        r = run_simple(model, sc, 300 + j); results.append(r)
        print(f"[{label}] control-hermes #{j} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    for j, (prompt, tools, expect) in enumerate(STREAM_PROMPTS[:2]):
        r = run_simple(model, dict(kind="stream", prompt=prompt, tools=tools, expect=expect), 100 + j, stream=True)
        results.append(r); print(f"[{label}] control-stream #{j} {'PASS' if r['pass'] else 'FAIL'}", flush=True)
    return results

def summarize(results):
    npass = sum(1 for r in results if r["pass"])
    by_kind = {}
    for r in results:
        k = r["kind"].split("-t0.7")[0]
        by_kind.setdefault(k, [0, 0]); by_kind[k][1] += 1; by_kind[k][0] += 1 if r["pass"] else 0
    derailed = [{"idx": r["idx"], "kind": r["kind"], "signatures": derailments(r)}
                for r in results if derailments(r)]
    return npass, {k: f"{v[0]}/{v[1]}" for k, v in sorted(by_kind.items())}, derailed

def main():
    parser = argparse.ArgumentParser(description="Steered tool-calling qualification gate (see header).")
    parser.add_argument("--base", default=BASE, help="OpenAI-compatible base URL")
    parser.add_argument("--model", default=WL, help="served model name to gate")
    parser.add_argument("--control", default=None, help="optional second model name for the 20-scenario control subset")
    parser.add_argument("--out", default=None, help="write the full JSON record here")
    args = parser.parse_args()
    globals()["BASE"] = args.base.rstrip("/")
    label = args.model
    results = run_suite(args.model, label)
    npass, by_kind, derailed = summarize(results)
    out = {"base": args.base, "model": args.model, "total": len(results), "passed": npass,
           "by_kind": by_kind, "derailed": derailed, "results": results}
    if args.control:
        control = run_control(args.control, args.control)
        cn, _bk, cder = summarize(control)
        out["control"] = {"model": args.control, "total": len(control), "passed": cn, "derailed": cder}
        print(f"[{args.control}] CONTROL {cn}/{len(control)} strict, {len(cder)} derailed", flush=True)
    if args.out:
        with open(args.out, "w") as f: json.dump(out, f, indent=1)
    errors = sum(1 for r in results if (r.get("detail") or "").startswith("ERROR"))
    print(f"[{label}] STRICT {npass}/{len(results)}  by_kind={by_kind}", flush=True)
    for d in derailed:
        print(f"[{label}] DERAILED #{d['idx']} {d['kind']}: {'; '.join(d['signatures'])}", flush=True)
    if errors == len(results):
        print(f"[{label}] GATE INCONCLUSIVE: every request errored; the endpoint could not be exercised", flush=True)
        return 2
    print(f"[{label}] GATE {'FAIL' if derailed else 'PASS'}: {len(derailed)} derailed scenario(s)", flush=True)
    return 1 if derailed else 0

if __name__ == "__main__":
    sys.exit(main())
