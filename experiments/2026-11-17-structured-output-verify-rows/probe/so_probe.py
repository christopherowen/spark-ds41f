#!/usr/bin/env python3
"""Structured-output burst probe (spark-ds41f#1194 guard).

Sends N simultaneous chat requests with response_format json_object / json_schema whose answers copy long spans
from the prompt (so DSpark copy drafts verify deep), then checks: every request returned 200 with parseable JSON,
and the server is still alive afterwards. usage: so_probe.py ROOT MODEL [N=32] [WAVES=2]
"""
import json, sys, time, random, urllib.request, concurrent.futures as cf

root, model = sys.argv[1].rstrip('/'), sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 32
WAVES = int(sys.argv[4]) if len(sys.argv) > 4 else 2
random.seed(7)
words = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa".split()


def make(i):
    recs = [{"id": f"R{i}-{k:03d}", "name": " ".join(random.choice(words) for _ in range(4)),
             "qty": random.randint(1, 999), "tag": random.choice(words)} for k in range(40)]
    doc = json.dumps(recs, indent=1)
    msg = (f"Here is a JSON array of records:\n{doc}\n\nReturn a JSON object {{\"records\": [...]}} containing every "
           f"record above, unchanged and in the same order. Output only JSON.")
    body = {"model": model, "messages": [{"role": "user", "content": msg}], "max_tokens": 4096, "temperature": 0,
            "chat_template_kwargs": {"thinking": False}}
    if i % 2:
        body["response_format"] = {"type": "json_object"}
    else:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "recs", "schema": {
            "type": "object", "properties": {"records": {"type": "array", "items": {"type": "object", "properties": {
                "id": {"type": "string"}, "name": {"type": "string"}, "qty": {"type": "integer"},
                "tag": {"type": "string"}}, "required": ["id", "name", "qty", "tag"]}}}, "required": ["records"]}}}
    return body, len(recs)


def call(i):
    body, n = make(i)
    t0 = time.time()
    try:
        r = urllib.request.urlopen(urllib.request.Request(root + "/v1/chat/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"}), timeout=900)
        d = json.load(r)
        txt = d["choices"][0]["message"]["content"]
        try:
            got = len(json.loads(txt).get("records", []))
            ok_json = True
        except Exception:
            got, ok_json = 0, False
        return {"i": i, "status": 200, "json": ok_json, "records": got, "want": n, "s": round(time.time() - t0, 1),
                "finish": d["choices"][0].get("finish_reason"), "toks": d.get("usage", {}).get("completion_tokens")}
    except urllib.error.HTTPError as e:
        return {"i": i, "status": e.code, "err": e.read()[:300].decode("utf8", "replace"), "s": round(time.time() - t0, 1)}
    except Exception as e:
        return {"i": i, "status": -1, "err": repr(e)[:300], "s": round(time.time() - t0, 1)}


out = {"root": root, "model": model, "n": N, "waves": WAVES, "results": []}
for w in range(WAVES):
    with cf.ThreadPoolExecutor(N) as ex:
        res = list(ex.map(call, [w * N + i for i in range(N)]))
    out["results"] += res
try:
    alive = urllib.request.urlopen(root + "/v1/models", timeout=20).status == 200
except Exception:
    alive = False
r = out["results"]
out["summary"] = {"ok_200": sum(x["status"] == 200 for x in r), "valid_json": sum(bool(x.get("json")) for x in r),
                  "exact_count": sum(x.get("records") == x.get("want") for x in r), "total": len(r), "alive_after": alive}
out["pass"] = out["summary"]["ok_200"] == len(r) and alive
print(json.dumps(out["summary"]), "PASS" if out["pass"] else "FAIL")
json.dump(out, open(f"so_probe-{int(time.time())}.json", "w"), indent=1)
