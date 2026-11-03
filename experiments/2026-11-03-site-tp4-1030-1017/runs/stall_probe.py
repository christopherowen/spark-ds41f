"""What a short chat request feels while a long prompt prefills.

Usage (on the head): python3 stall_probe.py <base> <label> [needle_words]
Streams one short temperature-0 request and records TTFT and the gaps between streamed chunks:
first alone, then about 10 s after a salted needle (175,000 words, about 545K tokens) has started.
Prints one JSON line per run. Read-only against the engine. needle_salted.py is
../../2026-10-17-site-tp4-4k-1017/runs/needle_salted.py, expected next to this script.
"""
import json, os, subprocess, sys, time, urllib.request

BASE, LABEL = sys.argv[1], sys.argv[2]
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 175000
NEEDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "needle_salted.py")
BODY = {"model": "deepseek-v4.1-flash", "temperature": 0, "max_tokens": 600, "reasoning_effort": "none",
        "stream": True, "messages": [{"role": "user", "content": "Count from 1 to 150, numbers only, separated by spaces."}]}


def stream_once():
    req = urllib.request.Request(f"{BASE}/chat/completions", json.dumps(BODY).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.monotonic(); stamps = []
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ch = json.loads(line[6:])["choices"]
            if ch and (ch[0].get("delta") or {}).get("content"):
                stamps.append(time.monotonic() - t0)
    gaps = sorted(b - a for a, b in zip(stamps, stamps[1:]))
    pick = lambda q: round(gaps[min(len(gaps) - 1, int(q * len(gaps)))] * 1000, 1) if gaps else None
    return {"ttft_s": round(stamps[0], 3) if stamps else None, "chunks": len(stamps),
            "total_s": round(stamps[-1], 2) if stamps else None,
            "gap_ms_p50": pick(0.5), "gap_ms_p99": pick(0.99), "gap_ms_max": round(gaps[-1] * 1000, 1) if gaps else None,
            "gaps_over_500ms": sum(g > 0.5 for g in gaps)}


print(json.dumps({"label": LABEL, "run": "alone", **stream_once()}), flush=True)
needle = subprocess.Popen([sys.executable, NEEDLE, BASE, str(NW)],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(10)
print(json.dumps({"label": LABEL, "run": "beside_needle_10s", **stream_once()}), flush=True)
out = needle.communicate(timeout=1800)[0]
try:
    j = json.loads(out[out.index("{"):])
    print(json.dumps({"label": LABEL, "run": "needle", "prompt_tokens": j.get("prompt_tokens"),
                      "pass": j.get("pass"), "prefill_tok_s": j.get("prefill_tok_s")}), flush=True)
except ValueError:
    print(json.dumps({"label": LABEL, "run": "needle", "raw": out[-300:]}), flush=True)
