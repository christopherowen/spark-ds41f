import concurrent.futures
import json
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROMPT = open("/tmp/m8_prompt.txt").read()


def one(i):
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 256,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    return {
        "i": i,
        "latency": time.time() - t0,
        "completion_tokens": d["usage"]["completion_tokens"],
        "prompt_tokens": d["usage"]["prompt_tokens"],
        "chars": len(msg.get("content") or "") + len(msg.get("reasoning_content") or ""),
        "finish": d["choices"][0].get("finish_reason"),
    }


t0 = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    rows = list(ex.map(one, range(8)))
wall = time.time() - t0
comp = sum(r["completion_tokens"] for r in rows)
prompt = sum(r["prompt_tokens"] for r in rows)
print("rows", json.dumps(rows))
print(
    "wall=%.2fs completion_tokens=%d prompt_tokens=%d output_tok_s=%.2f total_tok_s=%.2f"
    % (wall, comp, prompt, comp / wall, (comp + prompt) / wall)
)
