"""Cold needle with a random salt at the very start (no prefix-cache hit possible). Usage: needle_salted.py <base> <nwords>"""
import json, random, sys, time, urllib.request

BASE, NW = sys.argv[1], int(sys.argv[2])
rng = random.Random()
syl = ["ka", "ro", "mi", "tsu", "len", "vor", "da", "pi", "quo", "ser", "nal", "be", "ux", "tor", "fe", "gil"]
words = ["".join(rng.choice(syl) for _ in range(rng.randint(2, 4))) for _ in range(30000)]
fill = [rng.choice(words) for _ in range(NW)]
fill.insert(NW // 2, "The secret passphrase is MAGENTA-OTTER-4412.")
prompt = f"[salt {rng.random()}] " + " ".join(fill) + "\n\nWhat is the secret passphrase? Answer with the passphrase only."
body = {"model": "deepseek-v4.1-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 30, "temperature": 0,
        "chat_template_kwargs": {"thinking": False}}
t0 = time.time()
d = json.load(urllib.request.urlopen(urllib.request.Request(BASE + "/chat/completions", json.dumps(body).encode(),
                                                            {"Content-Type": "application/json"}), timeout=3600))
dt = time.time() - t0
pt = d["usage"]["prompt_tokens"]; ans = d["choices"][0]["message"]["content"]
print(json.dumps({"prompt_tokens": pt, "total_s": round(dt, 1), "prefill_tok_s": round(pt / dt, 1), "answer": ans,
                  "pass": "MAGENTA-OTTER-4412" in (ans or "")}))
