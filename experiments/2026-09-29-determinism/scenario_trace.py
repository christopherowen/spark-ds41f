#!/usr/bin/env python3
"""Serving scenarios for batch invariance, traced; every traced request becomes a run for analyze_trace4.py.

usage: scenario_trace.py BASE_URL OUT_DIR [--scenarios mixed,chunked,cache,identical] [--repeats R]
       (on dgx1, a trace arm serving with attn-exact4 and wide logs of at least 4160 rows)

Each scenario run resets the debug logs on every node, sends its requests on
a schedule (seconds after the start), waits for all, dumps the logs once and
copies every rank's logs to OUT_DIR/logs-<scenario>-<run>/; each traced
request gets OUT_DIR/<group>-<name>/ with target.json ("group" set, so
analyze_trace4.py compares it with the group's first request: the reference,
named "0-..." to sort first) and links to the shared logs.

- mixed: a short request decoding while a 3000-token prompt prefills beside
  it (its decode rows share sequence-parallel extend steps), against the short
  request alone; and the long request beside short ones, against it alone.
- chunked: a 9000-token prompt (three batch-budget chunks) alone, beside
  decoding requests (smaller first chunks), and beside another long prompt.
- cache: one prompt twice with prefix caching (the second reuses the first's
  KV and computes only its tail), and a request sharing a 2000-token prefix
  with a different question.
- identical: eight copies of one prompt arriving 0-350 ms apart.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import uuid

BASE, OUT = sys.argv[1].rstrip("/"), sys.argv[2]


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


SCENARIOS = arg("--scenarios", "mixed,chunked,cache,identical").split(",")
REPEATS = int(arg("--repeats", "1"))
NODES = ("dgx1", "dgx2", "dgx3")
LOGDIR = "/cache/kkref/moe-checksums"
with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    MODEL = json.load(response)["data"][0]["id"]
COLORS = ("red", "white", "striped", "black", "grey", "green", "blue", "amber")
PLACES = ("the northern cape", "a harbour mouth", "an offshore reef", "the river delta", "a granite island",
          "the old breakwater", "a sandbar", "the fjord entrance", "a chalk headland", "the south mole")
FUELS = ("whale oil", "colza oil", "paraffin", "acetylene", "coal gas", "electric arc")


def notes(count, seed=0):
    return " ".join(
        f"Note {i}: the {COLORS[(i + seed) % 8]} lighthouse at {PLACES[(i * 3 + seed) % 10]} was built in "
        f"{1650 + (i * 7 + seed * 13) % 300} and first lit with {FUELS[(i + 2 * seed) % 6]}."
        for i in range(count))


SHORT = "Return a JSON object describing three fictional planets with name, mass and moons."
LONG3K = "Here are lighthouse notes. " + notes(110, 1) + " Which fuel appears most often?"
LONG9K = "Here are lighthouse notes. " + notes(330, 2) + " Summarize when the lighthouses were built."
OTHER3K = "Here are lighthouse notes. " + notes(110, 3) + " Which place appears most often?"
PREFIX = "Here are lighthouse notes. " + notes(75, 4)
BACKGROUND = ["List ten prime numbers and explain why each is prime.",
              "Write a haiku about rain, then explain its imagery.",
              "Compare TCP and UDP for game networking."]


def on_nodes(command):
    for node in NODES:
        subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c '{command}'"],
                       check=True, capture_output=True)


def complete(prompt, tokens, tag=None, keep=None, salt=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    body["cache_salt"] = salt if salt is not None else uuid.uuid4().hex
    headers = {"Content-Type": "application/json"}
    if tag:
        body.update(logprobs=True, top_logprobs=1, return_tokens_as_token_ids=True)
        headers["X-Request-Id"] = tag
    request = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=1800) as response:
        data = json.load(response)
    if keep is not None:
        content = data["choices"][0]["logprobs"]["content"]
        keep.update(tokens=[c["token"] for c in content], logprobs=[c["logprob"] for c in content],
                    prompt_tokens=data["usage"]["prompt_tokens"], seconds=round(time.monotonic() - started, 3),
                    cached_tokens=(data["usage"].get("prompt_tokens_details") or {}).get("cached_tokens"),
                    tokens_ids=[int(c["token"].split(":", 1)[1]) for c in content
                                if c["token"].startswith("token_id:")])


def run(scenario, run_name, requests):
    """requests: [(start seconds, prompt, tokens, group or None, name, salt)]; returns the traced targets."""
    on_nodes(f"mkdir -p {LOGDIR}; rm -f {LOGDIR}/rank*; touch {LOGDIR}/reset")
    time.sleep(2.5)  # the watcher polls once a second
    targets, threads = {}, []
    t0 = time.monotonic()
    for start, prompt, tokens, group, name, salt in sorted(requests, key=lambda r: r[0]):
        delay = start - (time.monotonic() - t0)
        if delay > 0:
            time.sleep(delay)
        tag = f"trace-{scenario}-{run_name}-{name}" if group else None
        keep = targets.setdefault(name, {"group": group, "tag": tag}) if group else None
        thread = threading.Thread(target=complete, args=(prompt, tokens, tag, keep, salt))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    on_nodes(f"touch {LOGDIR}/dump")
    # The host schedule is written last (after the one-time plan inventory walk).
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        time.sleep(3)
        ready = [subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c "
                                 f"'ls {LOGDIR}/rank*-schedule-host-*.json 2>/dev/null | wc -l'"],
                                capture_output=True, text=True).stdout.strip() for node in NODES]
        if all(r not in ("", "0") for r in ready):
            break
    else:
        raise RuntimeError(f"{scenario} {run_name}: logs were not dumped on every node")
    logs = os.path.join(OUT, f"logs-{scenario}-{run_name}")
    for node in NODES:
        node_dir = os.path.join(logs, node)
        os.makedirs(node_dir, exist_ok=True)
        files = subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken sh -c 'cd {LOGDIR} && ls'"],
                               check=True, capture_output=True, text=True).stdout.split()
        for f in files:
            if f.startswith(("rank", "inventory", "plans")):
                with open(os.path.join(node_dir, f), "wb") as handle:
                    subprocess.run(["ssh", "-n", node, f"docker exec dsv41-karmic-kraken cat {LOGDIR}/{f}"],
                                   check=True, stdout=handle)
    for name, target in targets.items():
        run_dir = os.path.join(OUT, f"{target['group']}-{run_name}-{name}")
        os.makedirs(run_dir, exist_ok=True)
        for node in NODES:
            link = os.path.join(run_dir, node)
            if not os.path.lexists(link):
                os.symlink(os.path.join("..", f"logs-{scenario}-{run_name}", node), link)
        json.dump(target, open(os.path.join(run_dir, "target.json"), "w"))
        print(json.dumps({"scenario": scenario, "run": run_name, "request": name, "group": target["group"],
                          "prompt_tokens": target["prompt_tokens"], "cached": target.get("cached_tokens"),
                          "tokens": len(target["tokens"]), "seconds": target["seconds"],
                          "first_logprob": target["logprobs"][0]}), flush=True)


for repeat in range(REPEATS):
    r = f"r{repeat}"
    if "mixed" in SCENARIOS:
        run("mixed", f"0-solo-{r}", [(0, SHORT, 96, "mixed_short", "short", None)])
        run("mixed", f"0-solo-long-{r}", [(0, LONG3K, 16, "mixed_long", "long", None)])
        run("mixed", f"decode-in-prefill-{r}", [(0, SHORT, 96, "mixed_short", "short", None),
                                                 (0.6, LONG3K, 16, "mixed_long", "long", None)])
        run("mixed", f"prefill-beside-decodes-{r}", [(0, BACKGROUND[0], 160, None, "bg0", None),
                                                      (0, BACKGROUND[1], 160, None, "bg1", None),
                                                      (0.5, LONG3K, 16, "mixed_long", "long", None),
                                                      (0.5, SHORT, 96, "mixed_short", "short", None)])
    if "chunked" in SCENARIOS:
        run("chunked", f"0-solo-{r}", [(0, LONG9K, 16, "chunked", "long", None)])
        run("chunked", f"beside-decodes-{r}", [(0, BACKGROUND[0], 200, None, "bg0", None),
                                               (0, BACKGROUND[1], 200, None, "bg1", None),
                                               (0, BACKGROUND[2], 200, None, "bg2", None),
                                               (0.5, LONG9K, 16, "chunked", "long", None)])
        run("chunked", f"beside-long-{r}", [(0, OTHER3K, 16, None, "other", None),
                                            (0.05, LONG9K, 16, "chunked", "long", None)])
    if "cache" in SCENARIOS:
        salt = uuid.uuid4().hex
        run("cache", f"0-cold-{r}", [(0, LONG3K, 32, "cache_full", "first", salt)])
        run("cache", f"warm-{r}", [(0, LONG3K, 32, "cache_full", "again", salt)])
        salt2 = uuid.uuid4().hex
        run("cache", f"0-prefix-cold-{r}", [(0, PREFIX + " Which fuel appears most often?", 32, "cache_prefix",
                                             "a", salt2)])
        run("cache", f"prefix-warm-{r}", [(0, PREFIX + " Which fuel appears most often?", 32, "cache_prefix",
                                           "b", salt2),
                                          (0, PREFIX + " Which place appears most often?", 32, None, "c", salt2)])
    if "identical" in SCENARIOS:
        run("identical", f"0-solo-{r}", [(0, SHORT, 32, "identical", "solo", None)])
        run("identical", f"staggered-{r}", [(0.05 * i, SHORT, 32, "identical", f"s{i}", None) for i in range(8)])
