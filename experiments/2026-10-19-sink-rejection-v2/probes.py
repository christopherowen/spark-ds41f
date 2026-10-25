"""The collapse probes of the sink-rejection screen.

The thinking-collapse experiment's 16 prompts, thinking on at effort high, temperature 0:
each answer with the response's usage.sink_rejection, then the server's sink counters
(/metrics) and each rank's last "Sink detector, rank" log line. `Say hi.` and `Thanks!`
collapse on r6x (`How can\\n can`, `there\\n there's`); the other 14 answer normally.

Usage: python3 probes.py BASE_URL [--max-tokens N] [--nodes dgx1,dgx2,dgx3] [--port P] [--container NAME]
Prints one JSON line per prompt, then {"summary": ...} (lab.py keeps stdout). The rank lines
come from the serving container's log: name the arm's own (spark-ds41f since the rename), or
they are the stopped production container's.
"""

import argparse
import json
import subprocess
import urllib.request

MODEL = "deepseek-v4.1-flash"
CONTAINER = "dsv41-karmic-kraken"
PROMPTS = ["Say hi.", "Hello", "Thanks!", "Tell me a joke.", "What's 2+2?", "Who are you?",
           "Translate 'good morning' to French.", "Write one sentence about the sea.", "Good night!",
           "Thank you so much, that worked.", "Can you help me?", "Hi there, how are you?",
           "ok", "Great, thanks for the help!", "What day comes after Monday?", "Name a color."]
COLLAPSE_MARKERS = ("How can\n can", "there\n there's", "<ds_safety>")


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def sink_counters(base: str) -> dict[str, float]:
    with urllib.request.urlopen(base + "/metrics", timeout=60) as response:
        lines = response.read().decode().splitlines()
    return {line.rsplit(" ", 1)[0]: float(line.rsplit(" ", 1)[1])
            for line in lines if line.startswith("vllm:sink_")}


def rank_lines(nodes: list[str], container: str) -> dict[str, str | None]:
    lines = {}
    for node in nodes:
        logs = subprocess.run(["ssh", node, "docker", "logs", container],
                              capture_output=True, text=True)
        found = [line for line in (logs.stdout + logs.stderr).splitlines()
                 if "Sink detector, rank" in line]
        lines[node] = found[-1].split("Sink detector, ", 1)[1] if found else None
    return lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_url")
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--nodes", default="dgx1,dgx2,dgx3")
    parser.add_argument("--port", type=int, help="the arm's port, if not BASE_URL's (lab passes :8000)")
    parser.add_argument("--container", default=CONTAINER, help="the serving container whose log holds the rank lines")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.port:
        base = base.rsplit(":", 1)[0] + f":{args.port}"

    rows = []
    for prompt in PROMPTS:
        ids = post(base, "/tokenize", {
            "model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "add_generation_prompt": True,
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}})["tokens"]
        response = post(base, "/v1/completions", {
            "model": MODEL, "prompt": ids, "max_tokens": args.max_tokens, "temperature": 0,
            "skip_special_tokens": False})
        choice = response["choices"][0]
        row = {"prompt": prompt, "text": choice["text"], "finish_reason": choice["finish_reason"],
               "collapsed": any(marker in choice["text"] for marker in COLLAPSE_MARKERS),
               "sink_rejection": (response.get("usage") or {}).get("sink_rejection")}
        print(json.dumps(row), flush=True)
        rows.append(row)

    summary = {
        "prompts": len(rows),
        "collapsed": [row["prompt"] for row in rows if row["collapsed"]],
        "flagged": [row["prompt"] for row in rows
                    if (row["sink_rejection"] or {}).get("flagged_rows")],
        "counters": sink_counters(base),
        "ranks": rank_lines(args.nodes.split(","), args.container),
    }
    print(json.dumps({"summary": summary}), flush=True)


if __name__ == "__main__":
    main()
