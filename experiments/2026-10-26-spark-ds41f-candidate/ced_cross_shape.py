#!/usr/bin/env python3
"""CED's prompt-end anchor across chunkings: one token-exact prompt, every shape and budget.

usage: ced_cross_shape.py URL [--lengths 8208,16352] [--tokens 128] [--repeats 2]
       ced_cross_shape.py compare FILE FILE...

CED replays a prompt's last 128 rows wherever its chunks end (vLLM 0526). A prompt whose final
chunk has 1-128 rows used to replay rows other than a single chunk's, so its temperature-0
output depended on the prefill budget. Each length here is chosen to put the final chunk on
either side of that line (thresholds: TP4 16,288, TP4 at 12,288 12,192, TP3 8,144, TP3 at
6,144 6,096):
- 8,208 = 8,144 + 64: one chunk at TP4, a 64-row final at TP3-8192, 2,112 rows at TP3-6144;
- 16,352 = 16,288 + 64: a 64-row final at TP4 and TP3-8192, 4,160 rows at TP4-12288,
  4,160 at TP3-6144.

The prompt is a fixed synthetic document, tokenized by the server (/tokenize) and cut to each
length, then sent as token ids through /v1/completions: temperature 0, ignore_eos, the chosen
token's log probability, and a fresh cache salt so every request is cold (the reported
cached_tokens must be 0). Each request runs --repeats times. One JSON line per request, then a
"summary" line with each length's digests (prompt ids, output ids, output ids with log
probabilities) and whether the repeats agree.

compare reads the summary lines of several runs and prints, for each length, the digests
side by side; it exits 1 unless every run gave the same prompt, and the same output ids and
log probabilities. Compare runs at one tensor-parallel size: TP3 and TP4 reduce in different
orders, so their log probabilities differ whatever CED does (as the determinism gate's hashes
do between the ring and the triangle). Stdlib only.
"""
import argparse
import hashlib
import json
import sys
import time
import urllib.request

WORDS = ("amber basalt cedar delta ember fjord granite harbor iris juniper kelp lagoon meadow "
         "nickel orchid pewter quartz river sierra tundra umber violet willow xenon yarrow zinc").split()


def document(lines: int = 4000) -> str:
    """A fixed synthetic ledger: numbered entries whose words follow a linear congruence."""
    state, out = 20261026, []
    for i in range(lines):
        words = []
        for _ in range(9):
            state = (state * 1103515245 + 12345) % 2**31
            words.append(WORDS[state % len(WORDS)])
        out.append(f"Entry {i:05d}: {' '.join(words)}; checksum {state % 9973:04d}.")
    return "\n".join(out) + "\n"


def post(base: str, path: str, body: dict) -> dict:
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read())


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()[:16]


def probe(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
        model = json.loads(response.read())["data"][0]["id"]
    ids = post(base, "/tokenize", {"model": model, "prompt": document(),
                                   "add_special_tokens": False})["tokens"]
    lengths = [int(n) for n in args.lengths.split(",")]
    if max(lengths) > len(ids):
        raise SystemExit(f"the document has {len(ids)} tokens, fewer than {max(lengths)}")
    nonce, summary, failed = time.time_ns(), {}, False
    for n in lengths:
        prompt = ids[:n]
        runs = []
        for repeat in range(args.repeats):
            started = time.monotonic()
            data = post(base, "/v1/completions", {
                "model": model, "prompt": prompt, "max_tokens": args.tokens, "temperature": 0,
                "ignore_eos": True, "logprobs": 1, "return_token_ids": True,
                "cache_salt": f"ced-cross-shape-{nonce}-{n}-{repeat}"})
            choice = data["choices"][0]
            tokens = choice["token_ids"]
            logprobs = [repr(v) for v in choice["logprobs"]["token_logprobs"]]
            cached = ((data.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
            run = {"length": n, "repeat": repeat, "prompt": digest(prompt), "tokens": digest(tokens),
                   "logprobs": digest([tokens, logprobs]), "cached_tokens": cached,
                   "prompt_tokens": data["usage"]["prompt_tokens"],
                   "seconds": round(time.monotonic() - started, 3), "token_ids": tokens,
                   "token_logprobs": logprobs}
            print(json.dumps(run), flush=True)
            runs.append(run)
            failed |= cached != 0 or run["prompt_tokens"] != n
        keys = ("prompt", "tokens", "logprobs")
        summary[str(n)] = {**{k: runs[0][k] for k in keys},
                           "repeats_identical": all(r[k] == runs[0][k] for r in runs for k in keys),
                           "cold": all(r["cached_tokens"] == 0 for r in runs)}
        failed |= not summary[str(n)]["repeats_identical"]
    print(json.dumps({"summary": "ced cross-shape", "lengths": summary, "passed": not failed}))
    return 1 if failed else 0


def compare(files: list[str]) -> int:
    runs = {}
    for path in files:
        lines = [json.loads(line) for line in open(path) if line.strip().startswith("{")]
        runs[path] = next(line for line in reversed(lines) if line.get("summary") == "ced cross-shape")
    lengths = sorted({n for run in runs.values() for n in run["lengths"]}, key=int)
    same = True
    for n in lengths:
        rows = {path: run["lengths"].get(n) for path, run in runs.items()}
        for key in ("prompt", "tokens", "logprobs"):
            values = {path: (row or {}).get(key) for path, row in rows.items()}
            equal = len(set(values.values())) == 1 and None not in values.values()
            same &= equal
            print(f"{n:>6} {key:<8} {'equal' if equal else 'DIFFER'}  "
                  + "  ".join(f"{value}" for value in values.values()))
    print(f"{'identical' if same else 'not identical'}: {', '.join(runs)}")
    return 0 if same else 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        return compare(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--lengths", default="8208,16352")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=2)
    return probe(parser.parse_args())


if __name__ == "__main__":
    sys.exit(main())
