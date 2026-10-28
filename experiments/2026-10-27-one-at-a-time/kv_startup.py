#!/usr/bin/env python3
"""The serving container's startup KV pool: vLLM's "GPU KV cache size" line, as JSON.

usage: kv_startup.py URL [--container NAME]

Run on the head node by a lab measure job (URL is the head's, unused beyond the job's
convention). Memory an arm frees shows up here as pool tokens.
"""
import argparse
import json
import re
import subprocess
import sys

LINE = re.compile(r"GPU KV cache size: ([\d,]+) tokens, Maximum concurrency for ([\d,]+) tokens "
                  r"per request: ([\d.]+)x")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--container", default="spark-ds41f")
    args = parser.parse_args()
    logs = subprocess.run(["docker", "logs", args.container], capture_output=True, text=True)
    found = LINE.findall(logs.stdout + logs.stderr)
    if not found:
        print(json.dumps({"summary": "kv startup", "error": "no GPU KV cache size line"}))
        return 1
    tokens, limit, concurrency = found[-1]
    print(json.dumps({"summary": "kv startup", "kv_tokens": int(tokens.replace(",", "")),
                      "request_limit": int(limit.replace(",", "")), "concurrency": float(concurrency)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
