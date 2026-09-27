#!/usr/bin/env python3
"""Record one profiled cold prefill of ~16K tokens of real text.

usage: capture_prefill.py BASE_URL   (the sp-profile config must be serving)
"""
import importlib.machinery
import importlib.util
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
loader = importlib.machinery.SourceFileLoader("spark3", str(ROOT / "bin/spark3"))
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
BASE = sys.argv[1].rstrip("/")


def post(path, body=None):
    request = urllib.request.Request(BASE + path, data=json.dumps(body or {}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=1800) as response:
        return response.read()


with urllib.request.urlopen(BASE + "/v1/models", timeout=60) as response:
    model = json.load(response)["data"][0]["id"]
warm = {"model": model, "prompt": spark3.source_text(6000, 30) + "\n\nok", "max_tokens": 1,
        "temperature": 0, "cache_salt": "warm"}
post("/v1/completions", warm)
prefill = {"model": model, "prompt": spark3.source_text(16000, 11) + "\n\nReply with the word ok.",
           "max_tokens": 1, "temperature": 0, "cache_salt": "profile-prefill"}
post("/start_profile")
started = time.perf_counter()
usage = json.loads(post("/v1/completions", prefill))["usage"]
print(f"prefill request {time.perf_counter() - started:.2f} s, {usage['prompt_tokens']} tokens", flush=True)
post("/stop_profile")
