#!/bin/bash
# usage: cublas_probe.sh   (on dgx1, deployment checkout at this experiment's commit)
# Boots the cublas-log arm, marks the boot/serving boundary, sends one short
# decode request and one ~2K-token prefill, then copies every node's cuBLAS
# logs to results/private/boot/cublas-log. The arm stays up.
set -u
cd ~/projects/spark3-vllm-ds41f
E=experiments/2026-09-28-boot-time
out=results/private/boot/cublas-log
# cache/kkref belongs to root (the containers create it); cuBLAS does not
# create a missing log directory, so make it writable up front.
for n in dgx1 dgx2 dgx3; do
  ssh -n $n "d=~/projects/spark3-vllm-ds41f/cache/kkref/cublas-log; sudo -n rm -rf \$d && sudo -n mkdir -p \$d && sudo -n chmod 777 \$d"
done
$E/boot.sh cublas-log cublas-log
date -u +%FT%T.%3NZ > "$out/serving_t0"
python3 - <<'PY'
import importlib.machinery, importlib.util, json, urllib.request
loader = importlib.machinery.SourceFileLoader("spark3", "bin/spark")
spec = importlib.util.spec_from_loader("spark3", loader)
spark3 = importlib.util.module_from_spec(spec)
loader.exec_module(spark3)
base = "http://10.0.1.71:8000"
model = json.load(urllib.request.urlopen(base + "/v1/models"))["data"][0]["id"]
for prompt, tokens in (("Write a short poem about the sea.", 64), (spark3.source_text(2000, 5) + "\n\nok", 1)):
    body = {"model": model, "prompt": prompt, "max_tokens": tokens, "min_tokens": tokens,
            "ignore_eos": True, "temperature": 0}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    print(json.load(urllib.request.urlopen(req, timeout=600))["usage"])
PY
for n in dgx1 dgx2 dgx3; do
  mkdir -p "$out/$n"
  ssh -n $n "sudo -n tar -C ~/projects/spark3-vllm-ds41f/cache/kkref -cf - cublas-log" | tar -C "$out/$n" -xf -
done
du -sh "$out"/*/cublas-log
