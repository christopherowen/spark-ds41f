#!/usr/bin/env python3
"""Seeded top-p and top-k outputs as digests, then any FlashInfer code on the nodes.

usage: flashinfer_probe.py BASE_URL [--tokens T] [--seed S] [--nodes N,...] [--container C]

Three scenarios at temperature 1, thinking off, each request with its own fixed seed and
fresh cache salt, eight prompts one at a time and then all eight at once: top-p 0.9,
top-k 40, and both (top-k 20, top-p 0.8), the filters the V1 sampler would hand to
FlashInfer. Each scenario's line has "identical" (the eight at once equal to one at a
time, tokens and logprobs) and the one-at-a-time digest, which two builds whose sampling
is the same bit for bit share. A scenario vLLM refuses is recorded with its error.

Then, per node, in the serving container: the FlashInfer files any process has mapped
(the precompiled kernel packages' libraries, or a module FlashInfer compiled), and the
FlashInfer modules compiled at run time under the container's caches. The summary has
"flashinfer_used": any mapped or compiled FlashInfer module on any node.
Exits 0 either way; the run records.
"""
import argparse
import hashlib
import json
import shlex
import subprocess
import threading
import urllib.error
import urllib.request
import uuid

PROMPTS = [
    "Explain how a hash map handles collisions, with an example in Python.",
    "Describe a walk through a spice market in Marrakech in the morning.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Compare TCP and UDP for real-time multiplayer games and when to use each.",
    "Give a detailed recipe for lentil soup with timings and substitutions.",
    "Summarize the causes and consequences of the printing press in Europe.",
    "Explain the difference between supervised and unsupervised learning to a student.",
    "Plan a three-day hiking trip in the Alps with packing advice.",
]

SCENARIOS = {
    "top-p": {"temperature": 1.0, "top_p": 0.9},
    "top-k": {"temperature": 1.0, "top_k": 40},
    "top-k-top-p": {"temperature": 1.0, "top_k": 20, "top_p": 0.8},
}

# Mapped files whose path names FlashInfer: its kernel packages' libraries, its JIT modules.
MAPPED = "cat /proc/[0-9]*/maps 2>/dev/null | awk '{print $6}' | grep -i flashinfer | grep '\\.so' | sort -u"
# Modules FlashInfer compiles at run time land in its workspace (FLASHINFER_WORKSPACE_BASE,
# else ~/.cache/flashinfer) or under the mounted caches; older images' modules stay in
# /cache, so only those written since the container started count.
COMPILED = ("find \"${FLASHINFER_WORKSPACE_BASE:-$HOME/.cache}\" /root/.cache /cache -ipath '*flashinfer*'"
            " -name '*.so' -newermt STARTED 2>/dev/null | sort -u")


class Client:
    def __init__(self, base: str):
        self.base = base
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def complete(self, prompt: str, tokens: int, seed: int, controls: dict) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "seed": seed, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "logprobs": True,
                "chat_template_kwargs": {"thinking": False}, **controls}
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=900) as response:
            choice = json.load(response)["choices"][0]
        return {key: choice.get(key) for key in ("message", "logprobs", "finish_reason")}


def together(client: Client, tokens: int, seeds: list[int], controls: dict) -> list:
    """The eight at once; a request's error is raised here, as a request alone raises."""
    results: list = [None] * len(PROMPTS)
    errors: list = []

    def run(index):
        try:
            results[index] = client.complete(PROMPTS[index], tokens, seeds[index], controls)
        except Exception as error:  # noqa: BLE001 - re-raised in the caller's thread
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(PROMPTS))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    return results


def digest(results: list) -> str:
    return hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()


def in_container(node: str, container: str, command: str) -> list[str]:
    process = subprocess.run(["ssh", node, "docker", "exec", container, "sh", "-c", shlex.quote(command)],
                             capture_output=True, text=True, timeout=300)
    if process.returncode not in (0, 1):  # grep's 1 is "none"
        raise RuntimeError(f"{node}: {process.stderr.strip() or process.returncode}")
    return process.stdout.split()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--nodes", default="dgx1,dgx2,dgx3,dgx4")
    parser.add_argument("--container", default="spark-ds41f")
    args = parser.parse_args()
    client = Client(args.base_url.rstrip("/"))
    seeds = [args.seed + i for i in range(len(PROMPTS))]
    digests, failed = {}, []
    for name, controls in SCENARIOS.items():
        try:
            client.complete(PROMPTS[0], 1, seeds[0], controls)
            alone = [client.complete(p, args.tokens, s, controls) for p, s in zip(PROMPTS, seeds)]
            c8 = together(client, args.tokens, seeds, controls)
        except urllib.error.HTTPError as error:
            failed.append(name)
            print(json.dumps({"scenario": name, "controls": controls, "http": error.code,
                              "error": error.read().decode(errors="replace")[:400]}), flush=True)
            continue
        digests[name] = digest(alone)
        print(json.dumps({"scenario": name, "controls": controls, "identical": c8 == alone,
                          "digest": digests[name]}), flush=True)
    nodes = {}
    for node in args.nodes.split(","):
        started = subprocess.run(["ssh", node, "docker", "inspect", "--format", "'{{.State.StartedAt}}'",
                                  args.container], capture_output=True, text=True, check=True).stdout.strip()
        nodes[node] = {"started": started, "mapped": in_container(node, args.container, MAPPED),
                       "compiled": in_container(node, args.container,
                                                COMPILED.replace("STARTED", shlex.quote(started)))}
        print(json.dumps({"node": node, **nodes[node]}), flush=True)
    print(json.dumps({"summary": True, "digests": digests, "failed": failed,
                      "flashinfer_used": any(n["mapped"] or n["compiled"] for n in nodes.values())}))


if __name__ == "__main__":
    main()
