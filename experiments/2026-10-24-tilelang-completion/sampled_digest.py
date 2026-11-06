#!/usr/bin/env python3
"""Seeded temperature-1 outputs, as digests two builds can compare bit for bit.

usage: sampled_digest.py BASE_URL [--tokens T] [--seed S] [--record-dir PRIVATE_DIR]

token_determinism.py's eight prompts, thinking off, temperature 1, each request with its
own fixed seed (``--seed`` plus its index), a fixed number of tokens, fresh cache salts:

- alone: one at a time (the reference);
- c8: all eight at once (an observation, not a universal batch-invariance
  requirement: block verification also keys draws by local speculative step);
- again: one at a time once more.

Speculative decoding samples its drafts, verifies them and resamples rejected positions
on this path, so a build whose sampling kernels changed is compared on all of it. Each
request asks for the chosen tokens' logprobs. One JSON line per scenario, then a summary
with "identical" (every scenario equal to the reference, tokens and logprobs) and the
reference's SHA-256 digest. Equality requires the same RNG inputs and execution
contract, not merely the same request seed. Exits 0 for observed differences;
consumers must enforce the requirements they declare. --record-dir retains
private request/response identities, controls, token arrays and logprobs. Native
logit positions and local speculative steps are not exposed by this HTTP API;
the recorder marks them unavailable instead of inferring them from output order.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.parse
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


class Client:
    def __init__(self, base: str, record_dir: Path | None = None):
        self.base = base
        self.record_dir = record_dir
        if record_dir is not None:
            record_dir.mkdir(parents=True, exist_ok=False)
        with urllib.request.urlopen(base + "/v1/models", timeout=60) as response:
            self.model = json.load(response)["data"][0]["id"]

    def complete(self, prompt: str, tokens: int, seed: int, *, scenario: str, case: int) -> dict:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": tokens, "temperature": 1.0, "seed": seed, "ignore_eos": True,
                "cache_salt": uuid.uuid4().hex, "logprobs": True,
                "chat_template_kwargs": {"thinking": False}}
        headers = {"Content-Type": "application/json"}
        record, path = None, None
        if self.record_dir is not None:
            headers["X-Request-Id"] = "sampled-" + uuid.uuid4().hex
            body["return_token_ids"] = True
            parsed = urllib.parse.urlsplit(self.base)
            record = {"scenario": scenario, "case": case,
                      "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "endpoint": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}{parsed.path}/v1/chat/completions",
                      "client_request_id": headers["X-Request-Id"], "request": body,
                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                      "rng_identity": {"request_seed": seed, "native_logit_positions": None,
                                       "local_speculative_steps": None,
                                       "availability": "HTTP exposes neither native logit positions nor speculative-step grouping; output indexes are not substitutes."}}
            path = self.record_dir / f"{scenario}-{case}.json"
            with path.open("x") as handle:
                json.dump(record, handle, allow_nan=False)
        request = urllib.request.Request(self.base + "/v1/chat/completions",
                                         data=json.dumps(body).encode(),
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                payload = json.load(response)
                if record is not None:
                    record.update(http_status=response.status, response=payload,
                                  response_headers={key: response.headers[key] for key in
                                                    ("x-request-id", "x-correlation-id") if key in response.headers})
            choice = payload["choices"][0]
            content = choice["logprobs"]["content"]
            result = {"tokens": [c["token"] for c in content],
                      "logprobs": [c["logprob"] for c in content]}
            if record is not None:
                record.update(result=result, response_id=payload.get("id"),
                              native_token_ids=choice.get("token_ids"),
                              output_indexes=list(range(len(content))),
                              prompt_tokens=(payload.get("usage") or {}).get("prompt_tokens"))
            return result
        except Exception as error:
            if record is not None:
                record["error"] = f"{type(error).__name__}: {error}"
                if isinstance(error, urllib.error.HTTPError):
                    record.update(http_status=error.code, error_body=error.read().decode(errors="replace"))
            raise
        finally:
            if record is not None:
                record["ended"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                temporary.replace(path)


def together(client: Client, prompts: list[str], tokens: int, seeds: list[int]) -> list[dict]:
    results: list[dict] = [{} for _ in prompts]

    def run(index):
        results[index].update(client.complete(prompts[index], tokens, seeds[index], scenario="c8", case=index))

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(prompts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if any(not r for r in results):
        raise SystemExit("a request returned nothing")
    return results


def digest(results: list[dict]) -> str:
    return hashlib.sha256(json.dumps([[r["tokens"], r["logprobs"]] for r in results])
                          .encode()).hexdigest()


def compare(results: list[dict], reference: list[dict]) -> dict:
    """Keep token and logprob differences separate; request identity is not output."""
    def first(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                    None if len(a) == len(b) else min(len(a), len(b)))
    token_differing, logprob_differing, details = [], [], []
    for i, (value, ref) in enumerate(zip(results, reference)):
        token = first(value["tokens"], ref["tokens"])
        logprob = first(value["logprobs"], ref["logprobs"])
        if token is not None:
            token_differing.append(i)
        if logprob is not None:
            logprob_differing.append(i)
        details.append({"case": i, "first_token_diff": token, "first_logprob_diff": logprob})
    return {"differing": sorted(set(token_differing + logprob_differing)),
            "token_differing": token_differing, "logprob_differing": logprob_differing,
            "cases": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base_url")
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--record-dir", type=Path, help="new private directory for per-request evidence; never overwritten")
    args = parser.parse_args()
    if args.record_dir is not None:
        args.record_dir = args.record_dir.resolve()
    client = Client(args.base_url.rstrip("/"), args.record_dir)
    seeds = [args.seed + i for i in range(len(PROMPTS))]
    reference = [client.complete(p, args.tokens, s, scenario="alone", case=i)
                 for i, (p, s) in enumerate(zip(PROMPTS, seeds))]
    scenarios = {
        "c8": together(client, PROMPTS, args.tokens, seeds),
        "again": [client.complete(p, args.tokens, s, scenario="again", case=i)
                  for i, (p, s) in enumerate(zip(PROMPTS, seeds))],
    }
    identical, report = True, {}
    for name, results in scenarios.items():
        report[name] = compare(results, reference)
        identical &= not report[name]["differing"]
        print(json.dumps({"scenario": name, **report[name],
                          "sha256": digest(results)}), flush=True)
    request_evidence = None
    if args.record_dir is not None:
        manifest = {"collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "seed": args.seed, "tokens": args.tokens, "cases": len(PROMPTS), "requests": []}
        for scenario in ("alone", "c8", "again"):
            for case in range(len(PROMPTS)):
                path = args.record_dir / f"{scenario}-{case}.json"
                data = path.read_bytes()
                recorded = json.loads(data)
                if "error" in recorded or not recorded.get("ended") or "result" not in recorded:
                    raise ValueError(f"incomplete request recording: {path}")
                manifest["requests"].append({"scenario": scenario, "case": case, "file": path.name,
                                             "sha256": hashlib.sha256(data).hexdigest()})
        data = (json.dumps(manifest, indent=2) + "\n").encode()
        with (args.record_dir / "manifest.json").open("xb") as handle:
            handle.write(data)
        request_evidence = {"directory": str(args.record_dir), "requests": len(manifest["requests"]),
                            "manifest_sha256": hashlib.sha256(data).hexdigest(),
                            "native_logit_positions_available": False, "local_speculative_steps_available": False}
    print(json.dumps({"summary": "seeded temperature-1 digest", "identical": identical,
                      "tokens": args.tokens, "seed": args.seed, "scenarios": report,
                      "reference_sha256": digest(reference), "request_evidence": request_evidence}), flush=True)


if __name__ == "__main__":
    main()
