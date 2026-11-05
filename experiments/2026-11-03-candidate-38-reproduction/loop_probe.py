#!/usr/bin/env python3
"""Replay ten bounded loop probes under a lab window, or validate them offline.

The lab owns the serving lifecycle, hold, image checks and node telemetry.
This client sends one HTTP request at a time and never executes returned tools.
Example: python3 loop_probe.py --check
         python3 loop_probe.py http://head:8000 --run RUN --label BOOT
"""

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_REVISION = "f1957b71cccb76fae67d256fd04f43db569e2fb0"
OFF = {"max_pattern_size": 0, "min_pattern_size": 0, "min_count": 0}
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
FINISH_REASONS = {"stop", "length", "tool_calls", "repetition"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def validate_cases(path):
    data = read_json(path)
    require(data["schema_version"] == 1, "unsupported case schema")
    require(data["source_revision"] == SOURCE_REVISION, "source revision changed")
    expected = [(f"score-short-{effort}", "seeded-sustainment", effort, "default", 64)
                for effort in (100, 75)]
    expected += [(f"active-turn-{effort}-{detector}", "active-turn-containment", effort, detector, 1024)
                 for effort in (100, 75) for detector in ("default", "off")]
    expected += [(f"exhausted-{effort}-{detector}", "clean-onset", effort, detector, 6000)
                 for effort in (100, 75) for detector in ("default", "off")]
    require(len(data["cases"]) == len(expected), "expected exactly ten requests")
    for source in data["sources"].values():
        require(source["revision"] == SOURCE_REVISION, "source revision mismatch")
        require(re.fullmatch(r"[0-9a-f]{64}", source["file_sha256"]), "invalid source file hash")
        require(sha256(canonical(source["case"])) == source["case_sha256"], "embedded source case changed")
    for case, (name, phase, effort, detector, cap) in zip(data["cases"], expected):
        require(case["name"] == name and case["phase"] == phase, "request order or phase changed")
        derived = {"effort": effort, "detector": detector, "max_tokens": cap, "return_token_ids": True}
        require(case["derivation"] == derived, f"{name}: unexpected experimental variable")
        source = data["sources"][case["source"]]["case"]
        body = copy.deepcopy(source["body"])
        body.update(max_tokens=cap, return_token_ids=True)
        if source["path"] == "/v1/chat/completions":
            body["chat_template_kwargs"]["reasoning_effort"] = effort
        if detector == "off":
            require(source["path"] == "/v1/chat/completions", "completions has no detector-off override")
            body["repetition_detection"] = OFF
        else:
            body.pop("repetition_detection", None)
        require(case["path"] == source["path"] and case["body"] == body, f"{name}: source derivation changed")
        require(body["temperature"] == 0 and "cache_salt" not in body, f"{name}: expected greedy uncached input")
        require(body["model"] == "deepseek-v4.1-flash", f"{name}: wrong served model")
        if source["path"] == "/v1/completions":
            require(len(body["prompt"]) == 579 and all(type(x) is int and x >= 0 for x in body["prompt"]),
                    f"{name}: expected 579 exact prefix IDs")
    require(data["cases"][0]["body"]["prompt"][101:] == data["cases"][1]["body"]["prompt"][101:],
            "sustainment fixture tails differ")
    require(data["bounds"] == {"requests_per_boot": 10, "max_output_tokens_per_boot": 28224,
                              "concurrency": 1, "request_timeout_seconds": 600,
                              "max_response_bytes": MAX_RESPONSE_BYTES}, "bounds changed")
    require(sum(c["body"]["max_tokens"] for c in data["cases"]) == 28224, "token bound mismatch")
    return data


def validate_configuration():
    """Check the tracked plan without networking or resolving a site's ignored map."""
    arm = read_json(HERE / "arm-loop.json")
    require(set(arm) <= {"notes", "extends"}, "loop arm must not change runtime settings")
    base_path = "experiments/2026-11-03-candidate-38-reproduction/tp3-switched.json"
    require(arm["extends"] == {"config": base_path}, "unexpected loop base profile")
    base = read_json(ROOT / base_path)
    require(base["set"]["container"]["image"] == "spark-ds41f:candidate-38", "expected candidate-38 image")
    labels = base["set"]["container"]["expected_labels"]
    require(labels.get("local.spark-ds41f.vllm.tree") == "c061e5f6931306fa4655ba531428f639c365bf10"
            and labels.get("local.spark-ds41f.sparknet.tree") == "8cc28a5cd3e683c1f114ed843453e3ce34f323e0",
            "expected final candidate-38 source trees")
    require(base["set"]["upstreams_config"] ==
            "experiments/2026-10-26-spark-ds41f-candidate/candidate-38/upstreams.lock.json",
            "expected unchanged candidate-38 source lock")
    require(not base["set"]["container"].get("mounts"), "source mounts are outside this experiment")
    spec = read_json(HERE / "lab-loop.json")
    require(spec["experiment"] == str(HERE.relative_to(ROOT)), "wrong lab experiment")
    require(len(spec["jobs"]) == 2 and spec["jobs"][0] == {"kind": "sync"}, "unexpected lab jobs")
    job = spec["jobs"][1]
    require(job["kind"] == "measure" and job["bench"] is False, "only diagnostic requests may run")
    require(job["arms"] == [
        {"config": "arm-loop.json", "label": "candidate38-loop-boot1", "startup_memory": True},
        {"config": "arm-loop.json", "label": "candidate38-loop-boot2", "startup_memory": True}],
        "expected two boots of the same unchanged arm")
    require(job["extras"] == [
        ["loop_probe.py", ["--run", "{run}", "--label", "{label}", "--request-timeout", "600"], "loop"],
        ["collect_runtime.py", ["--config", "experiments/2026-11-03-candidate-38-reproduction/arm-loop.json",
                                "--run", "{run}", "--label", "{label}"], "runtime"]],
            "unexpected lab workload or timeout")
    require((HERE / "collect_runtime.py").is_file(), "runtime collector is missing")
    nodes = base["set"].get("nodes_config", "config/nodes.json")
    return {"arm": "arm-loop.json", "boots": 2, "source_overlay": False, "effort_clamp": False,
            "expected_image": base["set"]["container"]["image"], "expected_labels": labels,
            "arm_sha256": sha256((HERE / "arm-loop.json").read_bytes()),
            "base_sha256": sha256((ROOT / base_path).read_bytes()),
            "lab_sha256": sha256((HERE / "lab-loop.json").read_bytes()),
            "source_lock_sha256": sha256((ROOT / base["set"]["upstreams_config"]).read_bytes()),
            "site_map": nodes, "site_map_present_locally": (ROOT / nodes).is_file(),
            "runtime_prerequisites": "lab window, matching candidate-38 image on every rank, site map, doctor and idle service"}


@contextlib.contextmanager
def deadline(seconds):
    """Bound the entire HTTP exchange, including a peer that trickles response bytes."""
    def expired(_signum, _frame):
        raise TimeoutError(f"request exceeded {seconds:g} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def reject_invalid_numbers(value):
    if isinstance(value, float):
        require(not math.isnan(value) and value != math.inf, "response contains NaN or positive infinity")
    elif isinstance(value, dict):
        for child in value.values():
            reject_invalid_numbers(child)
    elif isinstance(value, list):
        for child in value:
            reject_invalid_numbers(child)


def response_result(response, case):
    require(isinstance(response, dict) and not response.get("error"), "inference returned an error")
    reject_invalid_numbers(response)
    choices = response.get("choices")
    require(isinstance(choices, list) and len(choices) == 1, "expected one generated choice")
    choice = choices[0]
    require(choice.get("finish_reason") in FINISH_REASONS, f"unexpected finish reason: {choice.get('finish_reason')!r}")
    ids = choice.get("token_ids")
    require(isinstance(ids, list) and ids and all(type(i) is int and i >= 0 for i in ids), "missing generated token IDs")
    require(len(ids) <= case["body"]["max_tokens"], "server exceeded the output token cap")
    require(isinstance(choice.get("logprobs"), dict), "missing generated logprobs")
    require(isinstance(response.get("usage"), dict), "missing token usage")
    if case["phase"] == "seeded-sustainment":
        require(isinstance(choice.get("prompt_logprobs"), list), "missing teacher-forced prompt logprobs")
    message = choice.get("message") or {}
    return {"response_id": response.get("id"), "model": response.get("model"),
            "usage": response["usage"], "finish_reason": choice["finish_reason"],
            "stop_reason": choice.get("stop_reason"), "token_ids": ids,
            "logprobs": choice["logprobs"], "prompt_logprobs": choice.get("prompt_logprobs"),
            "prompt_token_ids": choice.get("prompt_token_ids", response.get("prompt_token_ids")),
            "message": message, "text": choice.get("text"),
            "reasoning_chars": len(message.get("reasoning") or message.get("reasoning_content") or ""),
            "tool_calls_executed": False}


def run_case(base_url, case, out, timeout):
    out.mkdir()
    body = copy.deepcopy(case["body"])
    body["cache_salt"] = uuid.uuid4().hex
    request_id = "candidate38-loop-" + uuid.uuid4().hex
    payload = json.dumps(body, ensure_ascii=False).encode()
    write_json(out / "request.json", {"case": case["name"], "source": case["source"],
               "derivation": case["derivation"], "path": case["path"], "request_id": request_id,
               "body_sha256": sha256(payload), "body": body})
    record = {"name": case["name"], "phase": case["phase"], **case["derivation"],
              "request_id": request_id, "started_at": now(), "artifacts": str(out)}
    started = time.monotonic()
    request = urllib.request.Request(base_url + case["path"], payload,
                                    {"Content-Type": "application/json", "X-Request-Id": request_id})
    raw = bytearray()
    try:
        with deadline(timeout):
            try:
                response = urllib.request.urlopen(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                record["http_status"] = response.status
                record["response_headers"] = {k: response.headers[k] for k in
                                               ("Content-Type", "X-Request-Id") if k in response.headers}
                with (out / "response.raw").open("wb") as handle:
                    while chunk := response.read(65536):
                        handle.write(chunk)
                        raw.extend(chunk)
                        require(len(raw) <= MAX_RESPONSE_BYTES, "response exceeded the byte cap")
        require(record["http_status"] == 200, f"HTTP {record['http_status']}")
        result = response_result(json.loads(raw), case)
        write_json(out / "result.json", result)
        record.update(ok=True, response_id=result["response_id"], usage=result["usage"],
                      finish_reason=result["finish_reason"], stop_reason=result["stop_reason"],
                      output_tokens=len(result["token_ids"]), reasoning_chars=result["reasoning_chars"])
    except (Exception, KeyboardInterrupt) as error:
        record.update(ok=False, error=f"{type(error).__name__}: {error}")
    record.update(ended_at=now(), elapsed_seconds=time.monotonic() - started,
                  response_bytes=len(raw), response_sha256=sha256(raw))
    write_json(out / "record.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", nargs="?", help="lab head URL; unused with --check")
    parser.add_argument("--cases", type=Path, default=HERE / "loop_cases.json")
    parser.add_argument("--check", action="store_true", help="validate cases and plan without network calls or writes")
    parser.add_argument("--run", default="candidate-38-loop-reproduction")
    parser.add_argument("--label", default="manual")
    parser.add_argument("--output-dir", "--out", type=Path)
    parser.add_argument("--request-timeout", type=float, default=600)
    args = parser.parse_args()
    require(0 < args.request_timeout <= 900, "request timeout must be in (0, 900] seconds")
    data = validate_cases(args.cases)
    configuration = validate_configuration()
    if args.check:
        print(json.dumps({"check": "passed", "network_requests": 0, "cases_sha256": sha256(args.cases.read_bytes()),
                          "bounds": data["bounds"], "configuration": configuration}, sort_keys=True))
        return 0
    require(hasattr(signal, "setitimer"), "a POSIX host is required for wall-clock request deadlines")
    require(args.base_url, "base_url is required for inference")
    url = urllib.parse.urlsplit(args.base_url)
    require(url.scheme in ("http", "https") and url.hostname and not url.username and not url.password
            and not url.query and not url.fragment and url.path in ("", "/"), "expected a bare HTTP(S) server URL")
    for value in (args.run, args.label):
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value), "run and label must be simple path components")
    out = args.output_dir or ROOT / "results/private/candidate-38-reproduction" / args.run / args.label / "loop"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "cases.json", data)
    write_json(out / "run.json", {"started_at": now(), "base_url": args.base_url, "run": args.run,
               "label": args.label, "request_timeout": args.request_timeout,
               "cases_sha256": sha256(args.cases.read_bytes()), "runner_sha256": sha256(Path(__file__).read_bytes()),
               "configuration": configuration, "bounds": data["bounds"]})
    records = []
    with (out / "requests.jsonl").open("x") as log:
        for index, case in enumerate(data["cases"], 1):
            record = run_case(args.base_url.rstrip("/"), case, out / f"{index:02d}-{case['name']}", args.request_timeout)
            records.append(record)
            line = json.dumps(record, ensure_ascii=False)
            print(line, file=log, flush=True)
            print(line, flush=True)
            if not record["ok"]:
                break
    summary = {"ok": len(records) == len(data["cases"]) and all(r["ok"] for r in records),
               "requests_completed": len(records), "requests_planned": len(data["cases"]),
               "failed": [r["name"] for r in records if not r["ok"]], "ended_at": now(),
               "artifacts": str(out), "interpretation": "HTTP/inference completion only; loop reproduction requires output review"}
    write_json(out / "summary.json", summary)
    print(json.dumps({"summary": summary}), flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
