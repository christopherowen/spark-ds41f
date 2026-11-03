#!/usr/bin/env python3
"""Summarize downloaded loop receipts without networking or inference.

Verify the original file manifests first, then compare both boots and the
paired request controls. Generated texts remain in the private raw artifacts.
The output directory must not already contain the two report files.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = "candidate-38-loop-reproduction"
IMAGE = "sha256:213ba4df68d06665add7a26260c2321d55854a75b8a51ae71c72d0b97eac814b"
END_THINK = 128822


def load(path):
    return json.loads(path.read_text())


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return str(path.resolve().relative_to(ROOT))


def verify_snapshot(path, count):
    receipt = load(path / "receipt.json")
    require(len(receipt["files"]) == count, f"{path}: expected {count} source files")
    actual_files = {str(file.relative_to(path / "raw"))
                    for file in (path / "raw").rglob("*") if file.is_file()}
    require(actual_files == set(receipt["files"]),
            f"{path}: raw files and receipt manifest differ; refusing unverified inputs")
    for name, entry in receipt["files"].items():
        file = path / "raw" / name
        require(file.resolve().is_relative_to((path / "raw").resolve()), "invalid manifest path")
        raw = file.read_bytes()
        require(len(raw) == entry["size"] and digest(raw) == entry["sha256"], f"source changed: {file}")
    archive = path / "raw.tar.gz"
    archive_hash = digest(archive.read_bytes())
    if "archive_sha256" in receipt:
        require(archive_hash == receipt["archive_sha256"], "archive hash mismatch")
    return {"path": relative(path), "source_receipt_sha256": digest((path / "receipt.json").read_bytes()),
            "archive_sha256": archive_hash, "verified_files": count, "files": receipt["files"]}


def probability_stats(values):
    require(values and all(math.isfinite(x) and x <= 0 for x in values), "invalid probability values")
    return {"positions": len(values), "arithmetic_mean_probability": statistics.mean(math.exp(x) for x in values),
            "geometric_mean_probability": math.exp(statistics.mean(values)), "minimum_probability": math.exp(min(values))}


def longest_motif(ids):
    best = {"span_tokens": 0, "start_index": None, "copies": 0, "motif_ids": []}
    for size in range(1, 9):
        i = 0
        while i + size <= len(ids):
            motif, end = ids[i:i + size], i + size
            while end + size <= len(ids) and ids[end:end + size] == motif:
                end += size
            copies = (end - i) // size
            if copies >= 3 and end - i > best["span_tokens"]:
                best = {"span_tokens": end - i, "start_index": i, "copies": copies, "motif_ids": motif}
            i = end if copies > 1 else i + 1
    return best


def analyze_boot(raw, number):
    label = f"candidate38-loop-boot{number}"
    cases_dir = raw / "results/private/candidate-38-reproduction" / RUN / label / "loop"
    runtime_dir = raw / "results/private/determinism" / RUN / f"runtime-{label}"
    cases, run = load(cases_dir / "cases.json"), load(cases_dir / "run.json")
    require(digest((cases_dir / "cases.json").read_bytes()) == run["cases_sha256"], "case file hash mismatch")
    require(run["cases_sha256"] == digest((HERE / "loop_cases.json").read_bytes()), "requests differ from reviewed cases")
    runtime_summary = load(runtime_dir / "summary.json")
    require(runtime_summary["passed"], "runtime collector failed")
    ranks, all_logs = {}, []
    for node in ("sp5", "sp6", "sp7"):
        record = load(runtime_dir / node / "runtime.json")
        log_file = runtime_dir / node / "engine.log"
        log_bytes = log_file.read_bytes()
        require(digest(log_bytes) == record["log"]["sha256"], "engine log hash mismatch")
        require(not record["issues"] and record["image"]["id"] == IMAGE, "wrong image or rank evidence")
        require(not record["log"]["truncated"], "truncated logs cannot support absence of detector events")
        all_logs += [(relative(log_file), i + 1, line) for i, line in enumerate(log_bytes.decode().splitlines())]
        ranks[node] = {"image": record["image"], "container": record["container"],
                       "log": record["log"], "issues": record["issues"],
                       "record_path": relative(runtime_dir / node / "runtime.json")}
    rows, parsed = [], {}
    for index, case in enumerate(cases["cases"], 1):
        directory = cases_dir / f"{index:02d}-{case['name']}"
        request, result, record = [load(directory / f"{name}.json") for name in ("request", "result", "record")]
        require(record["ok"] and record["http_status"] == 200, "request failure must be investigated, not omitted")
        raw_response = (directory / "response.raw").read_bytes()
        require(digest(raw_response) == record["response_sha256"], "response hash mismatch")
        response = json.loads(raw_response)
        for field in ("token_ids", "logprobs", "finish_reason", "stop_reason"):
            require(response["choices"][0].get(field) == result.get(field), f"result {field} differs from raw response")
        require(response["usage"] == result["usage"], "result usage differs from raw response")
        body = dict(request["body"])
        body.pop("cache_salt")
        require(body == case["body"], "request differs from reviewed case")
        ids = result["token_ids"]
        logprobs = result["logprobs"].get("token_logprobs") or [entry["logprob"] for entry in result["logprobs"]["content"]]
        require(len(ids) == len(logprobs), "generated logprobs are not aligned")
        events = [{"path": path, "line": line, "text": text} for path, line, text in all_logs
                  if request["request_id"] in text and any(marker in text for marker in (
                      "Detected reasoning loop", "Suspected loop in the answer", "finished after loop detection"))]
        row = {"name": case["name"], "phase": case["phase"], "effort": case["derivation"]["effort"],
               "detector": case["derivation"]["detector"], "request_id": request["request_id"],
               "response_id": result["response_id"], "http_status": record["http_status"],
               "finish_reason": result["finish_reason"], "stop_reason": result["stop_reason"],
               "output_tokens": len(ids), "token_ids_sha256": digest(canonical(ids)), "usage": result["usage"],
               "reasoning_end_markers": [{"index": i, "token_id": END_THINK, "logprob": logprobs[i]}
                                          for i, token in enumerate(ids) if token == END_THINK],
               "longest_contiguous_motif_1_to_8": longest_motif(ids), "detector_events": events,
               "raw_response_path": relative(directory / "response.raw"),
               "result_path": relative(directory / "result.json")}
        if case["phase"] == "seeded-sustainment":
            prompt, plp = body["prompt"], result["prompt_logprobs"]
            require(len(prompt) == len(plp) == 579, "teacher-forced alignment differs")
            entries = [(prompt[i], plp[i].get(str(prompt[i]))) for i in range(101, 579)]
            require(all(entry is not None for _, entry in entries), "incomplete fixture scoring")
            thes = [entry for token, entry in entries if token == 455]
            row["teacher_forced_fixture"] = {"scored_tokens": len(entries), "fixture_tokens": 478,
                "token_id": 455, "token_text": " The", **probability_stats([entry["logprob"] for entry in thes]),
                "rank_counts": {str(rank): sum(entry["rank"] == rank for entry in thes)
                                for rank in sorted({entry["rank"] for entry in thes})}}
        rows.append(row)
        parsed[case["name"]] = {"request": request, "result": result}
    pairs = []
    for phase in ("active-turn", "exhausted"):
        for effort in (100, 75):
            default, off = [parsed[f"{phase}-{effort}-{mode}"] for mode in ("default", "off")]
            bodies = [dict(item["request"]["body"]) for item in (default, off)]
            for body in bodies:
                body.pop("cache_salt")
                body.pop("repetition_detection", None)
            require(bodies[0] == bodies[1], "paired prompts differ")
            a, b = [item["result"]["token_ids"] for item in (default, off)]
            prefix = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            pair = {"phase": phase, "effort": effort, "same_body_except_salt_and_detector": True,
                    "identical_output_tokens": a == b, "common_prefix_tokens": prefix}
            if prefix < min(len(a), len(b)):
                pair.update(default_next_id=a[prefix], off_next_id=b[prefix],
                            default_next_logprobs=default["result"]["logprobs"]["content"][prefix],
                            off_next_logprobs=off["result"]["logprobs"]["content"][prefix])
            pairs.append(pair)
    return {"label": label, "run_identity": run, "runner_summary": load(cases_dir / "summary.json"),
            "runtime_summary": runtime_summary, "ranks": ranks, "cases": rows, "paired_controls": pairs}, parsed


def check_run_claims(boots, repeated):
    """These conclusions describe this fixed run; changed evidence needs review."""
    expected_names = {f"score-short-{effort}" for effort in (100, 75)} | {
        f"{phase}-{effort}-{mode}" for phase in ("active-turn", "exhausted")
        for effort in (100, 75) for mode in ("default", "off")}
    require(len(boots) == 2, "the report requires two boots")
    require(len(repeated) == 10 and {row["name"] for row in repeated} == expected_names,
            "repeat comparison must cover all ten cases")
    for row in repeated:
        require(all(row[field] is True for field in ("token_ids_equal", "finish_reason_equal",
                                                    "stop_reason_equal", "generated_logprobs_equal")),
                f"repeat disagreement: {row['name']}")

    def forcing(row, expected):
        events = [event["text"] for event in row["detector_events"]
                  if "Detected reasoning loop" in event["text"]]
        require(len(events) == 1, f"{row['name']}: expected one forcing event")
        match = re.search(r"at (\d+) output tokens \(mode=([a-z_]+), pattern_len=(\d+), count=(\d+)\); forcing reasoning end token", events[0])
        require(match is not None and (int(match[1]), match[2], int(match[3]), int(match[4])) == expected,
                f"{row['name']}: forcing event changed")

    identities = []
    for boot in boots:
        summary = boot["runner_summary"]
        require(summary["ok"] and summary["requests_completed"] == summary["requests_planned"] == 10
                and not summary["failed"], "runner did not complete ten successful requests")
        rows = {row["name"]: row for row in boot["cases"]}
        require(len(boot["cases"]) == 10 and set(rows) == expected_names, "request set changed")
        require(all(row["http_status"] == 200 and row["stop_reason"] is None for row in rows.values()),
                "request status or stop reason changed")
        require(sum(row["phase"] == "seeded-sustainment" for row in rows.values()) == 2
                and sum(row["phase"] == "active-turn-containment" for row in rows.values()) == 4
                and sum(row["phase"] == "clean-onset" for row in rows.values()) == 4,
                "phase counts do not support the reported breakdown")
        pairs = {(pair["phase"], pair["effort"]): pair for pair in boot["paired_controls"]}
        require(len(pairs) == len(boot["paired_controls"]) == 4, "paired comparisons are incomplete")
        for effort, end_index, output, detected, mode, count, next_id, reasoning, clean_output in (
            (100, 172, 238, 161, "contiguous_phrase", 4, 455, 4497, 4666),
            (75, 194, 257, 192, "windowed_phrase", 5, 1926, 962, 1518),
        ):
            default, off = [rows[f"active-turn-{effort}-{setting}"] for setting in ("default", "off")]
            forcing(default, (detected, mode, 8, count))
            require(default["finish_reason"] == "tool_calls" and default["output_tokens"] == output
                    and [entry["index"] for entry in default["reasoning_end_markers"]] == [end_index],
                    f"effort {effort}: contained output changed")
            motif = off["longest_contiguous_motif_1_to_8"]
            require(off["finish_reason"] == "length" and off["output_tokens"] == 1024
                    and not off["reasoning_end_markers"] and not off["detector_events"]
                    and motif["span_tokens"] >= 750
                    and motif["start_index"] + motif["span_tokens"] >= 1008,
                    f"effort {effort}: disabled control no longer shows repetition to the cap")
            pair = pairs[("active-turn", effort)]
            require(pair["same_body_except_salt_and_detector"] and not pair["identical_output_tokens"]
                    and pair["common_prefix_tokens"] == end_index and pair["default_next_id"] == END_THINK
                    and pair["off_next_id"] == next_id, f"effort {effort}: containment comparison changed")
            for setting in ("default", "off"):
                clean = rows[f"exhausted-{effort}-{setting}"]
                require(clean["finish_reason"] == "stop" and not clean["detector_events"]
                        and clean["output_tokens"] == clean_output
                        and clean["usage"]["completion_tokens_details"]["reasoning_tokens"] == reasoning
                        and [entry["index"] for entry in clean["reasoning_end_markers"]] == [reasoning],
                        f"effort {effort}: clean-onset result no longer supports the conclusion")
            require(pairs[("exhausted", effort)]["identical_output_tokens"], "clean default/off outputs differ")
            short = rows[f"score-short-{effort}"]
            require(short["finish_reason"] == "length" and short["output_tokens"] == 64
                    and not short["reasoning_end_markers"], "short control no longer ends at its cap")
            score = short["teacher_forced_fixture"]
            require(score["scored_tokens"] == score["fixture_tokens"] == 478 and score["positions"] == 78
                    and sum(score["rank_counts"].values()) == 78, "incomplete teacher-forced statistics")
            if effort == 75:
                forcing(short, (60, "contiguous_phrase", 8, 4))
                require(any("status=FINISHED_LENGTH_CAPPED" in event["text"] for event in short["detector_events"]),
                        "pending forced end was not ended by the short control's cap")
            else:
                require(not short["detector_events"], "unexpected detector event in short effort-100 control")
        require(set(boot["ranks"]) == {"sp5", "sp6", "sp7"}, "rank evidence is incomplete")
        for rank in boot["ranks"].values():
            require(not rank["issues"] and not rank["log"]["truncated"] and rank["image"]["id"] == IMAGE
                    and rank["container"]["image_id"] == IMAGE and rank["container"]["status"] == "running",
                    "rank evidence does not support image/log claims")
            identities.append((rank["image"]["layers"], rank["image"]["config_sha256"]))
    require(all(identity == identities[0] for identity in identities), "image content differs across ranks or boots")
    for node in ("sp5", "sp6", "sp7"):
        first, second = [boot["ranks"][node]["container"] for boot in boots]
        require(first.get("id") and second.get("id") and first["id"] != second["id"],
                f"{node}: repeat uses the same container, not a fresh boot")
        require(first.get("started_at") and second.get("started_at")
                and first["started_at"] != second["started_at"],
                f"{node}: repeat has the same container start time")
    return {"repeat_agreement": True, "request_and_phase_counts": True, "seeded_control_caps": True,
            "forced_end_positions_and_events": True, "clean_onset_stops_and_no_events": True,
            "clean_reasoning_token_counts": True, "teacher_forced_coverage": True,
            "rank_image_content_and_untruncated_logs": True, "distinct_container_boots": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--boot1-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not any((args.output_dir / name).exists() for name in ("loop-analysis.json", "loop-receipt.json")),
            "refusing to overwrite recorded analysis")
    snapshots = [verify_snapshot(args.boot1_snapshot, 59), verify_snapshot(args.snapshot, 125)]
    raw = args.snapshot / "raw"
    boots, parsed = zip(*(analyze_boot(raw, number) for number in (1, 2)))
    repeated = []
    for name, first in parsed[0].items():
        second = parsed[1][name]
        one, two = first["result"], second["result"]
        bodies = [dict(x["request"]["body"]) for x in (first, second)]
        for body in bodies:
            body.pop("cache_salt")
        require(bodies[0] == bodies[1], "repeat request changed")
        repeated.append({"name": name, "token_ids_equal": one["token_ids"] == two["token_ids"],
                         "finish_reason_equal": one["finish_reason"] == two["finish_reason"],
                         "stop_reason_equal": one["stop_reason"] == two["stop_reason"],
                         "generated_logprobs_equal": one["logprobs"] == two["logprobs"]})
    claim_checks = check_run_claims(boots, repeated)
    analysis = {"schema_version": 1, "source": load(raw / "control/loop.command.json"),
        "request_source_revision": load(HERE / "loop_cases.json")["source_revision"],
        "request_fixture_provenance": load(HERE / "loop_cases.json")["fixture_provenance"],
        "counts": {"successful_http_requests": 20, "boots": 2, "seeded_sustainment_requests": 4,
                   "seeded_active_turn_requests": 8, "clean_onset_requests": 8, "clean_onset_conditions_per_boot": 4},
        "boots": boots, "repeat_comparison": repeated, "semantic_checks": claim_checks,
        "interpretation": [
            "Both seeded active-turn controls sustain degeneration with detection off through the 1,024-token cap.",
            "Default detection forces the reasoning end, confirmed by matched scheduler events. A later tool_calls finish is containment, not spontaneous recovery.",
            "All four clean-onset conditions per boot stop naturally without a detector event. Within each effort their default/off token IDs match exactly.",
            "Effort 100 produces 4,497 reasoning tokens versus 962 at effort 75 on this one clean fixture; this does not establish an onset threshold.",
            "The short completions are seeded sustainment probes with default detection active. At effort 75 the 64-token cap arrives before its pending forced end lands."],
        "limitations": [
            "Twenty successful HTTP requests are not twenty independent onset tests; only eight use the clean-onset fixture.",
            "The original long-session onset was not reproduced; these fixtures do not establish its absence or root cause.",
            "Two deterministic boots show repeatability for these inputs, not an incident-rate confidence interval.",
            "Tools were never executed. Syntactic tool-call completion is not an answer-quality assessment.",
            "No measured inference speed, CopyDrafter correctness, or promotion conclusion follows from these loop probes."]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "loop-analysis.json"
    output.write_text(json.dumps(analysis, indent=2) + "\n")
    receipt = {"schema_version": 1, "operation": "offline analysis only; no network or inference requests",
               "command": ["python3", relative(Path(__file__)), *sys.argv[1:]],
               "analyzer_sha256": digest(Path(__file__).read_bytes()), "snapshots": snapshots,
               "analysis_path": relative(output), "analysis_sha256": digest(output.read_bytes())}
    (args.output_dir / "loop-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": relative(output), "successful_http_requests": 20,
                      "all_repeat_tokens_equal": all(row["token_ids_equal"] for row in repeated),
                      "verified_source_files": [59, 125]}))


if __name__ == "__main__":
    main()
