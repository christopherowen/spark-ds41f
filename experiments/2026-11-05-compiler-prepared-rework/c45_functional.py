"""C45's pre-timing semantic barrier, preserving the recorded C40 contract."""
import hashlib
import json
from pathlib import Path
import re

from spark_ds41f import functional_evidence

ROOT = Path(__file__).resolve().parents[2]
NODES = ("dgx1", "dgx2", "dgx3")
QUALIFIED = "results/private/candidate-rounds/candidate-45-qualified-image/receipt.json"
FAULT = re.compile(rb"ERROR|CRITICAL|Traceback|device.side assert|CUDA error|illegal memory access|gate logits contain|EngineCore.*died", re.I)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def records(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            require(not line.lstrip().startswith(("{", "[")), f"malformed JSON: {path}")
            continue
        require(isinstance(value, dict), f"non-object JSON: {path}")
        rows.append(value)
    require(rows, f"no JSON records: {path}")
    return rows


def engine_log(path):
    digest, fault, tail = hashlib.sha256(), False, b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
            fault = fault or bool(FAULT.search(tail + block))
            tail = block[-1024:]
    return digest.hexdigest(), fault


def validate(_phase_out, spec, image, contract):
    """Read only this completed functional phase, before any metric boot starts.

    These predicates follow C40's final audit. The generic declared-evidence
    implementation remains the single owner of its four live contracts; this
    barrier also checks the other existing extras, rank identities and repeats.
    """
    require(contract["sampled_again"] == {"required": True, "expected": []}, "sampled repeat contract changed")
    job = spec["jobs"][0]
    declared = functional_evidence.contracts_for_extras(job, job["extras"])
    require(set(declared) == {"determinism", "greedy", "determinism-16k", "sampled"}, "declared functional stems changed")
    require(declared["sampled"]["batch_invariance"] == "observe"
            and declared["sampled"]["isolated_repeat"] == "require"
            and declared["sampled"].get("record_requests") is True, "sampled policy/recording changed")
    directory = ROOT / "results/private/determinism" / spec["run"]
    expected_image = image["image"]
    require(load(ROOT / QUALIFIED) == image, "qualified image receipt changed")
    collector_hash = sha((ROOT / "scripts/collect_runtime.py").read_bytes())
    receipt_hash = sha((ROOT / QUALIFIED).read_bytes())
    wanted_labels = image["source"]["expected_labels"]
    boots, evidence, sampled = [], [], []
    for arm in job["arms"]:
        label, summaries = arm["label"], {}
        for _, arguments, stem in job["extras"]:
            path = directory / f"{stem}-{label}.jsonl"
            rows = records(path)
            final = rows[-1]
            require(Path(str(path) + ".telemetry.json").is_file(), f"missing extra telemetry: {label}/{stem}")
            if stem in declared:
                native = load(Path(str(path) + ".evidence.json"))
                fresh = functional_evidence.evaluate(path, declared[stem])
                require(native.get("process_exit") == 0 and native.get("required_evidence_passed") is True
                        and fresh["required_evidence_passed"] is True,
                        f"declared evidence failed: {label}/{stem}")
                require(all(native.get(key) == value for key, value in fresh.items()),
                        f"declared receipt/source/contract differs: {label}/{stem}")
                evidence.append({"label": label, "stem": stem, "source_sha256": fresh["source_sha256"],
                                 "observed": fresh["observed"],
                                 "request_records": fresh.get("request_records")})
            if stem in ("determinism", "greedy", "determinism-16k"):
                require(final.get("identical") is True and all(v["token_diffs"] == v["logprob_diffs"] == 0
                        for v in final["scenarios"].values()), f"determinism failed: {label}/{stem}")
                names = {"c8", "staggered", "mixed", "cached", "again"} | ({"mixed15"} if "--wide" in arguments else set())
                require(set(final["scenarios"]) == names and len(rows) == len(names) + 1,
                        f"determinism coverage: {label}/{stem}")
                for row in rows[:-1]:
                    count = 16 if row["scenario"] == "mixed15" else 9 if row["scenario"] == "mixed" else 8
                    require(row["scenario"] in names and len(row["requests"]) == count
                            and row["token_diffs"] == row["logprob_diffs"] == 0,
                            f"determinism request count/diffs: {label}/{stem}")
                    require(all(r.get("first_token_diff") is None and r.get("first_char_diff") is None
                                and r.get("first_logprob_diff") is None and r.get("logprobs_equal") is not False
                                for r in row["requests"]), f"raw determinism differs: {label}/{stem}")
            elif stem in ("control-spellings", "controls"):
                require(final["summary"]["pass"] is True, f"API controls: {label}/{stem}")
            elif stem == "probes":
                require(final["summary"]["prompts"] == 16 and not final["summary"]["collapsed"], f"collapse probe: {label}")
            elif stem in ("sink-ordinary", "sink-after-probes"):
                require(final["summary"]["fallthroughs"] == 0, f"sink fallthrough: {label}/{stem}")
                if stem == "sink-ordinary":
                    require(final["summary"]["flagged_rows"] == 0, f"ordinary traffic flagged: {label}")
            elif stem in ("controls-digest", "prompt-logprobs"):
                require(not final["failed"] and final["digests"], f"control request failed: {label}/{stem}")
            elif stem == "images":
                require(final["digests"]["alone"] == final["digests"]["c4"], f"image batch differs: {label}")
            elif stem == "ced-cross-shape":
                require(final["passed"] and all(v["repeats_identical"] and v["cold"]
                        for v in final["lengths"].values()), f"CED repeats: {label}")
            elif stem == "runtime":
                require(final["expected_image_id"] == expected_image and final["collector_sha256"] == collector_hash,
                        f"collector immutable identity: {label}")
                require(final["profile"] == arm["config"]
                        and final["profile_sha256"] == sha((ROOT / arm["config"]).read_bytes()), f"profile hash: {label}")
                require(final["passed"] is True and not final["issues"]
                        and final["collected_nodes"] == final["expected_nodes"] == list(NODES), f"runtime collector: {label}")
                require(final["image_receipt"] == QUALIFIED and final["image_receipt_sha256"] == receipt_hash,
                        f"collector qualified receipt: {label}")
            elif stem in ("c4-distinct", "c8-distinct"):
                count = int(stem[1])
                require(final["samples"] == 4 and final["foreign_requests"] == 0 and len(rows) == 5,
                        f"concurrency coverage: {label}/{stem}")
                require(all(r.get("tokens") == count * 256 and r.get("foreign_requests") == 0 for r in rows[:-1]),
                        f"concurrency tokens/foreign: {label}/{stem}")
            elif stem == "copy-probe":
                require([r["category"] for r in rows] == ["prose", "copy", "edit", "tool"]
                        and all(r["requests"] == 12 and r["errors"] == 0 and r["output_tokens"] > 0 for r in rows),
                        f"copy categories/errors: {label}")
                final = {"categories": [{k: v for k, v in r.items() if k not in ("suffix", "dspark_acceptance_by_position")}
                                        for r in rows]}
            elif stem == "memory":
                ranks = {r["node"]: r for r in rows if r["node"] in NODES}
                require(set(ranks) == set(NODES) and all(r["mem_available_gib"] > 0 and r["mem_total_gib"] > 0
                        and r["container"] for r in ranks.values()), f"memory ranks: {label}")
                final = {"ranks": ranks}
            elif stem == "trace":
                trace_path = (ROOT / final["output"]).resolve()
                require(trace_path.is_relative_to((ROOT / "results/private").resolve()), "trace escapes private results")
                trace = load(trace_path)
                require(len(trace["requests"]) > 12 and all(not r.get("error") and r.get("completion_tokens") == 256
                        for r in trace["requests"]), f"trace requests: {label}")
                require(all(trace["phases"][name]["streams"] == count and trace["phases"][name]["seconds"] == seconds
                            and final["summary"][name]["foreign_requests"] == 0
                            for name, count, seconds in (("c4", 4, 120), ("c8", 8, 60))), f"trace workload: {label}")
            elif stem == "sampled":
                require(not final["scenarios"]["again"]["differing"], f"isolated sampled repeat: {label}")
                sampled.append(rows)
            elif stem == "startup":
                require(final.get("summary") == "startup lines", f"startup observation missing: {label}")
            else:
                raise ValueError(f"unreviewed functional stem: {stem}")
            summaries[stem] = final
        ranks, fingerprints = {}, set()
        for node in NODES:
            p = directory / f"runtime-{label}" / node
            record = load(p / "runtime.json")
            digest, fault = engine_log(p / "engine.log")
            c, actual = record["container"], record["image"]
            require(not record["issues"] and not record.get("error") and c["image_id"] == actual["id"] == expected_image
                    and c["profile"] == arm["config"], f"rank immutable identity: {label}/{node}")
            require(all(actual["labels"].get(k) == v for k, v in wanted_labels.items()), f"rank source labels: {label}/{node}")
            require(c["status"] == record["status_after_collection"] == "running" and c["restart_count"] == 0
                    and record["same_container_after_collection"], f"rank changed: {label}/{node}")
            require(record["log"]["truncated"] is False and digest == record["log"]["sha256"] and not fault,
                    f"rank log/hash/fault: {label}/{node}")
            fingerprints.add((tuple(actual["layers"]), actual["config_sha256"]))
            ranks[node] = {"container_id": c["id"], "started_at": c["started_at"], "image_id": actual["id"], "log_sha256": digest}
        require(len(fingerprints) == 1, f"rank image layers/config differ: {label}")
        startup = load(directory / f"startup-memory-{label}.json")
        require(set(startup) == set(NODES), f"startup memory coverage: {label}")
        for node in NODES:
            lines = (directory / f"startup-memory-{label}-{node}.txt").read_text().splitlines()
            require(len(lines) == startup[node]["samples"] and lines, f"startup memory samples: {label}/{node}")
            minimum = min(float(line.split()[1]) for line in lines) / 1048576
            require(round(minimum, 3) == startup[node]["min_available_gib"] and minimum >= 5,
                    f"startup memory guard/summary: {label}/{node}")
        boots.append({"label": label, "summaries": summaries, "ranks": ranks, "startup_memory": startup})
    require(len(boots) == 2 and len(evidence) == 8, "functional boots/declared receipts incomplete")
    for node in NODES:
        a, b = [boot["ranks"][node] for boot in boots]
        require(a["container_id"] != b["container_id"] and a["started_at"] != b["started_at"], f"repeat reuses boot: {node}")
    for stem in ("determinism", "greedy", "determinism-16k", "prompt-logprobs", "images", "ced-cross-shape"):
        require(boots[0]["summaries"][stem] == boots[1]["summaries"][stem], f"cross-boot digest: {stem}")
    a, b = [boot["summaries"]["controls-digest"]["digests"] for boot in boots]
    changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    require(set(changed) <= {"penalties-sampled/c8"}, "unexpected cross-boot control digest change")
    return {"passed": True, "image_id": expected_image, "boots": boots, "declared_evidence": evidence,
            "sampled_rows_equal_across_boots": sampled[0] == sampled[1],
            "stochastic_control_digest_changes": changed,
            "limitations": ["Seeded C8 sampling can vary with batching; this is not a batch-invariance pass.",
                            "Model semantic enforcement belongs to the existing first model step before the engine grid.",
                            "No native logit-position/step-layout or full-vocabulary quality claim."]}
