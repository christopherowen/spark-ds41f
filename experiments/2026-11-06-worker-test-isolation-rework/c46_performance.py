"""Compact native completion audit; no timing-output analysis on serving hosts."""
import json
from pathlib import Path
import re
from servebench import suite
from c46_native import conditions_pass

ROOT = Path(__file__).resolve().parents[2]
NODES = {"dgx1", "dgx2", "dgx3"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.stat().st_size < 1048576, f"oversized compact receipt: {path}")
    return json.loads(path.read_text())


def validate(spec, image, commit, model_contract_sha256):
    job, summaries, boots, total = spec["jobs"][0], [], [], 0
    require(job.get("fail_on_suite_error") is True, "suite failure must stop the remaining cohort")
    require(re.fullmatch(r"[0-9a-f]{64}", model_contract_sha256 or ""), "frozen model contract hash required")
    require(job["suite"] == ["release-tp3", "single-stream", "sampled"], "performance families changed")
    for arm in job["arms"]:
        started = None
        for number, name in enumerate(job["suite"]):
            suffix = "" if number == 0 else "-" + name
            directory = ROOT / "results/private/suites" / (spec["run"] + suffix) / arm["label"]
            index, definition = read(directory / "suite.json"), suite.load(name)
            require(index["suite"] == definition and index["suite_sha256"] == suite.digest(definition),
                    f"suite definition: {arm['label']}/{name}")
            require(index.get("finished_utc") and [(s["id"], s["command"], s["exit"]) for s in index["steps"]]
                    == [(s["id"], s["command"], 0) for s in definition["steps"]],
                    f"incomplete/failed/ordered suite: {arm['label']}/{name}")
            require([s["output"] for s in index["steps"]] == [suite.output_name(s) for s in definition["steps"]],
                    "suite output names changed")
            if name == "release-tp3":
                model = index.get("model_evidence", {})
                require(model.get("required_evidence_passed") is True
                        and model.get("output") == "model.json.gz.evidence.json"
                        and model.get("contract_sha256") == model_contract_sha256,
                        "required model barrier missing, failed or bound to another contract")
                evidence = read(directory / model["output"])
                require(evidence.get("required_evidence_passed") is True and evidence.get("process_exit") == 0
                        and not evidence.get("errors")
                        and evidence.get("contract", {}).get("sha256") == model_contract_sha256,
                        "native model evidence failed or differs from frozen contract")
            identity = index["metadata"]["identity"]
            require(not identity["live_problems"] and not identity["client_dirty"]
                    and identity["client_commit"] == commit and identity["cluster_config"] == arm["config"]
                    and set(identity["nodes"]) == NODES, f"suite source/profile/ranks: {arm['label']}/{name}")
            times, content = {}, set()
            for node, state in identity["nodes"].items():
                require(state["image_id"] == image["image"] and state["checkout"] == commit,
                        f"suite image/checkout: {arm['label']}/{name}/{node}")
                require(all(state["image_labels"].get(k) == v for k, v in image["source"]["expected_labels"].items()),
                        f"suite image source labels: {node}")
                content.add((tuple(state["image_layers"]), state["image_content_sha256"]))
                times[node] = state["container_started_at"]
            require(len(content) == 1, "rank image content differs")
            started = times if started is None else started
            require(started == times, "companion suite reused another boot")
            conditions = read(directory / "conditions.json")
            require(conditions_pass(conditions, NODES, 3), "suite guard evidence is incomplete or adverse")
            telemetry = read(ROOT / "results/private/determinism" / spec["run"]
                             / f"suite-{arm['label']}{suffix}.telemetry.json")
            require(set(telemetry["nodes"]) == NODES and all(r["samples"] > 0 for r in telemetry["nodes"].values()),
                    "suite telemetry coverage incomplete")
            total += len(index["steps"])
            summaries.append({"label": arm["label"], "suite": name, "steps": index["steps"],
                              "finished_utc": index["finished_utc"], "suite_sha256": index["suite_sha256"]})
        startup = read(ROOT / "results/private/determinism" / spec["run"] / f"startup-memory-{arm['label']}.json")
        require(set(startup) == NODES and all(v["samples"] > 0 and v["min_available_gib"] >= 5
                for v in startup.values()), "metric boot startup memory incomplete/below floor")
        boots.append(started)
    require(len(boots) == 2 and len(summaries) == 6 and total == 26, "complete two-boot26-step cohort required")
    require(all(boots[0][node] != boots[1][node] for node in NODES), "metric repeat reused first boot")
    return {"passed": True, "commit": commit, "image": image["image"], "suite_count": 6,
            "ordered_step_count": total, "boot_started_at": boots, "suites": summaries,
            "limit": "Compact native completion/identity/guard audit only; final model/request/timing analysis remains required."}
