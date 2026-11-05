#!/usr/bin/env python3
"""Publish a completed local snapshot without changing original evidence."""
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / "runs/2026-11-05-dgx/attempt-2"
RAW = OUT / "final-export/raw"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def main():
    receipt = read(OUT / "final-export/receipt.json")
    for name, record in receipt["files"].items():
        path = RAW / name
        assert path.stat().st_size == record["size"]
        assert digest(path) == record["sha256"], name
    control = read(RAW / "results/private/lab/candidate-41-round-attempt-2/receipt.json")
    actual = read(RAW / "control/exit.json")
    restoration = read(RAW / "control/restoration-ranks.json")
    assert control["completed"] and control["restored"] and not control.get("error")
    assert not control.get("restore_error") and not control.get("management_loss_latched")
    assert actual["actual_exit"] == 0 and restoration["passed"]
    assert set(restoration["process_states"].values()) == {"absent"}
    performance = read(OUT / "performance-analysis.json")
    functional = read(OUT / "functional-audit.json")
    recipes = read(OUT / "recipe-comparison.json")
    assert performance["evidence_passed"] and functional["native_evidence_passed"]

    # Preserve the mutable report state that the independent review actually read.
    review = read(OUT / "result-review.json")
    preserved = {}
    for record in review["bindings"]:
        path = ROOT / record["path"]
        assert digest(path) == record["sha256"], str(path)
        if path.name in ("decision.md", "qualification.md", "results.json"):
            dest = OUT / "review-inputs" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert not dest.exists()
            shutil.copyfile(path, dest)
            preserved[record["path"]] = {"snapshot": str(dest.relative_to(ROOT)),
                                        "sha256": digest(dest)}
    write_new(OUT / "review-input-snapshots.json", preserved)

    published = {}
    def publish(source, destination):
        target = OUT / destination
        assert source in receipt["files"] and not target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RAW / source, target)
        assert digest(target) == receipt["files"][source]["sha256"]
        published[destination] = {"source": source, "sha256": digest(target)}

    for family, suffix in (("release-tp3", ""), ("single-stream", "-single-stream"), ("sampled", "-sampled")):
        for label in ("candidate-41", "candidate-41-repeat"):
            prefix = f"results/private/suites/candidate-41-release-tp3{suffix}/{label}/"
            for source in receipt["files"]:
                if source.startswith(prefix) and source.endswith(".json"):
                    publish(source, f"performance/{label}/{family}/{Path(source).name}")
    for label in ("candidate-41", "candidate-41-repeat"):
        prefix = f"results/private/determinism/candidate-41-functional-tp3/runtime-{label}/"
        publish(prefix + "summary.json", f"functional/{label}/native-runtime-summary.json")
        for node in ("dgx1", "dgx2", "dgx3"):
            publish(prefix + node + "/runtime.json", f"functional/{label}/{node}/runtime.json")
    for source, destination in (
        ("results/private/lab/candidate-41-round-attempt-2/receipt.json", "controller-final.json"),
        ("results/private/lab/candidate-41-round-attempt-2/functional-exits.json", "functional-exits.json"),
        ("results/private/lab/candidate-41-round-attempt-2/cluster-start-plan.json", "restoration-plan.json"),
        ("results/private/candidate-rounds/candidate-41-qualified-image/receipt.json", "qualified-image.json"),
        ("control/exit.json", "exit-final.json"),
        ("control/restore-live-doctor.json", "restore-live-doctor.json"),
        ("control/restoration-ranks.json", "restoration-ranks.json"),
    ):
        publish(source, destination)
    write_new(OUT / "publication-receipt.json", {
        "native_file_hashes_verified": True, "verified_native_files": len(receipt["files"]),
        "published": published, "source_archive_sha256": receipt["archive_sha256"],
        "omitted": "Full raw export, generated model text/tokens and original logs remain private in final-export/raw and raw.tar.gz. Every native file is bound by final-export/receipt.json. Published timing/conditions/identity/control JSON is byte-identical. Audits preserve every adverse row.",
    })

    conditions = [v["conditions"] for boot in performance["boots"] for v in boot["suites"].values()]
    thermal = [v for c in conditions for v in c["thermal"].values()]
    memory = [v for c in conditions for v in c["memory"].values()]
    telemetry = []
    for boot in performance["boots"]:
        for family, data in boot["suites"].items():
            t = data["telemetry"]
            telemetry.append({"label": boot["label"], "suite": family,
                              "started_utc": t["started_utc"], "seconds": t["seconds"],
                              "nodes": {n: {"samples": v["samples"], "event_reasons": v["event_reasons"]}
                                        for n, v in t["nodes"].items()},
                              "halted": data["conditions"]["halted"],
                              "fan_stop_failures": data["conditions"]["fan_stop_failures"]})
    write_new(OUT / "telemetry-review.json", {
        "performance_analysis_sha256": digest(OUT / "performance-analysis.json"),
        "suites": telemetry,
        "operator_timeout": "Original passive observer SSH timeout is preserved. Native first-release telemetry spans 3272 seconds with 3271 samples on EACH rank, including the later observer gap. All six native guards are clear; no native management-loss latch or runtime relaunch occurred. Cause of operator transport failure is unproven; aggregate samples do not prove every management probe succeeded.",
        "power_event_observation": "Repeat release dgx3 has seven samples with event mask 0x4 (software power cap), despite conditions recording zero power-cap duration. Preserve both observations; do not claim no power-cap events or infer causality/duration.",
        "event_definition": "https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlClocksEventReasons.html",
        "no_hardware_or_policy_changes": True,
    })
    boots = []
    server_keys = ("accepted_per_draft", "draft_tokens_per_draft", "output_tokens_per_step", "preemptions", "prefix_cache_hit_rate", "steps_per_s")
    for boot in performance["boots"]:
        label, serving = boot["label"], {}
        for family, step in (("single-stream", "c1"), ("release-tp3", "c4"), ("release-tp3", "c8"), ("sampled", "c16")):
            summary = read(OUT / f"performance/{label}/{family}/{step}.json")["summary"]
            serving[f"{family}/{step}"] = {k: summary[k] for k in ("output_tps", "latency", "requests")}
            serving[f"{family}/{step}"]["server"] = {k: summary["server"][k] for k in server_keys}
        boots.append({"label": label, "serving": serving})
    write_new(OUT / "core-metrics.json", {
        "candidate": 41, "purpose": "cleanup_simplification",
        "performance_goal": "Correctness and no material incremental core/serving regression versus both C40 boots; declared before measurement.",
        "performance_verdict": "aggregate_serving_neutral_but_core_regressions_reject_simplification",
        "quality_verdict": "all_native_gates_and_functional_checks_passed_with_recorded_limits",
        "best_validated_default_enabled_base": 40,
        "image": performance["image_id"], "execution_commit": performance["deployment_commit"],
        "boots": boots,
        "timed_conditions": {
            "min_available_gib": min(v["min_available_gib"] for v in memory),
            "max_swap_growth_gib": max(v["swap_growth_gib"] for v in memory),
            "max_gpu_c": max(v["max_gpu_c"] for v in thermal),
            "min_thermal_headroom_c": min(v["min_headroom_c"] for v in thermal),
            "max_system_c": max(v["max_system_c"] for v in thermal),
            "thermal_slowdown_ms": max(v["thermal_slowdown_ms"] for v in thermal),
            "power_cap_ms": max(v["power_cap_ms"] for v in thermal),
            "sampled_sw_power_cap_events": {"candidate-41-repeat/release-tp3/dgx3": 7},
        },
        "api_ready_seconds": {phase: [float(v) for v in re.findall(r"cluster ready.*\(\+([0-9.]+)s\)", (RAW / f"results/private/lab/candidate-41-round-attempt-2/{phase}.log").read_text())] for phase in ("functional", "performance")},
        "all_reference_headlines": {name: {family: value["headlines"] for family, value in record["comparisons"].items()} for name, record in recipes.items()},
        "adverse_rows_vs_1030": performance["adverse_comparison_rows"],
        "limits": performance["limitations"] + [
            "C36 has one compatible saved boot; no reference repeat is invented.",
            "Seven software power-cap event samples coexist with zero recorded condition-counter duration; attribution is unresolved.",
            "Grid and pure cold-prefill do not directly exercise changed preparation; image-level regression attribution remains unresolved.",
            "All original comparisons and every adverse row remain in recipe-comparison.json. Boot2 non-paired floor views are supplemented without changing tolerances.",
        ],
    })
    print(json.dumps({"native_files_verified": len(receipt["files"]), "byte_identical_publications": len(published), "verdict": "reject_C41_retain_C40"}))


if __name__ == "__main__":
    main()
