#!/usr/bin/env python3
"""Coordinate the authorized C45 compiler build, complete gates and two-boot TP3 metrics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spark_ds41f import lab
import c45_cache
import c45_policy as policy
import c45_native
import c45_window as owned_window
from c45_management import ManagementLatch
import compiler_gate

EXPERIMENT = Path(__file__).resolve().parent.relative_to(ROOT).as_posix()
RESTORE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-36/tp3-dev.json"
CANDIDATE = "experiments/2026-10-26-spark-ds41f-candidate/candidate-45/tp3-qualification.json"
OUT = ROOT / "results/private/lab/candidate-45-round"
BUILD_ROOT = "/home/swank/projects/spark-ds41f-copy-default"
IMAGE = "spark-ds41f:candidate-45"
QUALIFIED = ROOT / "results/private/candidate-rounds/candidate-45-qualified-image/receipt.json"
EXPECTED_HEAD = "5f926df6d3126266a9c88c82c06544060008c33c"
EXPECTED_TREE = "ca1496c6481dc5bc4b4234b743e4ca023a6c7597"
TILELANG_HEAD = "a470d5daf8529fcf2b661f1119c8f4bb1aecc5e2"
TILELANG_TREE = "80fcfc4521b06e0c9776bbf1b5419849557e1090"


def validate_image_gate(image):
    gate, conditions = image.get("gate", {}), image.get("conditions", {})
    if not c45_native.conditions_pass(conditions, ["dgx4"], 4):
        raise ValueError("completed DGX4 memory/thermal/cooling guard evidence is missing or adverse")
    labels = image.get("source", {}).get("expected_labels", {})
    expected_hash = compiler_gate.TOPK_SHA
    if (not re.fullmatch(r"sha256:[0-9a-f]{64}", image.get("image", ""))
            or not image.get("ended") or image.get("passed") is not True or image.get("error")
            or gate.get("passed") is not True or gate.get("failed") or gate.get("errors")
            or not lab.suite_verdict([gate.get("summary", "")])["passed"]
            or not re.search(r"\b[1-9]\d* passed\b", gate.get("summary", ""))
            or conditions.get("halted") or conditions.get("fan_stop_failures")):
        raise ValueError("image lacks a clean completed release-test gate")
    if (image.get("source", {}).get("vllm_head") != EXPECTED_HEAD
            or labels.get("local.spark-ds41f.vllm.tree") != EXPECTED_TREE
            or image.get("topk_initcheck", {}).get("passed") is not True
            or image.get("topk_initcheck", {}).get("source_sha256") != expected_hash):
        raise ValueError("image source or actual-image top-k sanitizer gate differs")
    if (image.get("source", {}).get("tilelang_head") != TILELANG_HEAD
            or labels.get("local.spark-ds41f.tilelang.tree") != TILELANG_TREE
            or image.get("source", {}).get("tilelang_version") != compiler_gate.VERSION
            or image.get("source", {}).get("tilelang_patchset_sha256") != compiler_gate.PATCHSET):
        raise ValueError("image did not preserve the validated C45 compiler identity")
    cache = image.get("release_cache", {})
    manifest = image.get("gate_manifest", {})
    required_iket = "tests/kernels/test_deepseek_v41_tilelang_iket.py"
    if (cache.get("absent_before") is not True
            or cache.get("namespace_absent_before_import") is not True
            or cache.get("namespace") != compiler_gate.NAMESPACE
            or cache.get("version") != compiler_gate.VERSION
            or cache.get("image") != image["image"]
            or Path(cache.get("host_root", "")).name != "candidate-45-" + image["image"].split(":", 1)[1]
            or Path(cache.get("host_root", "")) != Path(BUILD_ROOT).parents[1] / ".cache/spark-lab/qualification" / ("candidate-45-" + image["image"].split(":", 1)[1])
            or any(not re.fullmatch(r"[0-9a-f]{64}", cache.get(k, "")) for k in ("generated_bundle_sha256", "actual_bundle_sha256"))
            or not re.fullmatch(r"[0-9a-f]{64}", manifest.get("sha256", ""))
            or manifest.get("files") != 136
            or manifest.get("required_iket_file") != required_iket):
        raise ValueError("full 136-file C40 release gates or fresh source-bound gate cache are missing")
    expected = ["build-prepare.log", "build-check.log", "build-image.log", "build-image-apply.log",
                "gate-manifest.log", "gate-bundle.log", "release-tests.log", "topk-initcheck.log",
                "compiler-identity.log", "compiler-features/boot1.log", "compiler-features/boot2.log"]
    if ([x.get("log") for x in image.get("steps", [])] != expected
            or any(x.get("exit") != 0 for x in image.get("steps", []))):
        raise ValueError("image preparation/build/gate execution is incomplete or out of order")
    audit = image.get("native_gate_audit", {})
    if (audit.get("passed") is not True or audit.get("files") != 136
            or audit.get("expected_files") != 136 or audit.get("passed_cases", 0) < 1
            or audit.get("failed_cases") != 0 or audit.get("error_cases") != 0
            or not re.fullmatch(r"[0-9a-f]{64}", audit.get("results_sha256", ""))):
        raise ValueError("all 136 original per-file native outcomes must pass with exact coverage")
    resources = image.get("resource_limits", {})
    if (resources.get("source_prepare") != policy.PREP_LIMITS
            or resources.get("release_gate") != policy.GATE_LIMITS
            or resources.get("topk_initcheck") != policy.INITCHECK_LIMITS
            or image.get("builder_worker", {}).get("passed") is not True):
        raise ValueError("bounded source/build/gate resource evidence is missing")
    regressions = image.get("native_regression_audit", {})
    if (regressions.get("passed") is not True or regressions.get("invariant_passed") is not True
            or regressions.get("invariant_occurrences") != 1
            or set(regressions.get("critical_files", {})) != set(c45_native.EXPECTED)
            or not all(v.get("passed") is True for v in regressions.get("critical_files", {}).values())):
        raise ValueError("original failed IKET invariant and worker cases must execute and pass")
    installed = image.get("installed_compiler_identity", {})
    features = image.get("compiler_features", {})
    plan_hash = hashlib.sha256((ROOT / EXPERIMENT / "gpu-gate-plan.json").read_bytes()).hexdigest()
    boots = features.get("boots", [])
    if (installed.get("passed") is not True
            or installed.get("cache", {}).get("namespace") != compiler_gate.NAMESPACE
            or features.get("passed") is not True or features.get("error")
            or features.get("image") != image["image"]
            or features.get("commit") != image.get("commit")
            or features.get("tilelang_head") != TILELANG_HEAD
            or features.get("tilelang_tree") != TILELANG_TREE
            or features.get("version") != compiler_gate.VERSION
            or features.get("namespace") != compiler_gate.NAMESPACE
            or features.get("plan_sha256") != plan_hash
            or len(boots) != 2
            or boots[0].get("container_id") == boots[1].get("container_id")
            or any(x.get("exit") != 0 or x.get("receipt", {}).get("passed") is not True for x in boots)):
        raise ValueError("new installed compiler lacks complete two-process feature/correctness gates")
    policy.validate_worker_receipt(image.get("builder_worker", {}), image.get("builder_worker_stop", {}))
    return image


def validate_gate_resume(previous, image):
    if (not previous.get("ended") or previous.get("restored") is not True
            or previous.get("completed") or previous.get("restore_error")
            or previous.get("management_loss_latched") or previous.get("management_unreachable")
            or previous.get("error") != "RuntimeError: window changed or closure requested"):
        raise ValueError("prior execution is not a safely closed scheduling handoff")
    steps = previous.get("steps", [])
    allowed = ["image-and-gate.log", "seed-image.log", "build-distribute.log"]
    if (not steps or len(steps) > len(allowed) or [x.get("log") for x in steps] != allowed[:len(steps)]
            or any(x.get("exit") != 0 for x in steps)):
        raise ValueError("a prior phase failed or functional/timed work already started")
    validate_image_gate(image)
    if image.get("commit") != previous.get("commit"):
        raise ValueError("image build lineage differs from the handoff")
    return image


def validate_functional_exits(log, spec):
    actual = re.findall(r"\b(\S+\.py) (candidate-45(?:-repeat)?) exit (\d+)\b", log)
    job = spec["jobs"][0]
    expected = [(Path(script).name, arm["label"], "0")
                for arm in job["arms"] for script, _, _ in job["extras"]]
    if actual != expected:
        raise ValueError("a functional extra failed, is missing, or differs from the ordered plan")
    return {"passed": True, "ordered_exits": actual}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--attempt", type=int, default=1)
    parser.add_argument("--resume-image-gate-from", metavar="RECEIPT",
                        help="reuse the completed gate after a scheduled handoff; never repeat timed work")
    args = parser.parse_args()
    if args.attempt < 1:
        parser.error("attempt must be positive")
    if args.attempt > 1:
        OUT = OUT.with_name(OUT.name + f"-attempt-{args.attempt}")
    if not args.apply:
        print(json.dumps({"restore": RESTORE, "candidate": CANDIDATE, "build_host": "dgx4", "serving_nodes": ["dgx1", "dgx2", "dgx3"], "specs": ["functional.json", "performance.json"], "output": str(OUT), "attempt": args.attempt, "resume_image_gate_from": args.resume_image_gate_from, "shape_change": False, "launch_blockers": policy.launch_blockers()}, indent=2))
        return 0
    policy.require_launch_ready()
    if os.uname().nodename != "dgx1" or OUT.exists() or lab.read_hold() or lab.REQUEST.exists() or lab.LEGACY_REQUEST.exists():
        raise SystemExit("wrong host, prior receipts, existing reservation or pending request; defer without duplicating")
    resume_path = (ROOT / args.resume_image_gate_from).resolve() if args.resume_image_gate_from else None
    resumed_image = None
    if resume_path:
        if not resume_path.is_relative_to((ROOT / "results/private/lab").resolve()):
            raise SystemExit("resume receipt must be retained under this checkout's native lab results")
        resumed_image = validate_gate_resume(json.loads(resume_path.read_text()),
                                            json.loads((resume_path.parent / "image-receipt.json").read_text()))
    for previous in (ROOT / "results/private/lab").glob("candidate-45-round*/receipt.json"):
        if previous.resolve() != resume_path and json.loads(previous.read_text()).get("steps"):
            raise SystemExit("a prior phase started; retain it and review before another round")
    lab.PRODUCTION_CONFIG = RESTORE
    token = uuid.uuid4().hex
    note = "owner-authorized compiler preparation qualification; DGX4 build/gates and TP3 repeat metrics; owner=" + token
    my_holder = lab.HOLDER_PREFIX + ": " + note
    management = ManagementLatch()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    try:
        opening = owned_window.open_window(token, note, commit, management)
    except BaseException as error:
        preflight = {"commit": commit, "owner_token": token, "steps": [],
                     "error": f"{type(error).__name__}: {error}", **management.snapshot()}
        actual = lab.read_hold()
        if isinstance(actual, dict) and actual.get("c45_owner_token") == token:
            try:
                preflight["restored"] = owned_window.close_window(actual["window"], my_holder, token, commit,
                    ROOT / "results/private/lab" / ("c45-open-cleanup-" + token), management)
            except BaseException as restore_error:
                preflight["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
        lab.write_json_atomic(ROOT / "results/private/lab" / ("c45-pre-window-" + token + ".json"), preflight)
        raise
    opened = lab.read_hold()
    if not opened or opened.get("holder") != my_holder:
        raise SystemExit("window ownership changed during opening; leave the foreign reservation alone")
    window = opened["window"]
    OUT.mkdir(parents=True)
    try:
        controller, watchdog = policy.capture_control_processes(commit, opening["watchdog_argv"], opening["watchdog_pid"])
        opened["resource_scope"] = policy.scope(token, my_holder, window, controller, RESTORE, watchdog)
        owned_window.update(window, my_holder, token, lambda current: current.update(resource_scope=opened["resource_scope"]))
        policy.assert_owned(lab.read_hold(), window, my_holder, token)
    except BaseException as error:
        setup = {"window": window, "holder": my_holder, "owner_token": token, "commit": commit,
                 "error": f"{type(error).__name__}: {error}", "steps": [], "restored": False}
        try:
            actual = lab.read_hold()
            if actual and actual.get("window") == window and actual.get("holder") == my_holder:
                setup["restored"] = owned_window.close_window(window, my_holder, token, commit, OUT / "setup-restoration", management)
        except BaseException as restore_error:
            setup["restore_error"] = f"{type(restore_error).__name__}: {restore_error}"
        setup["ended"] = lab.now()
        setup.update(management.snapshot())
        lab.write_json_atomic(OUT / "receipt.json", setup)
        raise
    receipt = {"started": lab.now(), "window": window, "holder": my_holder,
        "owner_token": token, "resource_scope": opened["resource_scope"],
        "commit": commit, "initial_source_identity": opening["sources"],
        "initial_restore_identity": opening["initial_restore_identity"], "steps": [], "restored": False}
    if resume_path:
        receipt["resumed_from"] = str(resume_path.relative_to(ROOT))
        receipt["previous_receipt_sha256"] = hashlib.sha256(resume_path.read_bytes()).hexdigest()
    done = threading.Event()
    def owns():
        hold = lab.read_hold()
        try:
            policy.assert_owned(hold, window, my_holder, token)
            return True
        except RuntimeError:
            return False
    def local_claims():
        snapshots = {}
        for host in policy.NODES:
            raw = management.ssh(host, ["python3", "-c", policy.CLAIM_PROBE], "local-claims", timeout=15)
            snapshots[host] = policy.validate_local_records(json.loads(raw), token, window)
        receipt["local_claim_snapshots"] = snapshots
    def save():
        receipt.update(management.snapshot())
        lab.write_json_atomic(OUT / "receipt.json", receipt)
    management.changed = save
    def heartbeat():
        while not done.wait(30):
            if owns():
                owned_window.beat(window, my_holder, token, receipt.get("phase", "candidate-45"))
    thread = threading.Thread(target=heartbeat, daemon=True); thread.start()
    def run(argv, name, allow_closing=False):
        if management.latched:
            raise RuntimeError("management/telemetry loss remains latched; no further phase")
        if not owns() or (not allow_closing and lab.window_should_close(lab.read_hold())):
            raise RuntimeError("window changed or closure requested")
        local_claims()
        receipt["latest_source_identity"] = owned_window.clean_sources(commit, management)
        receipt["phase"] = name
        record = {"argv": argv, "started": lab.now(), "log": name + ".log"}
        receipt["steps"].append(record); save()
        with (OUT / record["log"]).open("w") as log:
            process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=process.pid, process_group=process.pid, new_session=True); save()
            process.wait()
        record.update(exit=process.returncode, ended=lab.now()); save()
        management.observe_file(OUT / record["log"], name)
        if name == "image-and-gate":
            # Inspect only compact native guard reasons even when the child
            # failed before the ordinary full-receipt validation path.
            path = BUILD_ROOT + "/results/private/candidate-rounds/candidate-45-image/receipt.json"
            code = "import json,pathlib;p=pathlib.Path(" + repr(path) + ");x=json.loads(p.read_text()) if p.exists() else {};print(json.dumps({'exists':p.exists(),'conditions':{'halted':x.get('conditions',{}).get('halted',[])},'error':x.get('error')}))"
            try:
                compact = json.loads(management.ssh("dgx4", ["python3", "-c", code], "native-image-guard-status"))
                record["native_guard_status"] = compact
                management.observe_conditions(compact["conditions"], name, "dgx4")
                if compact.get("error"):
                    management.observe_text(compact["error"], name, "dgx4")
            except Exception as error:
                record["native_guard_read_error"] = f"{type(error).__name__}: {error}"
            save()
        if process.returncode == 255 and (argv[0] == "ssh" or name == "seed-image"):
            management.latch("management_transport_loss", name, "dgx4", "phase SSH exited255")
        if management.latched:
            raise RuntimeError("management/telemetry loss; no next phase or restoration launch")
        if process.returncode:
            raise RuntimeError(f"{name} exited {process.returncode}")
    def cli(config, operation, action, *extra):
        argv = [sys.executable, "bin/spark", "--cluster-config", config, operation, action, *extra]
        plan = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True)
        lab.write_json_atomic(OUT / f"{operation}-{action}-plan.json", {"argv": argv, "exit": plan.returncode, "stdout": plan.stdout, "stderr": plan.stderr})
        print(plan.stdout, flush=True)
        management.observe_text(plan.stdout + plan.stderr, operation + "-" + action + "-plan")
        if plan.returncode or "--apply would refuse" in plan.stdout or "shape:   CHANGES" in plan.stdout:
            raise RuntimeError("plan refused; no shape bypass")
        run([*argv, "--apply"], operation + "-" + action, allow_closing=config == RESTORE)
    code = 1
    try:
        if resumed_image is None:
            command = ["python3", f"{EXPERIMENT}/build_gate.py", "--window", window,
                       "--holder", my_holder, "--owner-token", token, "--apply"]
            remote = f"cd {shlex.quote(BUILD_ROOT)} && test $(git rev-parse HEAD) = {shlex.quote(receipt['commit'])} && " + shlex.join(command)
            run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "dgx4", remote], "image-and-gate")
        raw = management.ssh("dgx4", ["cat", f"{BUILD_ROOT}/results/private/candidate-rounds/candidate-45-image/receipt.json"], "image-gate-receipt")
        image = json.loads(raw)
        management.observe_conditions(image.get("conditions", {}), "completed-image-gate", "dgx4")
        if management.latched:
            raise RuntimeError("native image guard loss is sticky; no distribution or restoration launch")
        if resumed_image is not None:
            if image != resumed_image:
                raise RuntimeError("native image/gate receipt changed since the retained handoff")
            actual_build_image = management.ssh("dgx4", ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], "resume-immutable-image").strip()
            if actual_build_image != resumed_image["image"]:
                raise RuntimeError("DGX4 image differs from the completed gate")
        validate_image_gate(image)
        if image.get("commit") != commit:
            raise RuntimeError("image gate and all node checkouts must use this exact published execution commit")
        if QUALIFIED.exists():
            if json.loads(QUALIFIED.read_text()) != image:
                raise RuntimeError("refusing to replace a different qualified image receipt")
        else:
            QUALIFIED.parent.mkdir(parents=True, exist_ok=True)
            lab.write_json_atomic(QUALIFIED, image)
        lab.write_json_atomic(OUT / "image-receipt.json", image)
        if not image.get("passed") or not image.get("gate", {}).get("passed"):
            raise RuntimeError("image gate lacks passing receipt")
        existing = subprocess.run(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True, capture_output=True)
        if existing.returncode == 0 and existing.stdout.strip() != image["image"]:
            raise RuntimeError("refusing to overwrite a different image under the candidate-45 tag")
        run(["bash", "-o", "pipefail", "-c", "ssh -o BatchMode=yes -o ConnectTimeout=8 dgx4 docker save spark-ds41f:candidate-45 | docker load"], "seed-image")
        actual = subprocess.check_output(["docker", "image", "inspect", "spark-ds41f:candidate-45", "--format", "{{.Id}}"], text=True).strip()
        if actual != image["image"]:
            raise RuntimeError("head image digest differs from build receipt")
        cli(CANDIDATE, "build", "distribute")
        receipt["serving_cache_first_use"] = c45_cache.collect(
            management, lambda: policy.assert_owned(lab.read_hold(), window, my_holder, token),
            CANDIDATE, image["image"], commit, OUT / "serving-cache-first-use.json")
        save()
        for spec in ["functional.json", "performance.json"]:
            phase_environment = ["env", "SPARK_DS41F_BENCH_STREAM_POLICY=current"]
            if spec == "performance.json":
                template = (ROOT / EXPERIMENT / "model-contract.template.json").read_bytes()
                model_contract = json.loads(template)
                model_contract["candidate"].update(client_commit=commit, image_id=image["image"])
                frozen_contract = OUT / "model-contract.json"
                with frozen_contract.open("x") as output:
                    output.write(json.dumps(model_contract, indent=2) + "\n")
                contract_sha = hashlib.sha256(frozen_contract.read_bytes()).hexdigest()
                receipt["model_contract"] = {"path": str(frozen_contract), "sha256": contract_sha,
                    "template_sha256": hashlib.sha256(template).hexdigest(), "image": image["image"], "commit": commit}
                save()
                phase_environment += ["SPARK_DS41F_SUITE_MODEL_CONTRACT=" + str(frozen_contract),
                                      "SPARK_DS41F_SUITE_MODEL_CONTRACT_SHA256=" + contract_sha]
            run([*phase_environment, sys.executable, f"{EXPERIMENT}/c45_phase.py",
                 "--spec", f"{EXPERIMENT}/{spec}", "--out", str(OUT / (spec.removesuffix(".json") + "-control")),
                 "--window", window, "--holder", my_holder, "--owner-token", token], spec.removesuffix(".json"))
            if spec == "functional.json":
                checked = validate_functional_exits((OUT / "functional.log").read_text(),
                                                    json.loads((ROOT / EXPERIMENT / spec).read_text()))
                lab.write_json_atomic(OUT / "functional-exits.json", checked)
            audit_output = OUT / ("functional-semantics.json" if spec == "functional.json" else "performance-completion.json")
            audit_command = ["systemd-run", "--user", "--scope", "--quiet", "--collect",
                "-p", "CPUQuota=100%", "-p", "MemoryMax=4G", "-p", "MemorySwapMax=0", "-p", "TasksMax=64",
                sys.executable, f"{EXPERIMENT}/audit_phase.py", spec.removesuffix(".json"),
                "--spec", f"{EXPERIMENT}/{spec}", "--image-receipt", str(OUT / "image-receipt.json"),
                "--output", str(audit_output), "--commit", commit]
            if spec == "performance.json":
                audit_command += ["--model-contract-sha256", receipt["model_contract"]["sha256"]]
            run(audit_command, spec.removesuffix(".json") + "-audit")
            if json.loads(audit_output.read_text()).get("passed") is not True:
                raise RuntimeError("declared semantic/completion audit failed")

        receipt["completed"] = True
        code = 0
    except BaseException as error:
        receipt["error"] = f"{type(error).__name__}: {error}"
        print(receipt["error"], flush=True)
    finally:
        receipt["phase"] = "restoration"
        try:
            hold = lab.read_hold()
            if owned_window.matches(hold, window, my_holder, token, True) and hold.get("management_loss_latched"):
                management.latch("owned_child_hardware_review", "restoration", detail="owned child retained a sticky hardware-review lease")
            if management.latched:
                if owned_window.matches(lab.read_hold(), window, my_holder, token, True):
                    owned_window.hardware_review(window, my_holder, token, management)
                raise RuntimeError("management loss: reservation preserved; no restoration launch")
            if owns():
                receipt["restored"] = owned_window.close_window(window, my_holder, token, commit, OUT / "restoration", management)
            else:
                raise RuntimeError("window ownership changed; no serving action attempted")
            if not receipt["restored"]:
                raise RuntimeError("restoration verification failed; retain reservation")
        except BaseException as error:
            if management.latched and owned_window.matches(lab.read_hold(), window, my_holder, token, True):
                owned_window.hardware_review(window, my_holder, token, management)
            receipt["restore_error"] = f"{type(error).__name__}: {error}"
            code = 1
        done.set(); thread.join(timeout=2)
        receipt["ended"] = lab.now(); save()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
