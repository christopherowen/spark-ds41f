"""Opt-in exact model barrier for an existing suite step; never sends requests.

The contract and both saved reference files are hash-bound. References are read
only when the opted-in suite is about to run, under its caller's owned window.
This is separate from observational JSONL functional-collector requirements.
"""
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time

from . import behaviour

CONTRACT_ENV = "SPARK_DS41F_SUITE_MODEL_CONTRACT"
HASH_ENV = CONTRACT_ENV + "_SHA256"
SCHEMA = "servebench-model-evidence/1"
WORKLOAD_KEYS = (*behaviour.WORKLOAD_KEYS, "extra_body")
SCORE_KEYS = ("id", "tokens", "logprob", "top_ids", "top_logprobs")
MAX_ENCODED_BYTES = 32 * 1024**2
MAX_DECODED_BYTES = 256 * 1024**2
PARSER_AS_BYTES = 4 * 1024**3
PARSER_CPU_SECONDS = 30
PARSER_WALL_SECONDS = 60
PARSER_LIMITS = {"cpu_affinity_count": 1, "address_space_bytes": PARSER_AS_BYTES,
                 "cpu_soft_seconds": PARSER_CPU_SECONDS, "cpu_hard_seconds": PARSER_CPU_SECONDS + 5,
                 "wall_seconds": PARSER_WALL_SECONDS, "encoded_bytes": MAX_ENCODED_BYTES,
                 "decoded_bytes": MAX_DECODED_BYTES,
                 "scope": "Parser subprocess only. AS is a virtual-address-space bound, not an RSS, cgroup or swap guarantee; the benchmark client is unchanged."}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hash(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is not None


def _nonfinite(value):
    raise ValueError("non-finite model evidence: " + value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _score_number(value):
    # Existing compare_scores uses numeric ==, including signed zero and 1/1.0.
    # Exact integer ratios also avoid rounding large integral values together.
    return value.as_integer_ratio() if type(value) is float else (value, 1)


class _HashReader(io.RawIOBase):
    def __init__(self, raw, limit, name):
        self.raw = raw
        self.limit = limit
        self.name = name
        self.digest = hashlib.sha256()
        self.size = 0

    def readable(self):
        return True

    def readinto(self, target):
        # Probe at most one byte past the cap, never an unrestricted read.
        amount = min(len(target), self.limit - self.size + 1)
        require(amount > 0, self.name + " byte limit exceeded")
        count = self.raw.readinto(memoryview(target)[:amount])
        if count:
            self.digest.update(memoryview(target)[:count])
            self.size += count
            require(self.size <= self.limit, self.name + " byte limit exceeded")
        return count


def read_model(path, expected_hash=None, expected_bytes=None):
    """Low-level byte-capped parser; production callers use bounded_fingerprints."""
    path = Path(path)
    size_before = path.stat().st_size
    require(size_before <= MAX_ENCODED_BYTES, "encoded model byte limit exceeded before parsing")
    if expected_bytes is not None:
        require(size_before == expected_bytes, "saved model artifact size differs before parsing: " + str(path))
    with path.open("rb") as raw:
        reader = _HashReader(raw, min(MAX_ENCODED_BYTES, expected_bytes or MAX_ENCODED_BYTES), "encoded model")
        with io.BufferedReader(reader) as buffered:
            decoded = gzip.GzipFile(fileobj=buffered) if path.suffix == ".gz" else buffered
            decoded_reader = _HashReader(decoded, MAX_DECODED_BYTES, "decoded model")
            with io.TextIOWrapper(io.BufferedReader(decoded_reader), encoding="utf-8") as stream:
                value = json.load(stream, parse_constant=_nonfinite)
        digest, size = reader.digest.hexdigest(), reader.size
    if expected_hash is not None:
        require(digest == expected_hash, "saved model artifact hash differs: " + str(path))
    if expected_bytes is not None:
        require(size == expected_bytes, "saved model artifact size differs: " + str(path))
    return value, {"path": str(path), "sha256": digest, "bytes": size,
                   "decoded_bytes": decoded_reader.size}


def fingerprints(value):
    """Compact exact oracle using the existing generation/score comparison fields."""
    require(isinstance(value, dict) and value.get("schema") == behaviour.SCHEMA,
            "model result schema differs")
    workload = value["workload"]
    count = workload["prompt_count"]
    require(type(count) is int and count > 0, "model prompt count is invalid")
    require(is_hash(workload.get("prompts_sha256")), "model prompt source hash is missing")
    sequences, scores = value["generation"]["sequences"], value["scores"]
    require(isinstance(sequences, list) and isinstance(scores, list), "model arrays are missing")
    expected_ids = set(range(count))
    for name, rows in (("generation", sequences), ("scores", scores)):
        require(len(rows) == count and all(isinstance(row, dict) and not row.get("error") for row in rows),
                "model " + name + " is incomplete or failed")
        ids = [row.get("id") for row in rows]
        require(all(type(i) is int for i in ids) and len(set(ids)) == count and set(ids) == expected_ids,
                "model " + name + " prompt IDs differ")
    by_id = {row["id"]: row for row in sequences}
    positions = 0
    finite = lambda x: type(x) is int or type(x) is float and math.isfinite(x)
    tokens = lambda xs: isinstance(xs, list) and bool(xs) and all(type(i) is int and i >= 0 for i in xs)
    for row in sequences:
        require(tokens(row.get("prompt_ids")) and tokens(row.get("output_ids")), "model token IDs are missing")
        require(len(row["output_ids"]) <= workload["max_tokens"], "model output exceeds its declared budget")
    for row in scores:
        ids = row["tokens"]
        require(tokens(ids) and ids == by_id[row["id"]]["output_ids"], "model scores cover different tokens")
        require(all(isinstance(row.get(k), list) and len(row[k]) == len(ids)
                    for k in SCORE_KEYS[2:]), "model score positions are missing")
        require(all(finite(x) for x in row["logprob"]), "model given-token logprob is non-finite")
        for top_ids, top_lps in zip(row["top_ids"], row["top_logprobs"]):
            require(tokens(top_ids) and len(set(top_ids)) == len(top_ids)
                    and isinstance(top_lps, list) and len(top_ids) == len(top_lps) <= workload["top"]
                    and all(finite(x) for x in top_lps), "model top-logprob positions are incomplete or non-finite")
        positions += len(ids)
    generation_hash = behaviour.corpus_digest(sequences)
    require(value["corpus"]["sha256"] == generation_hash and value["corpus"]["sequences"] == count,
            "model corpus hash/count contradicts the actual generated/scored tokens")
    score_payload = [{"id": row["id"], "tokens": row["tokens"], "top_ids": row["top_ids"],
                      "logprob": [_score_number(x) for x in row["logprob"]],
                      "top_logprobs": [[_score_number(x) for x in position] for position in row["top_logprobs"]]}
                     for row in sorted(scores, key=lambda row: row["id"])]
    return {"generation_sha256": generation_hash,
            "scores_sha256": hashlib.sha256(_canonical(score_payload)).hexdigest(),
            "corpus_sha256": value["corpus"]["sha256"], "sequences": count, "positions": positions,
            "workload": {key: workload.get(key) for key in WORKLOAD_KEYS}}


def rank_observation(value):
    """Rank metadata is observed, not part of compare_scores distribution equality."""
    rows = value["scores"]
    shape_valid = all(isinstance(row.get("rank"), list)
                      and len(row["rank"]) == len(row["tokens"]) for row in rows)
    values = [rank for row in rows if isinstance(row.get("rank"), list) for rank in row["rank"]]
    return {"shape_valid": shape_valid,
            "value_types_valid": all(isinstance(row.get("rank"), list) for row in rows)
                                 and all(rank is None or type(rank) is int for rank in values),
            "positions": len(values), "none_positions": sum(rank is None for rank in values),
            "sha256": hashlib.sha256(_canonical([row.get("rank") for row in rows])).hexdigest()}


class ParserFailure(ValueError):
    def __init__(self, message, receipt):
        super().__init__(message)
        self.receipt = receipt


def _artifact_stat(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": None,
            "hash_status": "not completed; original file retained"}


def _bounded_output(process):
    stdout, stderr = bytearray(), bytearray()
    deadline = time.monotonic() + PARSER_WALL_SECONDS
    with selectors.DefaultSelector() as ready:
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            ready.register(stream, selectors.EVENT_READ, name)
        while ready.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("parser wall limit exceeded")
            for key, _ in ready.select(min(1, remaining)):
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    ready.unregister(key.fileobj)
                    continue
                if key.data == "stdout":
                    require(len(stdout) + len(chunk) <= 65536, "parser receipt is oversized")
                    stdout.extend(chunk)
                else:
                    stderr.extend(chunk)
                    del stderr[:-1500]
        process.wait(timeout=max(0.01, deadline - time.monotonic()))
    return bytes(stdout), bytes(stderr)


def bounded_fingerprints(path, expected_hash=None, expected_bytes=None):
    """Fresh CPU-only interpreter confines parsing; no client affinity/limits change."""
    source_root = Path(__file__).resolve().parents[1]
    argv = [sys.executable, "-m", "servebench.model_evidence", "--fingerprint-worker", str(Path(path).resolve())]
    if expected_hash is not None:
        argv += ["--expected-hash", expected_hash]
    if expected_bytes is not None:
        argv += ["--expected-bytes", str(expected_bytes)]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "MAX_JOBS"):
        environment[key] = "1"
    record = {"requested_limits": PARSER_LIMITS, "started_monotonic": time.monotonic()}
    process = subprocess.Popen(argv, cwd=source_root, env=environment, stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    record["pid"] = process.pid
    try:
        stdout, stderr = _bounded_output(process)
    except BaseException as error:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            record["kill_reap_timeout"] = True
        record.update(exit=process.poll(), error=type(error).__name__ + ": " + str(error),
                      seconds=round(time.monotonic() - record.pop("started_monotonic"), 3))
        raise ParserFailure("bounded model parser failed: " + record["error"], record) from error
    finally:
        process.stdout.close()
        process.stderr.close()
    record.update(exit=process.returncode, seconds=round(time.monotonic() - record.pop("started_monotonic"), 3),
                  stdout_sha256=hashlib.sha256(stdout).hexdigest(), stderr_tail=stderr.decode(errors="replace")[-1500:])
    try:
        require(len(stdout) <= 65536, "parser receipt is oversized")
        parsed = json.loads(stdout, parse_constant=_nonfinite)
        record["actual_limits"] = parsed.get("limits")
        require(process.returncode == 0 and parsed.get("passed") is True,
                "model parser failed: " + str(parsed.get("error", process.returncode)))
        require(parsed.get("module_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model parser source differs from the client")
        return {**parsed, "worker": record}
    except Exception as error:
        if isinstance(locals().get("parsed"), dict):
            record["parse_error"] = parsed.get("error")
            record["artifact"] = parsed.get("artifact")
        raise ParserFailure(str(error), record) from error


def _worker_main():
    import argparse
    import resource
    parser = argparse.ArgumentParser(description="Private bounded model-file parser; no requests.")
    parser.add_argument("--fingerprint-worker", type=Path, required=True)
    parser.add_argument("--expected-hash")
    parser.add_argument("--expected-bytes", type=int)
    args = parser.parse_args()
    reserve = bytearray(256 * 1024)
    result = {"passed": False, "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    try:
        require(sys.platform == "linux", "bounded model parsing requires the owned Linux runner")
        inherited_as = resource.getrlimit(resource.RLIMIT_AS)[1]
        address_limit = PARSER_AS_BYTES if inherited_as == resource.RLIM_INFINITY else min(PARSER_AS_BYTES, inherited_as)
        inherited_cpu = resource.getrlimit(resource.RLIMIT_CPU)[1]
        cpu_hard = PARSER_CPU_SECONDS + 5 if inherited_cpu == resource.RLIM_INFINITY else min(PARSER_CPU_SECONDS + 5, inherited_cpu)
        resource.setrlimit(resource.RLIMIT_AS, (address_limit, address_limit))
        resource.setrlimit(resource.RLIMIT_CPU, (min(PARSER_CPU_SECONDS, cpu_hard), cpu_hard))
        os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
        result["limits"] = {**PARSER_LIMITS, "actual_affinity": sorted(os.sched_getaffinity(0)),
                            "actual_address_space": resource.getrlimit(resource.RLIMIT_AS),
                            "actual_cpu_seconds": resource.getrlimit(resource.RLIMIT_CPU)}
        value, artifact = read_model(args.fingerprint_worker, args.expected_hash, args.expected_bytes)
        result["artifact"] = artifact
        result["fingerprints"] = fingerprints(value)
        result["observed"] = {"rank": rank_observation(value)}
        require(len(_canonical(result)) <= 32768, "compact model parser receipt is oversized")
        result["passed"] = True
    except Exception as error:
        del reserve
        result.pop("fingerprints", None)
        result.pop("observed", None)
        result["error"] = type(error).__name__ + ": " + str(error)[:500]
        if "artifact" not in result:
            try:
                result["artifact"] = _artifact_stat(args.fingerprint_worker)
            except OSError:
                pass
    print(json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["passed"] else 1


def load_binding(path, expected_hash):
    require(is_hash(expected_hash), "an exact frozen model-contract SHA256 is required")
    path = Path(path).resolve()
    require(path.stat().st_size <= 65536, "model contract is oversized")
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == expected_hash, "model contract bytes changed")
    contract = json.loads(data, parse_constant=_nonfinite)
    require(contract.get("schema") == SCHEMA, "model contract schema differs")
    require(isinstance(contract.get("suite"), str) and is_hash(contract.get("suite_sha256")),
            "model contract must bind an immutable suite definition")
    require(contract.get("step") == "model", "only the existing model step can be a model barrier")
    candidate = contract["candidate"]
    require(is_hash(candidate.get("client_commit"), 40) and is_hash(candidate.get("vllm_head"), 40)
            and is_hash(candidate.get("vllm_tree"), 40)
            and isinstance(candidate.get("image_id"), str)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", candidate["image_id"]), "candidate model source/image is not frozen")
    require(isinstance(candidate.get("profiles"), dict) and candidate["profiles"]
            and isinstance(candidate.get("nodes"), list) and len(candidate["nodes"]) == len(set(candidate["nodes"])) > 0,
            "candidate model profiles/rank set is missing")
    source = contract["reference_source"]
    require(is_hash(source.get("client_commit"), 40) and is_hash(source.get("vllm_head"), 40)
            and is_hash(source.get("vllm_tree"), 40)
            and re.fullmatch(r"sha256:[0-9a-f]{64}", source.get("image_id", "")),
            "saved model reference source is not bound")
    references = contract["references"]
    require(isinstance(references, list) and len(references) == 2
            and len({r["label"] for r in references}) == 2
            and len({r["path"] for r in references}) == 2, "both distinct saved reference boots are required")
    for reference in references:
        require(Path(reference["path"]).is_absolute() and is_hash(reference.get("sha256"))
                and type(reference.get("bytes")) is int and reference["bytes"] > 0,
                "saved model path/hash/size is missing")
    shape = contract["workload_shape"]
    for key in ("prompt_count", "concurrency", "max_tokens", "top"):
        require(type(shape.get(key)) is int and shape[key] > 0, "model workload shape is incomplete")
    require(type(shape.get("fixed_length")) is bool
            and type(contract.get("positions")) is int and contract["positions"] > 0
            and is_hash(contract.get("corpus_sha256")), "model score/corpus shape is missing")
    provenance_record = contract["reference_provenance"]
    require(is_hash(provenance_record.get("sha256")), "saved reference provenance hash is missing")
    provenance_path = Path(provenance_record["path"])
    if not provenance_path.is_absolute():
        require(".." not in provenance_path.parts, "reference provenance path escapes the source packet")
        provenance_path = Path(__file__).resolve().parents[1] / provenance_path
    require(provenance_path.stat().st_size <= 65536, "saved reference provenance is oversized")
    provenance_bytes = provenance_path.read_bytes()
    require(hashlib.sha256(provenance_bytes).hexdigest() == provenance_record["sha256"],
            "saved reference provenance bytes changed")
    provenance = json.loads(provenance_bytes, parse_constant=_nonfinite)
    require(provenance.get("execution_commit") == source["client_commit"]
            and provenance.get("image") == source["image_id"], "saved reference provenance source differs")
    for key in ("performance_analysis", "export_receipt"):
        require(is_hash(provenance.get(key, {}).get("sha256"))
                and isinstance(provenance[key].get("path"), str), "saved source analysis/export proof is missing")
    require(isinstance(provenance.get("references"), list) and len(provenance["references"]) == 2,
            "saved reference provenance needs exactly two boot records")
    frozen = {item["label"]: item for item in provenance["references"]}
    require(len(frozen) == 2 and set(frozen) == {reference["label"] for reference in references},
            "saved reference provenance boot labels differ")
    for reference in references:
        original = frozen[reference["label"]]
        starts = reference.get("boot_started_at", {})
        require(starts == original.get("boot_started_at") and set(starts) == set(candidate["nodes"])
                and all(isinstance(stamp, str) and stamp for stamp in starts.values()),
                "saved reference boot identities differ or omit a rank")
        require(original.get("artifact") == {"source": reference["path"], "size": reference["bytes"],
                                             "sha256": reference["sha256"]},
                "saved reference artifact differs from published boot provenance")
    require(all(references[0]["boot_started_at"][node] != references[1]["boot_started_at"][node]
                for node in candidate["nodes"]), "saved references reuse a rank boot")
    return {"path": str(path), "sha256": expected_hash, "contract": contract,
            "reference_provenance": {"path": str(provenance_path), "sha256": provenance_record["sha256"],
                "performance_analysis": provenance["performance_analysis"], "export_receipt": provenance["export_receipt"]}}


def from_environment():
    path, digest = os.environ.get(CONTRACT_ENV), os.environ.get(HASH_ENV)
    if not path and not digest:
        return None
    require(bool(path) and bool(digest), "model contract path and SHA256 must be supplied together")
    return load_binding(path, digest)


def prepare(binding, definition, suite_digest, label, metadata):
    """Load only the two saved files; do not send or repeat a model request."""
    contract = binding["contract"]
    require(definition["name"] == contract["suite"] and suite_digest == contract["suite_sha256"],
            "model barrier suite definition differs")
    require(definition["steps"][0]["id"] == contract["step"]
            and definition["steps"][0]["command"] == "model"
            and sum(step["command"] == "model" for step in definition["steps"]) == 1,
            "required existing model step must occur once before timing")
    candidate = contract["candidate"]
    require(label in candidate["profiles"], "model barrier arm label is not declared")
    identity = (metadata or {}).get("identity", {})
    require(identity.get("client_commit") == candidate["client_commit"] and identity.get("client_dirty") is False
            and not identity.get("live_problems") and identity.get("cluster_config") == candidate["profiles"][label],
            "model barrier client/profile/live identity differs")
    nodes = identity.get("nodes", {})
    require(set(nodes) == set(candidate["nodes"]), "model barrier rank coverage differs")
    for state in nodes.values():
        labels = state.get("image_labels", {})
        require(state.get("image_id") == candidate["image_id"] and state.get("checkout") == candidate["client_commit"]
                and labels.get("local.spark-ds41f.vllm.tree") == candidate["vllm_tree"]
                and labels.get("local.spark-ds41f.vllm.patch-head") == candidate["vllm_head"],
                "model barrier rank source/image differs")
    oracles = []
    for reference in contract["references"]:
        parsed = bounded_fingerprints(reference["path"], reference["sha256"], reference["bytes"])
        oracle, artifact = parsed["fingerprints"], parsed["artifact"]
        require(all(oracle["workload"].get(key) == expected for key, expected in contract["workload_shape"].items())
                and oracle["corpus_sha256"] == contract["corpus_sha256"]
                and oracle["positions"] == contract["positions"], "saved reference contradicts the declared C40 shape/corpus")
        oracles.append({"label": reference["label"], "boot_started_at": reference["boot_started_at"],
                        "artifact": artifact, "fingerprints": oracle,
                        "observed": parsed["observed"], "parser_worker": parsed["worker"]})
    require(oracles[0]["fingerprints"] == oracles[1]["fingerprints"],
            "saved model reference boots differ; no single-reference selection is allowed")
    return {"binding": {"path": binding["path"], "sha256": binding["sha256"]},
            "reference_source": contract["reference_source"],
            "reference_provenance": binding["reference_provenance"], "references": oracles}


def evaluate(path, process_exit, prepared):
    result = {"schema": SCHEMA, "contract": prepared["binding"],
              "reference_source": prepared["reference_source"], "references": prepared["references"],
              "reference_provenance": prepared["reference_provenance"],
              "process_exit": process_exit, "errors": []}
    try:
        require(type(process_exit) is int and process_exit == 0, "existing model process failed")
        parsed = bounded_fingerprints(path)
        result.update(artifact=parsed["artifact"], parser_worker=parsed["worker"], observed=parsed["observed"])
        actual = parsed["fingerprints"]
        result["fingerprints"] = actual
        for reference in prepared["references"]:
            require(actual == reference["fingerprints"], "model output/score/workload differs from saved " + reference["label"])
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        if isinstance(error, ParserFailure):
            result["parser_worker"] = error.receipt
            if error.receipt.get("artifact"):
                result["artifact"] = error.receipt["artifact"]
        if "artifact" not in result and Path(path).is_file():
            try:
                result["artifact"] = _artifact_stat(path)
            except OSError as hash_error:
                result["artifact_hash_error"] = str(hash_error)
    result["required_evidence_passed"] = not result["errors"]
    result["status"] = "requirements-passed" if not result["errors"] else "failed"
    return result


if __name__ == "__main__":
    raise SystemExit(_worker_main())
