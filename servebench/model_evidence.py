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

from . import behaviour

CONTRACT_ENV = "SPARK_DS41F_SUITE_MODEL_CONTRACT"
HASH_ENV = CONTRACT_ENV + "_SHA256"
SCHEMA = "servebench-model-evidence/1"
WORKLOAD_KEYS = (*behaviour.WORKLOAD_KEYS, "extra_body")
SCORE_KEYS = ("id", "tokens", "logprob", "rank", "top_ids", "top_logprobs")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_hash(value, length=64):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{" + str(length) + r"}", value) is not None


def _nonfinite(value):
    raise ValueError("non-finite model evidence: " + value)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class _HashReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()
        self.size = 0

    def readable(self):
        return True

    def readinto(self, target):
        count = self.raw.readinto(target)
        if count:
            self.digest.update(memoryview(target)[:count])
            self.size += count
        return count


def read_model(path, expected_hash=None, expected_bytes=None):
    """Hash the exact encoded bytes while parsing; no second or unverified read."""
    path = Path(path)
    with path.open("rb") as raw:
        reader = _HashReader(raw)
        with io.BufferedReader(reader) as buffered:
            decoded = gzip.GzipFile(fileobj=buffered) if path.suffix == ".gz" else buffered
            with io.TextIOWrapper(decoded, encoding="utf-8") as stream:
                value = json.load(stream, parse_constant=_nonfinite)
        digest, size = reader.digest.hexdigest(), reader.size
    if expected_hash is not None:
        require(digest == expected_hash, "saved model artifact hash differs: " + str(path))
    if expected_bytes is not None:
        require(size == expected_bytes, "saved model artifact size differs: " + str(path))
    return value, {"path": str(path), "sha256": digest, "bytes": size}


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
    finite = lambda x: type(x) in (int, float) and math.isfinite(x)
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
        require(all(x is None or type(x) is int for x in row["rank"]), "model token rank is invalid")
        for top_ids, top_lps in zip(row["top_ids"], row["top_logprobs"]):
            require(tokens(top_ids) and len(set(top_ids)) == len(top_ids)
                    and isinstance(top_lps, list) and len(top_ids) == len(top_lps) <= workload["top"]
                    and all(finite(x) for x in top_lps), "model top-logprob positions are incomplete or non-finite")
        positions += len(ids)
    generation_hash = behaviour.corpus_digest(sequences)
    require(value["corpus"]["sha256"] == generation_hash and value["corpus"]["sequences"] == count,
            "model corpus hash/count contradicts the actual generated/scored tokens")
    score_payload = [{key: row[key] for key in SCORE_KEYS} for row in sorted(scores, key=lambda row: row["id"])]
    return {"generation_sha256": generation_hash,
            "scores_sha256": hashlib.sha256(_canonical(score_payload)).hexdigest(),
            "corpus_sha256": value["corpus"]["sha256"], "sequences": count, "positions": positions,
            "workload": {key: workload.get(key) for key in WORKLOAD_KEYS}}


def load_binding(path, expected_hash):
    require(is_hash(expected_hash), "an exact published model-contract SHA256 is required")
    path = Path(path).resolve()
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
    return {"path": str(path), "sha256": expected_hash, "contract": contract}


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
        value, artifact = read_model(reference["path"], reference["sha256"], reference["bytes"])
        oracle = fingerprints(value)
        require(all(oracle["workload"].get(key) == expected for key, expected in contract["workload_shape"].items())
                and oracle["corpus_sha256"] == contract["corpus_sha256"]
                and oracle["positions"] == contract["positions"], "saved reference contradicts the declared C40 shape/corpus")
        oracles.append({"label": reference["label"], "artifact": artifact, "fingerprints": oracle})
        del value
    require(oracles[0]["fingerprints"] == oracles[1]["fingerprints"],
            "saved model reference boots differ; no single-reference selection is allowed")
    return {"binding": {"path": binding["path"], "sha256": binding["sha256"]},
            "reference_source": contract["reference_source"], "references": oracles}


def evaluate(path, process_exit, prepared):
    result = {"schema": SCHEMA, "contract": prepared["binding"],
              "reference_source": prepared["reference_source"], "references": prepared["references"],
              "process_exit": process_exit, "errors": []}
    try:
        require(type(process_exit) is int and process_exit == 0, "existing model process failed")
        value, artifact = read_model(path)
        result["artifact"] = artifact
        actual = fingerprints(value)
        result["fingerprints"] = actual
        for reference in prepared["references"]:
            require(actual == reference["fingerprints"], "model output/score/workload differs from saved " + reference["label"])
    except (OSError, ValueError, TypeError, AttributeError, KeyError, OverflowError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        if "artifact" not in result and Path(path).is_file():
            try:
                digest, size = hashlib.sha256(), 0
                with Path(path).open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                result["artifact"] = {"path": str(path), "sha256": digest.hexdigest(), "bytes": size}
            except OSError as hash_error:
                result["artifact_hash_error"] = str(hash_error)
    result["required_evidence_passed"] = not result["errors"]
    result["status"] = "requirements-passed" if not result["errors"] else "failed"
    return result
