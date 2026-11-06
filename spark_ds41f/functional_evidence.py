"""Explicit requirements for observational functional collectors.

A successful process is not a successful verdict. Conversely, an observation is
not a requirement unless the experiment declares it as one. This module does not
assign new semantics to historical collector output.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

SAMPLED_COLLECTOR = "experiments/2026-10-24-tilelang-completion/sampled_digest.py"


def validate_contract(contract: dict) -> None:
    if not isinstance(contract, dict):
        raise ValueError("evidence contract must be an object")
    kind = contract.get("kind")
    if kind == "summary":
        allowed = {"kind", "label", "require", "observe"}
        required, observed = contract.get("require", {}), contract.get("observe", [])
        if not isinstance(required, dict) or not isinstance(observed, list):
            raise ValueError("summary require/observe must be an object/list")
        if not required and not observed:
            raise ValueError("summary contract declares no requirements or observations")
        paths = list(required) + observed
        if any(not isinstance(p, str) or not p or any(not part for part in p.split(".")) for p in paths):
            raise ValueError("invalid evidence field path")
        if len(paths) != len(set(paths)):
            raise ValueError("duplicate required/observed evidence field")
        if "label" in contract and not isinstance(contract["label"], str):
            raise ValueError("summary label must be a string")
        json.dumps(required, allow_nan=False)
    elif kind == "sampled":
        allowed = {"kind", "batch_invariance", "isolated_repeat", "tokens", "seed", "cases", "record_requests"}
        for field in ("batch_invariance", "isolated_repeat"):
            if contract.get(field) not in ("require", "observe"):
                raise ValueError(f"sampled {field} must explicitly be require or observe")
        for field in ("tokens", "seed", "cases"):
            if type(contract.get(field)) is not int or contract[field] < (0 if field == "seed" else 1):
                raise ValueError(f"sampled {field} must declare a valid integer")
        if "record_requests" in contract and type(contract["record_requests"]) is not bool:
            raise ValueError("record_requests must be boolean")
    else:
        raise ValueError(f"unknown evidence kind {kind!r}")
    if set(contract) - allowed:
        raise ValueError(f"unknown evidence fields: {sorted(set(contract) - allowed)}")


def contracts_for_extras(job: dict, extras: list) -> dict:
    contracts = job.get("evidence", {})
    if not isinstance(contracts, dict):
        raise ValueError("job evidence must map extra stems to contracts")
    stems = [item[2] for item in extras]
    if set(contracts) - set(stems):
        raise ValueError("evidence names an extra that is not in this job")
    for stem, contract in contracts.items():
        if stems.count(stem) != 1:
            raise ValueError(f"evidence stem {stem!r} is not unique")
        validate_contract(contract)
        script, args, _ = next(item for item in extras if item[2] == stem)
        if contract.get("record_requests"):
            if script != SAMPLED_COLLECTOR:
                raise ValueError("request recording is supported only by the sampled collector")
            if any(str(arg).split("=", 1)[0] == "--record-dir" for arg in args):
                raise ValueError("the lab supplies the sampled request record directory")
    return contracts


def _field(value: dict, path: str):
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing evidence field {path}")
        value = value[part]
    return value


def _records(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line, parse_constant=_nonfinite)
        except ValueError:
            if line.startswith(("{", "[")):
                raise ValueError("malformed JSON evidence") from None
            continue  # Collectors can also emit human-readable diagnostics.
        if not isinstance(value, dict):
            raise ValueError("JSON evidence record must be an object")
        records.append(value)
    return records


def _nonfinite(value):
    raise ValueError(f"non-finite JSON value {value}")


def _same(actual, expected) -> bool:
    # In particular, JSON true must not match the number 1.
    return json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)


def _sampled(records: list[dict], summary: dict, contract: dict, result: dict) -> None:
    if summary.get("summary") != "seeded temperature-1 digest":
        raise ValueError("unexpected sampled summary")
    if type(summary.get("identical")) is not bool:
        raise ValueError("sampled identical verdict is absent or not boolean")
    for field in ("tokens", "seed"):
        if not _same(summary.get(field), contract[field]):
            raise ValueError(f"sampled {field} differs from the declared workload")
    scenarios = summary.get("scenarios")
    if not isinstance(scenarios, dict) or set(scenarios) != {"c8", "again"}:
        raise ValueError("sampled scenario coverage is incomplete")
    rows = [row for row in records if "scenario" in row]
    if len(rows) != 2 or {row["scenario"] for row in rows} != {"c8", "again"}:
        raise ValueError("sampled scenario rows are missing or duplicated")
    reference = summary.get("reference_sha256", "")
    if not isinstance(reference, str) or not re.fullmatch(r"[0-9a-f]{64}", reference):
        raise ValueError("sampled reference digest is absent or malformed")
    differs = {}
    for row in rows:
        name = row["scenario"]
        indexes = row.get("differing")
        if (not isinstance(indexes, list) or any(type(i) is not int or not 0 <= i < contract["cases"] for i in indexes)
                or len(indexes) != len(set(indexes)) or scenarios[name].get("differing") != indexes):
            raise ValueError(f"sampled {name} cases are malformed or contradict the summary")
        digest = row.get("sha256", "")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"sampled {name} digest is absent or malformed")
        if (digest == reference) != (not indexes):
            raise ValueError(f"sampled {name} digest contradicts its differing cases")
        differs[name] = indexes
    if summary["identical"] != (not any(differs.values())):
        raise ValueError("sampled identical verdict contradicts its scenarios")
    for field, scenario in (("batch_invariance", "c8"), ("isolated_repeat", "again")):
        value = {"matches": not differs[scenario], "differing": differs[scenario]}
        if contract[field] == "require":
            result["required"][field] = value
            if not value["matches"]:
                result["errors"].append(f"required sampled {field} failed")
        else:
            result["observed"][field] = value


def _request_records(path: Path, records: list[dict], summary: dict, contract: dict) -> dict:
    directory = path.with_suffix(".requests").resolve()
    declared = summary.get("request_evidence")
    if not isinstance(declared, dict) or declared.get("directory") != str(directory):
        raise ValueError("required request recording is missing or has the wrong directory")
    data = (directory / "manifest.json").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if declared.get("manifest_sha256") != digest or declared.get("requests") != 3 * contract["cases"]:
        raise ValueError("request manifest identity/count differs")
    manifest = json.loads(data)
    for field in ("seed", "tokens", "cases"):
        if not _same(manifest.get(field), contract[field]):
            raise ValueError(f"request manifest {field} differs")
    expected = {(scenario, case) for scenario in ("alone", "c8", "again") for case in range(contract["cases"])}
    seen, outputs = set(), {scenario: {} for scenario in ("alone", "c8", "again")}
    for entry in manifest["requests"]:
        key = (entry["scenario"], entry["case"])
        if key not in expected or key in seen or entry["file"] != f"{key[0]}-{key[1]}.json":
            raise ValueError("request manifest case is unexpected or repeated")
        seen.add(key)
        raw = (directory / entry["file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("request evidence hash differs")
        value = json.loads(raw, parse_constant=_nonfinite)
        if (value.get("scenario"), value.get("case")) != key or value.get("error") or not value.get("ended"):
            raise ValueError("request evidence is incomplete or failed")
        for field in ("client_request_id", "response_id"):
            if not isinstance(value.get(field), str) or not value[field]:
                raise ValueError(f"request evidence lacks {field}")
        controls = value["request"]
        required = {"seed": contract["seed"] + key[1], "max_tokens": contract["tokens"],
                    "temperature": 1.0, "ignore_eos": True, "logprobs": True,
                    "chat_template_kwargs": {"thinking": False}, "return_token_ids": True}
        if any(not _same(controls.get(field), wanted) for field, wanted in required.items()):
            raise ValueError("recorded generation controls differ")
        result = value["result"]
        if any(not isinstance(result.get(field), list) or len(result[field]) != contract["tokens"]
               for field in ("tokens", "logprobs")):
            raise ValueError("recorded token/logprob arrays are incomplete")
        outputs[key[0]][key[1]] = [result["tokens"], result["logprobs"]]
    if seen != expected:
        raise ValueError("request recording omits a case")
    digests = {scenario: hashlib.sha256(json.dumps([values[i] for i in range(contract["cases"])]).encode()).hexdigest()
               for scenario, values in outputs.items()}
    if digests["alone"] != summary["reference_sha256"] or any(
            digests[row["scenario"]] != row["sha256"] for row in records if "scenario" in row):
        raise ValueError("request arrays disagree with the collector digests")
    return {"directory": str(directory), "manifest_sha256": digest, "requests": len(seen),
            "native_logit_positions_available": False, "local_speculative_steps_available": False}


def evaluate(path: Path, contract: dict) -> dict:
    """Return a receipt that distinguishes required verdicts from observations."""
    result = {"source": str(path), "contract": contract, "required": {}, "observed": {}, "errors": []}
    try:
        validate_contract(contract)
        data = path.read_bytes()
        result["source_sha256"] = hashlib.sha256(data).hexdigest()
        records = _records(data.decode())
        summaries = [row for row in records if "summary" in row]
        if len(summaries) != 1:
            raise ValueError("missing or ambiguous evidence summary")
        summary = summaries[0]
        if contract["kind"] == "sampled":
            _sampled(records, summary, contract, result)
            if contract.get("record_requests"):
                result["request_records"] = _request_records(path, records, summary, contract)
        else:
            if "label" in contract and summary["summary"] != contract["label"]:
                raise ValueError("evidence summary label differs")
            for field, expected in contract.get("require", {}).items():
                actual = _field(summary, field)
                result["required"][field] = {"expected": expected, "actual": actual, "matches": _same(actual, expected)}
                if not result["required"][field]["matches"]:
                    result["errors"].append(f"required field {field} failed")
            for field in contract.get("observe", []):
                result["observed"][field] = _field(summary, field)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    result["required_evidence_passed"] = not result["errors"]
    result["status"] = "failed" if result["errors"] else "requirements-passed" if result["required"] else "observed"
    return result
