"""Require the corrected C45 IKET case and complete native guard evidence."""
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

EXPECTED = {
    "tests/kernels/test_deepseek_v41_tilelang_iket.py": 17,
    "tests/v1/worker/test_gpu_tilelang_logprob_softmax.py": 419,
    "tests/v1/worker/test_gpu_verified_logprob_token_ids.py": 29,
}
INVARIANT = "test_on_matches_off_and_keeps_its_own_cache"


def conditions_pass(conditions, nodes, memory_floor):
    """Require complete guard evidence, retaining all original adverse fields."""
    try:
        wanted = set(nodes)
        if (conditions["halted"] or conditions["fan_stop_failures"]
                or any(set(conditions[key]) != wanted for key in ("thermal", "memory"))
                or {r["node"] for r in conditions["cooling"]} != wanted
                or any(r["final_c"] >= 55 for r in conditions["cooling"])
                or {r["node"] for r in conditions["fan_curve"]} != wanted):
            return False
        for node in wanted:
            thermal, memory = conditions["thermal"][node], conditions["memory"][node]
            values = [thermal[key] for key in ("max_gpu_c", "min_headroom_c", "max_system_c",
                                              "max_power_w", "thermal_slowdown_ms", "power_cap_ms")]
            values += [memory[key] for key in ("min_available_gib", "swap_growth_gib")]
            if not all(type(value) in (int, float) and math.isfinite(value) for value in values):
                return False
            if (memory["min_available_gib"] < memory_floor or memory["swap_growth_gib"] != 0
                    or thermal["thermal_slowdown_ms"] != 0 or thermal["power_cap_ms"] < 0):
                return False
        return True
    except (KeyError, TypeError):
        return False


def audit(directory):
    directory = Path(directory)
    data = (directory / "results.jsonl").read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    result = {"passed": False, "expected_cases": EXPECTED, "critical_files": {},
              "invariant": INVARIANT, "invariant_passed": False,
              "results_sha256": hashlib.sha256(data).hexdigest(), "xml": {}}
    for name, count in EXPECTED.items():
        matches = [row for row in rows if row.get("file") == name]
        passed = (len(matches) == 1 and matches[0].get("status") == "ok"
                  and matches[0].get("passed") == count and all(matches[0].get(k) == 0
                      for k in ("failed", "error", "skipped")) and not matches[0].get("failures"))
        result["critical_files"][name] = {"passed": passed, "outcomes": matches}
    cases = []
    for path in sorted(directory.rglob("*.xml")):
        blob = path.read_bytes()
        result["xml"][str(path.relative_to(directory))] = hashlib.sha256(blob).hexdigest()
        for case in ET.fromstring(blob).iter("testcase"):
            if (case.get("name") == INVARIANT
                    and case.get("classname", "").split(".")[-1] == "test_deepseek_v41_tilelang_iket"):
                cases.append(case)
    result["invariant_occurrences"] = len(cases)
    result["invariant_passed"] = (len(cases) == 1 and not any(
        child.tag in ("failure", "error", "skipped") for child in cases[0]))
    result["passed"] = result["invariant_passed"] and all(v["passed"] for v in result["critical_files"].values())
    return result
