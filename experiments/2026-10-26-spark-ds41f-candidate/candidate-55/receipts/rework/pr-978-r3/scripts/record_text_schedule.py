"""Records the pinned values of tests/v1/core/test_text_request_scheduling.py on the
mounted tree (run with pytest -s on candidate-54 as built plus the new test file)."""
import hashlib
import json

from tests.v1.core.test_text_request_scheduling import run_workload


def test_record():
    steps, admitted = run_workload()
    digest = hashlib.sha256(json.dumps(steps).encode()).hexdigest()
    print("RECORD", json.dumps({"admitted": admitted, "sha256": digest, "steps": len(steps)}))
