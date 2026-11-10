#!/usr/bin/env python3
"""The temperature-0 determinism gate, unchanged, with every response kept.

usage: token_determinism_record.py BASE_URL --record FILE [the gate's arguments]

Runs experiments/2026-10-12-r6g/token_determinism_text.py's main() as it is: the same
requests, in the same order, with the same comparisons and the same JSON lines on
stdout (lab.py's tables read them). It only wraps the gate's Client.complete and
compare to keep, in FILE, every response (tokens, logprobs and text, in request
order) and, for each request that differs from its reference, both whole responses.
A difference is then readable to its token: where the two outputs part, and with
logprobs, by how much.
"""
import importlib.util
import json
import sys
import threading
from pathlib import Path

GATE = Path(__file__).resolve().parents[2] / "2026-10-12-r6g" / "token_determinism_text.py"


def main() -> None:
    argv = sys.argv[1:]
    if "--record" not in argv:
        raise SystemExit("usage: token_determinism_record.py BASE_URL --record FILE [gate arguments]")
    at = argv.index("--record")
    record = Path(argv[at + 1])
    del argv[at:at + 2]
    sys.argv = [str(GATE), *argv]

    spec = importlib.util.spec_from_file_location("token_determinism_text", GATE)
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)

    lock = threading.Lock()
    calls, differing = [], []
    complete, compare = gate.Client.complete, gate.compare

    def recorded_complete(self, prompt, tokens, salt=None):
        result = complete(self, prompt, tokens, salt)
        with lock:
            calls.append({"index": len(calls), "prompt": prompt[:96], "max_tokens": tokens,
                          "shared_salt": salt is not None, "result": result})
        return result

    def recorded_compare(result, reference):
        row = compare(result, reference)
        if (row.get("first_token_diff") is not None or row.get("first_char_diff") is not None
                or row.get("logprobs_equal") is False):
            differing.append({"row": row, "result": result, "reference": reference})
        return row

    gate.Client.complete = recorded_complete
    gate.compare = recorded_compare
    try:
        gate.main()
    finally:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"gate": str(GATE.relative_to(GATE.parents[2])), "argv": argv,
                                      "calls": calls, "differing": differing}))


if __name__ == "__main__":
    main()
