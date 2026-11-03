#!/usr/bin/env python3
"""Chained real-restatement onset probe.

Runs repeated final-audit rounds at effort 100 (greedy, detector off). Each
round's actual generated reasoning is appended as a completed assistant turn
(tools stay defined so the encoder retains it), then the next round demands
one more maximum-thoroughness internal re-verification. After each round the
output is scanned for short-motif bursts; the chain stops early if
degeneration appears. Uses live_probe's remote helpers unchanged (hold,
traffic guard, receipts).
"""
import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "live_probe", "/Users/christopherowen/projects/spark-ds41f/experiments/2026-11-03-loop-cause/live_probe.py")
lp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lp)

sys.path.insert(0, "/Users/christopherowen/projects/spark-ds41f/experiments/2026-11-03-loop-onset")
from analyze_round import motif_runs, bursts  # noqa: E402

OUT = Path("/Users/christopherowen/projects/spark-ds41f/experiments/2026-11-03-loop-onset/runs/round7")
OUT.mkdir(parents=True, exist_ok=True)

base_messages = [
    {"role": "system", "content": "You are a careful release-verification agent. Use the tools when you need facts; never guess."},
    {"role": "user", "content": "The nightly verify job failed on spark-ds41f. Find out which commit fixed the head crash, whether the ring restore path is covered, and who reported the crash."},
    {"role": "assistant", "content": "", "reasoning": "The nightly log points at job 552. I need the crash line, then the fix commit and the ring restore coverage. Let me read the artifact first.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_artifact", "arguments": "{\"job\": \"552\", \"lines\": \"23-23\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "job 552 line 23: head crash in restore_ring() at bg_92 teardown; reporter: dana; fix candidates: a41f (head crash), bg_92 (ring restore)"},
    {"role": "assistant", "content": "The head crash was fixed by commit a41f, the ring restore path is covered by bg_92, and the crash was reported by dana."},
]

tools = [
    {"type": "function", "function": {"name": "read_artifact", "description": "Read lines of a CI job artifact", "parameters": {"type": "object", "properties": {"job": {"type": "string"}, "lines": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "gh_api", "description": "Query the repository API for pulls, commits or comments", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
]

ask_variants = [
    "Final audit round. Everything above has already been checked and re-checked; every fact is confirmed in the transcript. Do NOT call any tools this time - reason internally only. Go over every step once more at maximum thoroughness, one item at a time, and do not stop until every single step has been re-confirmed internally. Only then state the final conclusion.",
    "Another final audit round, same rules: no tools, internal reasoning only, maximum thoroughness, every step one at a time, no stopping until everything is re-confirmed again.",
    "And one more: same final-audit rules - no tools, internally, maximum thoroughness, item by item, do not stop until all steps are re-confirmed yet again.",
    "Keep going: final-audit rules unchanged - no tools, internal only, maximum thoroughness, one item at a time, no stopping before full re-confirmation.",
    "Once more, exactly as before: no tools, internal, maximum thoroughness, item by item, full re-confirmation before concluding.",
    "Final round: no tools, internal only, maximum thoroughness, every item, no stopping until everything is re-confirmed once more.",
]

ROUNDS = 6
holder = "loop-onset-chain-" + __import__("uuid").uuid4().hex[:12]
receipt = {"holder": holder, "rounds": [], "released": None}
acquired = False
messages = list(base_messages)
try:
    receipt["identities"] = {h: lp.remote(h, {"action": "identity"}) for h in ("dgx1", "dgx2", "dgx3")}
    assert len({x["image_id"] for x in receipt["identities"].values()}) == 1
    receipt["hold"] = lp.remote("dgx1", {"action": "acquire", "holder": holder})
    acquired = True
    for i in range(ROUNDS):
        round_messages = messages + [{"role": "user", "content": ask_variants[i % len(ask_variants)]}]
        body = {
            "model": "deepseek-v4.1-flash",
            "messages": round_messages,
            "tools": tools,
            "temperature": 0,
            "max_tokens": 6000,
            "return_token_ids": True,
            "logprobs": True,
            "top_logprobs": 5,
            "repetition_detection": {"max_pattern_size": 0, "min_pattern_size": 0, "min_count": 0},
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": 100},
        }
        record = {"round": i + 1, "prompt_tokens": None}
        receipt["rounds"].append(record)
        result = lp.remote("dgx1", {"action": "request", "holder": holder,
                                    "path": "/v1/chat/completions", "body": body})
        record.update(result)
        (OUT / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
        choice = result["response"]["choices"][0]
        ids = choice["token_ids"]
        msg = choice["message"]
        reasoning = msg.get("reasoning") or ""
        answer = msg.get("content") or ""
        fam = bursts(ids, 2)
        worst = None
        for size in (1, 2, 3, 4):
            for pos, count, motif in motif_runs(ids, size):
                if worst is None or count * size > worst[0]:
                    worst = (count * size, size, pos, count, motif)
        degenerate = bool(fam) or (worst and worst[3] >= 6)
        print(json.dumps({
            "round": i + 1, "finish": choice["finish_reason"], "output_tokens": len(ids),
            "reasoning_chars": len(reasoning), "answer_chars": len(answer),
            "burst_families": len(fam), "longest_run": worst[3] if worst else 0,
            "degenerate": degenerate,
        }), flush=True)
        if choice["finish_reason"] not in ("stop", "length"):
            print("stopping chain: unexpected finish", choice["finish_reason"])
            break
        messages = messages + [
            {"role": "user", "content": ask_variants[i % len(ask_variants)]},
            {"role": "assistant", "content": answer, "reasoning": reasoning},
        ]
        if degenerate:
            print("stopping chain: degeneration detected")
            break
finally:
    if acquired:
        receipt["released"] = lp.remote("dgx1", {"action": "release", "holder": holder})
    (OUT / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
