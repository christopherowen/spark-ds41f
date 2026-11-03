#!/usr/bin/env python3
"""Round 3 cases: retained-reasoning feedback at effort 100 vs 75.

Reuses the actual round-2 assistant output (4,497-token re-verification reasoning)
as retained prior reasoning in a longer transcript, then demands one more
maximum-thoroughness internal re-verification. The seeded variant additionally
retains the screenshot fixture's degenerate reasoning from a prior session.
"""
import json

ROUND2 = json.load(open("experiments/2026-11-03-loop-onset/runs/round2/receipt.json"))
by_name = {r["name"]: r for r in ROUND2["requests"]}
prior = by_name["exhausted-100-notools"]["response"]["choices"][0]["message"]
prior_reasoning = prior.get("reasoning") or ""
prior_answer = prior.get("content") or ""
assert len(prior_reasoning) > 10000, len(prior_reasoning)

fixture = json.load(open("experiments/2026-11-03-short-repetition-bursts/fixture.json"))
loop_text = fixture["text"]

base_messages = [
    {"role": "system", "content": "You are a careful release-verification agent. Use the tools when you need facts; never guess."},
    {"role": "user", "content": "The nightly verify job failed on spark-ds41f. Find out which commit fixed the head crash, whether the ring restore path is covered, and who reported the crash."},
    {"role": "assistant", "content": "", "reasoning": "The nightly log points at job 552. I need the crash line, then the fix commit and the ring restore coverage. Let me read the artifact first.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_artifact", "arguments": "{\"job\": \"552\", \"lines\": \"23-23\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "job 552 line 23: head crash in restore_ring() at bg_92 teardown; reporter: dana; fix candidates: a41f (head crash), bg_92 (ring restore)"},
    {"role": "assistant", "content": "The head crash was fixed by commit a41f, the ring restore path is covered by bg_92, and the crash was reported by dana."},
    {"role": "user", "content": "Please re-verify that conclusion carefully before we file it."},
    {"role": "assistant", "content": "", "reasoning": "Re-verification round 1: check the artifact line, then each commit mapping, then the reporter. Reading the artifact line again first.",
     "tool_calls": [{"id": "call_2", "type": "function", "function": {"name": "read_artifact", "arguments": "{\"job\": \"552\", \"lines\": \"23-23\"}"}}]},
    {"role": "tool", "tool_call_id": "call_2", "content": "job 552 line 23 (unchanged): head crash in restore_ring() at bg_92 teardown; reporter: dana; fix candidates: a41f (head crash), bg_92 (ring restore)"},
    {"role": "assistant", "content": "Re-verified round 1: the artifact line confirms reporter dana, head crash fix a41f, ring restore coverage bg_92. Conclusion unchanged."},
    {"role": "user", "content": "Once more, with extra care - audit every item individually."},
    {"role": "assistant", "content": "", "reasoning": "Re-verification round 2, item by item: item 1 the artifact line, item 2 the a41f mapping, item 3 the bg_92 coverage, item 4 the reporter. Checking each in turn.",
     "tool_calls": [{"id": "call_3", "type": "function", "function": {"name": "gh_api", "arguments": "{\"path\": \"commits/a41f\"}"}}]},
    {"role": "tool", "tool_call_id": "call_3", "content": "commit a41f: 'Fix head crash in restore_ring teardown'; author: dana; reviewed-by: mira; refs job 552"},
    {"role": "assistant", "content": "Re-verified round 2 item 1: a41f fixes the head crash, authored by dana. Items 2-4 remain as confirmed in round 1. Conclusion unchanged: a41f head crash fix, bg_92 ring restore coverage, reporter dana."},
    # yesterday's session: either absent or the degenerate screenshot reasoning
    {"role": "user", "content": "Continue the audit of the same nightly failure."},
    {"role": "assistant", "content": "", "reasoning": prior_reasoning, "tool_calls": []},
    {"role": "tool", "tool_call_id": "none", "content": "(no tool calls in that round)"},
    {"role": "assistant", "content": prior_answer},
]

seed_exchange = [
    {"role": "user", "content": "What did the earlier audit session conclude? Summarize its trail."},
    {"role": "assistant", "content": "", "reasoning": loop_text,
     "tool_calls": [{"id": "call_s", "type": "function", "function": {"name": "gh_api", "arguments": "{\"path\": \"pulls?state=all\"}"}}]},
    {"role": "tool", "tool_call_id": "call_s", "content": "pulls list unchanged: #858 merged, #861 open (device-side fix), #862 open (comments)"},
    {"role": "assistant", "content": "The earlier session's trail is summarized above in its reasoning; its conclusion matched ours."},
]

final_user = {"role": "user", "content": "Final audit round for today. Everything above has already been checked and re-checked; every fact is confirmed in the transcript. Do NOT call any tools this time - reason internally only. Go over every step once more at maximum thoroughness, one item at a time, and do not stop until every single step has been re-confirmed internally. Only then state the final conclusion."}

tools = [
    {"type": "function", "function": {"name": "read_artifact", "description": "Read lines of a CI job artifact", "parameters": {"type": "object", "properties": {"job": {"type": "string"}, "lines": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "gh_api", "description": "Query the repository API for pulls, commits or comments", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
]

def case(name, effort, seeded):
    messages = list(base_messages)
    if seeded:
        messages = messages[:-2] + seed_exchange + messages[-2:]
    messages = messages + [final_user]
    return {
        "name": name,
        "path": "/v1/chat/completions",
        "body": {
            "model": "deepseek-v4.1-flash",
            "messages": messages,
            "tools": tools,
            "temperature": 0,
            "max_tokens": 6000,
            "return_token_ids": True,
            "logprobs": True,
            "top_logprobs": 5,
            "repetition_detection": {"max_pattern_size": 0, "min_pattern_size": 0, "min_count": 0},
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": effort},
        },
    }

cases = [
    case("iter1-100", 100, False),
    case("iter1-75", 75, False),
    case("iter1-100-seeded", 100, True),
]
json.dump(cases, open("experiments/2026-11-03-loop-onset/cases-round3.json", "w"), indent=1)
print("prior reasoning chars:", len(prior_reasoning), "| prior answer chars:", len(prior_answer))
print("wrote", len(cases), "cases")
