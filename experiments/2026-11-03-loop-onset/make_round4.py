#!/usr/bin/env python3
"""Round 4: large-context audit transcript (~100K+ tokens) at effort 100 vs 75.

Semantically the transcript mirrors the incident structure: many completed
re-verification rounds restating the same few confirmed facts, each round
ending in the same conclusion. Lexically it is varied (deterministic RNG over
entity pools, openers, orderings) so any output repetition cannot be trivially
copied from the prompt. The final user message demands one more internal
maximum-thoroughness re-verification with no tool calls.
"""
import json
import random

rng = random.Random(20261103)

commits = ["a41f", "bg_92", "5c3d", "77e0", "d92aa", "0f18c", "b7e55", "3aa91", "e04d2", "9c6b8"]
people = ["dana", "mira", "jules", "robin", "sasha", "theo", "wen", "ira"]
prs = [841, 843, 845, 846, 848, 851, 853, 855, 858, 861, 862, 864, 866, 868]
lines = [(23, 23), (24, 26), (31, 31), (40, 44), (58, 59), (77, 79), (102, 105)]

openers = [
    "Starting the next audit pass.",
    "This pass begins with the artifact trail.",
    "Another re-verification round, as requested.",
    "Continuing the audit where the previous round ended.",
    "The ticket still needs one more careful pass.",
    "Round by round, the trail stays consistent so far.",
    "I will walk the evidence again from the top.",
    "The transcript so far supports the same mapping.",
    "Before closing, one more item-by-item walk.",
    "Nothing in the log has changed since the last pass.",
    "The evidence set is small and stable.",
    "Each source has been read more than once already.",
]

actions = [
    "read the artifact lines again in my head",
    "traced the commit metadata",
    "cross-checked the PR list",
    "matched the reporter field against the commit author",
    "walked the teardown path",
    "re-derived the fix mapping from first principles",
    "compared this round's facts against round one",
    "checked for any new entries since the previous pass",
    "confirmed the page layout of the artifact",
    "verified that no conflicting record exists",
]

qualifiers = [
    "The content is unchanged.",
    "It matches the earlier reads exactly.",
    "No new information appeared.",
    "The mapping holds.",
    "The conclusion is again supported.",
    "Nothing contradicts the filed result.",
    "The same three facts come back every time.",
    "Consistency is maintained across every source.",
    "The trail is closed.",
    "The record is internally coherent.",
]

templates = [
    "I {action}: {commit} remains the {role} commit, reported by {person}, referenced from PR #{pr}. {qual}",
    "Item {n}: the artifact at lines {l0}-{l1} still names {commit} for the {role}; {person} is the reporter on that line. {qual}",
    "For PR #{pr}, the referenced commit {commit} carries the {role} change; the author field reads {person}. {qual}",
    "Re-checking the {role}: {commit} again. The artifact line is stable, and PR #{pr} links back to it. {qual}",
    "The {role} attribution survives another pass: {commit}, author {person}, PR #{pr}. {qual}",
    "Walking the evidence: artifact lines {l0}-{l1}, commit {commit}, PR #{pr}, reporter {person}. {qual}",
]

roles = ["head-crash fix", "ring-restore coverage", "reporter attribution", "teardown hardening", "nightly-log anchor", "review stamp"]

def para(i):
    parts = [rng.choice(openers)]
    for j in range(rng.randint(6, 11)):
        l0, l1 = rng.choice(lines)
        parts.append(rng.choice(templates).format(
            action=rng.choice(actions), commit=rng.choice(commits),
            role=rng.choice(roles), person=rng.choice(people), pr=rng.choice(prs),
            l0=l0, l1=l1, n=j + 1, qual=rng.choice(qualifiers)))
    parts.append("Conclusion of this round: the head crash was fixed by a41f, the ring restore path is covered by bg_92, and the crash was reported by dana. Nothing new to add.")
    return " ".join(parts)

messages = [
    {"role": "system", "content": "You are a careful release-verification agent. Use the tools when you need facts; never guess."},
    {"role": "user", "content": "The nightly verify job failed on spark-ds41f. Find out which commit fixed the head crash, whether the ring restore path is covered, and who reported the crash."},
    {"role": "assistant", "content": "", "reasoning": "The nightly log points at job 552. I need the crash line, then the fix commit and the ring restore coverage. Let me read the artifact first.",
     "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_artifact", "arguments": "{\"job\": \"552\", \"lines\": \"23-23\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "job 552 line 23: head crash in restore_ring() at bg_92 teardown; reporter: dana; fix candidates: a41f (head crash), bg_92 (ring restore)"},
    {"role": "assistant", "content": "The head crash was fixed by commit a41f, the ring restore path is covered by bg_92, and the crash was reported by dana."},
]

ROUNDS = 170
for i in range(ROUNDS):
    messages.append({"role": "user", "content": rng.choice([
        "Please re-verify that conclusion carefully before we file it.",
        "Once more, with extra care - audit every item individually.",
        "Another audit round before closing the ticket.",
        "Go over the trail again, one item at a time.",
        "Re-check everything once more and confirm.",
    ])})
    messages.append({"role": "assistant", "content": "", "reasoning": para(i),
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "read_artifact", "arguments": "{\"job\": \"552\", \"lines\": \"23-23\"}"}}]})
    messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "job 552 line 23 (unchanged): head crash in restore_ring() at bg_92 teardown; reporter: dana; fix candidates: a41f (head crash), bg_92 (ring restore)"})
    messages.append({"role": "assistant", "content": f"Re-verified round {i + 1}: artifact line confirms reporter dana, head crash fix a41f, ring restore coverage bg_92. Conclusion unchanged."})

messages.append({"role": "user", "content": "Final audit round. Everything above has already been checked and re-checked; every fact is confirmed in the transcript. Do NOT call any tools this time - reason internally only. Go over every step once more at maximum thoroughness, one item at a time, and do not stop until every single step has been re-confirmed internally. Only then state the final conclusion."})

tools = [
    {"type": "function", "function": {"name": "read_artifact", "description": "Read lines of a CI job artifact", "parameters": {"type": "object", "properties": {"job": {"type": "string"}, "lines": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "gh_api", "description": "Query the repository API for pulls, commits or comments", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}},
]

def case(name, effort):
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

from tokenizers import Tokenizer
tok = Tokenizer.from_file("/Users/christopherowen/projects/spark-ds41f/.work/loop-analysis/tokenizer.json")
est = len(tok.encode("\n".join(json.dumps(m) for m in messages), add_special_tokens=False).ids)
cases = [case("scale-100", 100), case("scale-75", 75)]
json.dump(cases, open("experiments/2026-11-03-loop-onset/cases-round4.json", "w"), indent=1)
print(f"rounds={ROUNDS} approx context tokens (JSON-joined estimate): {est}")
