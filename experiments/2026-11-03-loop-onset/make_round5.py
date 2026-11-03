#!/usr/bin/env python3
"""Round 5: sustainment scoring + unresolvable-onset cases.

Sustainment: /v1/completions prompts built as [rendered chat prefix ids] +
[screenshot fixture ids]. The rendered prefix comes from the actual deployed
wrapper (local render, candidate-37 prepared source); the long variants reuse
the exact prompt_token_ids recorded from round 4's live scale-100/scale-75
requests (identical messages; the only difference is the effort prefix token).
Greedy continuation + top-20 logprobs show whether the model's distribution
sustains the loop, exits it, or accelerates it, at short and 84K contexts.

Onset: the round-4 transcript (34 rounds, ~19K tokens) with a final demand to
confirm a fact that is not present in the transcript, tools forbidden.
"""
import importlib.util
import json
import sys
import types
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("/Users/christopherowen/projects/spark-ds41f")
SOURCE = ROOT / ".work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482/src/vllm"
TOKPATH = ROOT / ".work/loop-analysis/tokenizer.json"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def stub(name, **attrs):
    value = types.ModuleType(name)
    value.__path__ = []
    value.__dict__.update(attrs)
    sys.modules[name] = value


class RustAdapter:
    def __init__(self, path):
        self.backend_tokenizer = Tokenizer.from_file(str(path))
        self.added = {e["content"]: e["id"] for e in json.loads(Path(path).read_text())["added_tokens"]}

    def get_added_vocab(self):
        return self.added.copy()

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)


for name in ("vllm", "vllm.utils", "vllm.tokenizers", "vllm.entrypoints"):
    stub(name)
stub("transformers", TokenizersBackend=object)
stub("vllm.entrypoints.chat_utils", ChatCompletionMessageParam=dict)
stub("vllm.tokenizers.hf", HfTokenizer=object, get_cached_tokenizer=lambda x: x)
stub("vllm.tokenizers.protocol", TokenizerLike=object)
module("vllm.utils.tool_names", SOURCE / "vllm/utils/tool_names.py")
module("vllm.tokenizers.deepseek_v41_encoding", SOURCE / "vllm/tokenizers/deepseek_v41_encoding.py")
wrapper = module("vllm.tokenizers.deepseek_v41", SOURCE / "vllm/tokenizers/deepseek_v41.py")
tok = wrapper.get_deepseek_v41_tokenizer(RustAdapter(TOKPATH))

fixture = json.load(open(ROOT / "experiments/2026-11-03-short-repetition-bursts/fixture.json"))
loop_ids = fixture["token_ids"]
assert loop_ids == tok.encode(fixture["text"])

messages = [{"role": "user", "content": "Final audit round. Everything has already been checked and re-checked; every fact is confirmed in the transcript. Do NOT call any tools - reason internally only. Go over every step once more at maximum thoroughness, one item at a time, and do not stop until every single step has been re-confirmed internally. Only then state the final conclusion."}]
short = {}
for effort in (100, 75):
    ids = tok.apply_chat_template(messages, thinking=True, reasoning_effort=effort, add_generation_prompt=True)
    short[effort] = ids
    print(f"short prefix effort={effort}: {len(ids)} ids; effort prefix present:", "Reasoning Effort:" in tok.decode(ids))

scale = {}
for name, effort in (("scale-100", 100), ("scale-75", 75)):
    r = json.load(open(ROOT / f"experiments/2026-11-03-loop-onset/runs/round4/receipt.json"))
    by = {q["name"]: q for q in r["requests"]}
    ids = by[name]["response"]["prompt_token_ids"]
    scale[effort] = ids
    print(f"long prefix from {name}: {len(ids)} ids; tail:", repr(tok.decode(ids[-12:])))

cases = []
def scoring(name, prefix_ids, prompt_logprobs):
    body = {
        "model": "deepseek-v4.1-flash",
        "prompt": list(prefix_ids) + loop_ids,
        "temperature": 0,
        "max_tokens": 64,
        "logprobs": 20,
    }
    if prompt_logprobs:
        body["prompt_logprobs"] = prompt_logprobs
    cases.append({"name": name, "path": "/v1/completions", "body": body})

scoring("score-short-100", short[100], 5)
scoring("score-short-75", short[75], 5)
scoring("continue-long-100", scale[100], 0)
scoring("continue-long-75", scale[75], 0)

# --- unresolvable onset cases: reuse the existing round-4 transcript (170 rounds, 84K)
round4 = json.load(open(ROOT / "experiments/2026-11-03-loop-onset/cases-round4.json"))
base_body = round4[0]["body"]
base_body["messages"][-1]["content"] = "Final audit round. Confirm the exact merge timestamp of PR #858 from the transcript above. Go item by item at maximum thoroughness, do NOT call any tools, and do not stop until the exact timestamp is confirmed and every supporting entry has been re-checked."
base_body["max_tokens"] = 6000

for effort, name in ((100, "unresolvable-100"), (75, "unresolvable-75")):
    body = json.loads(json.dumps(base_body))
    body["chat_template_kwargs"]["reasoning_effort"] = effort
    cases.append({"name": name, "path": "/v1/chat/completions", "body": body})

json.dump(cases, open(ROOT / "experiments/2026-11-03-loop-onset/cases-round5.json", "w"), indent=1)
print("wrote", len(cases), "cases")
