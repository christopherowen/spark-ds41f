#!/usr/bin/env python3
"""Verify the reasoning-effort ceiling against the actual patched wrapper.

Loads vllm/tokenizers/deepseek_v41.py from a prepared source tree through the
same lightweight stub adapter used by the 2026-11-03 loop-cause probes (no
torch, no server import), renders the effort prefix for every accepted effort
spelling, and compares against the expected budget table.

--baseline expects the pre-patch behavior (values above 85 pass through);
without it the ceiling table is enforced. Exit code 1 on any mismatch.
"""
import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path("/Users/christopherowen/projects/spark-ds41f")
DEFAULT_SOURCE = ROOT / ".work/build/vllm-19f2c20ed4d6-b12x-236ddff04a5f-9b160482/src/vllm"


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
        self.added = {
            e["content"]: e["id"]
            for e in json.loads(Path(path).read_text())["added_tokens"]
        }

    def get_added_vocab(self):
        return self.added.copy()

    def encode(self, text, add_special_tokens=False, **kwargs):
        return self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def decode(self, ids):
        return self.backend_tokenizer.decode(ids, skip_special_tokens=False)


CASES = [
    # (effort, patched budget, baseline budget)
    (None, 75, 75),
    ("low", 50, 50),
    ("high", 75, 75),
    ("xhigh", 75, 75),
    ("max", 85, 100),
    ("minimal", 50, 50),
    ("medium", 75, 75),
    ("ultra", 85, 100),
    (1, 1, 1),
    (42, 42, 42),
    (85, 85, 85),
    (86, 86, 86),
    (95, 95, 95),
    (100, 100, 100),
]
INVALID = ["auto", "Medium", -1, 0, 101, True, 1.5, []]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / ".work/loop-analysis/tokenizer.json")
    parser.add_argument("--baseline", action="store_true", help="expect pre-patch pass-through")
    parser.add_argument("--receipt", type=Path, default=None)
    args = parser.parse_args()
    source = args.source.resolve()

    for name in ("vllm", "vllm.utils", "vllm.tokenizers", "vllm.entrypoints"):
        stub(name)
    stub("transformers", TokenizersBackend=object)
    stub("vllm.entrypoints.chat_utils", ChatCompletionMessageParam=dict)
    stub("vllm.tokenizers.hf", HfTokenizer=object, get_cached_tokenizer=lambda x: x)
    stub("vllm.tokenizers.protocol", TokenizerLike=object)
    module("vllm.utils.tool_names", source / "vllm/utils/tool_names.py")
    module("vllm.tokenizers.deepseek_v41_encoding", source / "vllm/tokenizers/deepseek_v41_encoding.py")
    wrapper = module("vllm.tokenizers.deepseek_v41", source / "vllm/tokenizers/deepseek_v41.py")
    tok = wrapper.get_deepseek_v41_tokenizer(RustAdapter(args.tokenizer))

    messages = [{"role": "user", "content": "question"}]
    results = []
    ok = True
    for effort, patched_budget, baseline_budget in CASES:
        expected = baseline_budget if args.baseline else patched_budget
        text = tok.apply_chat_template(messages, tokenize=False, thinking=True, reasoning_effort=effort)
        prefix = "Reasoning Effort: "
        start = text.index(prefix) + len(prefix)
        rendered = int(text[start : text.index(" ", start)])
        passed = rendered == expected
        ok = ok and passed
        results.append({"effort": effort, "rendered": rendered, "expected": expected, "pass": passed})
        assert text.index("<think>") > start, "effort prefix must precede the thinking marker"
    invalid_results = []
    for effort in INVALID:
        try:
            tok.apply_chat_template(messages, tokenize=False, thinking=True, reasoning_effort=effort)
            passed = False
        except ValueError as error:
            passed = "reasoning_effort" in str(error)
        invalid_results.append({"effort": repr(effort), "rejected": passed})
        ok = ok and passed

    receipt = {
        "source": str(source),
        "mode": "baseline" if args.baseline else "ceiling",
        "cases": results,
        "invalid_still_rejected": invalid_results,
        "all_pass": ok,
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
