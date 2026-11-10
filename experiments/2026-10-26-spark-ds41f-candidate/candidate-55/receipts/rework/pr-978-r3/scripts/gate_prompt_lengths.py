#!/usr/bin/env python3
"""Token counts of token_determinism_text.py's prompts as the server renders them
(DS4.1 tokenizer and chat template, thinking off), against the KV block sizes."""
import json
import os
import sys

sys.path.insert(0, "/opt/exp")
import token_determinism_text as gate  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402

tok = get_tokenizer(os.environ["DS41F_SNAPSHOT"], tokenizer_mode="deepseek_v41")
out = {}
for name, prompts in (("PROMPTS", gate.PROMPTS), ("MORE_PROMPTS", gate.MORE_PROMPTS),
                      ("long", [gate.long_prompt(700)])):
    rows = []
    for p in prompts:
        ids = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=True,
                                      add_generation_prompt=True, thinking=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        rows.append(len(ids))
    out[name] = rows
print(json.dumps(out))
