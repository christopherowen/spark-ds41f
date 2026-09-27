#!/usr/bin/env python3
"""Compare prompt logprobs between arms: usage compare_logprobs.py A.json B.json"""
import json
import sys

a, b = (json.load(open(path))["prompt_logprobs"][0] for path in sys.argv[1:3])
diffs = [abs(x - y) for x, y in zip(a, b) if x is not None and y is not None]
diffs.sort()
n = len(diffs)
print(f"{n} tokens: mean |diff| {sum(diffs) / n:.5f}, median {diffs[n // 2]:.5f}, "
      f"p99 {diffs[int(n * 0.99)]:.4f}, max {diffs[-1]:.4f}")
