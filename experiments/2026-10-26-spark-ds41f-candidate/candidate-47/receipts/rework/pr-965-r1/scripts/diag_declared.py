"""Names the difference between launched and declared TileLang calls in
test_declared_calls_cover_the_call_sites, and checks whether the fused staged write
of test_apply_fused_staged_writes reaches the tables."""
import importlib.util
import math
import sys

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "tli_test", "tests/v1/worker/test_gpu_tilelang_inputs.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
tli = mod.tli

names = {}
launched = []


def kname(f):
    orig = getattr(getattr(f, "func", None), "orig_func", None)
    return getattr(orig, "__name__", None) or getattr(f, "__name__", type(f).__name__)

original = tli._kernel


def spy(factory, *key):
    names[id(factory)] = kname(factory)
    launched.append((id(factory), key))
    return original(factory, *key)


tli._kernel = spy
mp = pytest.MonkeyPatch()
err = None
try:
    mod.test_declared_calls_cover_the_call_sites(mp)
    print("test_declared_calls_cover_the_call_sites: PASS")
except AssertionError as e:
    err = e
    print("test_declared_calls_cover_the_call_sites: FAIL")
finally:
    mp.undo()
    tli._kernel = original

DRAFTS = mod.DRAFTS
declared = (
    tli.block_table_calls(cp_size=1, cp_interleave=1)
    + tli.staged_write_calls()
    + tli.zero_kv_blocks_calls()
    + tli.input_batch_calls(
        num_lookahead=1,
        num_speculative_steps=DRAFTS,
        num_new_sampled_tokens=1,
        dead_rows=True,
        sampled_widths=(1, 1 + DRAFTS),
        commits=((True, True, False),),
    )
    + tli.dflash_calls(False, [(1, 1)])
)
for f, _, _ in declared:
    names.setdefault(id(f), kname(f))
declared_keys = {(id(f), a) for f, a, _ in declared}
# The Engram lookback part of the test does not go through tli._kernel.
launched_keys = set(launched)
print("launched:", len(launched_keys), "declared:", len(declared_keys))
print("launched - declared (undeclared launches, would JIT after freeze):")
for i, k in sorted((names[i], k) for i, k in launched_keys - declared_keys):
    print("   ", i, k)
print("declared - launched (declared, not exercised by the call sites):")
for i, k in sorted((names[i], k) for i, k in declared_keys - launched_keys):
    print("   ", i, k)

# test_apply_fused_staged_writes, instrumented.
gen = mod._gen(11)
tables = mod._block_tables(gen)
for t in tables.block_tables:
    mod._stage(t, gen, 9, long=True)
pending = sum(len(t._staged_write_indices) for t in tables.block_tables)
before = [t.gpu.clone() for t in tables.block_tables]
print(f"fused: staged writes pending before apply: {pending}, _dirty={tables._dirty}")
tables.apply_staged_writes()
torch.accelerator.synchronize()
changed = sum(int(not torch.equal(b, t.gpu)) for b, t in zip(before, tables.block_tables))
left = sum(len(t._staged_write_indices) for t in tables.block_tables)
print(f"fused: tables changed by apply: {changed}/{len(before)}, staged writes left: {left}")
sys.exit(0)
