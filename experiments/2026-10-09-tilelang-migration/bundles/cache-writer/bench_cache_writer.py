"""Port B3: DeepSeek-V4.1's KV-cache record writers, B12X's write_cache (prepared as
the attention prepares it) against the TileLang writers in tilelang/cache_writer.py,
for the SWA (528-byte E4M3) and indexed (288-byte E2M1) records at the TP4 page sizes.

1. Bytes: the TileLang records equal B12X's at every row count, on activations that
   reach every scale regime (zero groups, saturating scales), with skipped slots.
2. Repeatability: graph replays, warm, L2-evicted and beside other traffic.
3. Time per call under CUDA graphs, warm and cold, at the decode capture sizes and
   prefill rows.

Exits 1 if bytes differ, a replay differs, or TileLang is more than 3% slower.
"""
import sys

import torch

sys.path.insert(0, "/b")
from kbench import (  # noqa: E402
    CAPACITY, DECODE, DEVICE, PREFILL, REPEAT, REPLAYS, per_call, repeatable, slower, timing_line,
)

from b12x.attention import compressed_sparse_mla as mla  # noqa: E402
from b12x.preparation import PreparationSession, PreparedCall  # noqa: E402

from vllm.models.deepseek_v4_1.tilelang.cache_writer import RECORD_BYTES, write_cache  # noqa: E402

PAGES = {"swa": 64, "indexed": 64}
failures = []
gen = torch.Generator(device=DEVICE).manual_seed(91)
kv = torch.randn((CAPACITY, 512), generator=gen, device=DEVICE)
kv *= torch.logspace(-4, 3, CAPACITY, device=DEVICE)[:, None]  # every scale regime
kv[::97, :64] = 0.0
kv = kv.bfloat16()

with PreparationSession(device="cuda", autotune=False, compile_workers=2) as session:
    for kind, page_size in PAGES.items():
        record = RECORD_BYTES[kind]
        pages = -(-2 * CAPACITY // page_size)
        plan = mla.plan_cache_writer(
            mla.CacheWriterQuery(max_rows=CAPACITY, page_size=page_size, cache_kind=kind, slot_dtype="int64"),
            device=DEVICE)

        def prepare(state, page_size=page_size, kind=kind):
            source = torch.ones((1, 512), dtype=torch.bfloat16, device=DEVICE)
            cache = torch.empty((1, mla.page_nbytes(page_size, cache_kind=kind, cache_format="deepseek_v41")),
                                dtype=torch.uint8, device=DEVICE)
            slots = torch.zeros(1, dtype=torch.int64, device=DEVICE)
            return PreparedCall(run=lambda: state.run(source, cache, slots))

        session.prepare((plan.request(name=f"cache.{kind}", prepare_call=prepare),))
        width = mla.page_nbytes(page_size, cache_kind=kind, cache_format="deepseek_v41")
        b_cache = torch.zeros((pages, width), dtype=torch.uint8, device=DEVICE)
        t_cache = torch.zeros_like(b_cache)
        slots = torch.randperm(pages * page_size, generator=gen, device=DEVICE)[:CAPACITY]
        slots[::13] = -1

        def b12x(rows, kind=kind, page_size=page_size, plan=plan, b_cache=b_cache, slots=slots):
            mla.write_cache(kv[:rows], b_cache, slots[:rows], page_size=page_size, cache_kind=kind,
                            cache_format="deepseek_v41", plan=plan)

        def tilelang(rows, kind=kind, page_size=page_size, t_cache=t_cache, slots=slots):
            write_cache(kv[:rows], t_cache, slots[:rows], page_size=page_size, cache_kind=kind)

        # 1. Bytes.
        differ = []
        for rows in DECODE + PREFILL:
            b_cache.zero_()
            t_cache.zero_()
            b12x(rows)
            tilelang(rows)
            torch.cuda.synchronize()
            if not torch.equal(b_cache, t_cache):
                bad = (b_cache != t_cache).view(-1, record).any(-1).sum().item()
                differ.append((rows, bad))
        if differ:
            failures.append(f"{kind}: records differ (rows, records) {differ}")
        print(f"{kind} ({record} B, page {page_size}): records differ from B12X at {differ or 'no'} sizes", flush=True)

        # 2. Repeatability.
        flaky = [rows for rows in REPEAT if repeatable(lambda rows=rows: tilelang(rows), lambda: [t_cache])]
        if flaky:
            failures.append(f"{kind}: not repeatable at {flaky}")
        print(f"{kind}: not repeatable at {flaky or 'no'} sizes ({REPLAYS} replays each)", flush=True)

        # 3. Time.
        print(f"{kind}: us per call, warm / cold", flush=True)
        for rows in DECODE + PREFILL:
            base = per_call(lambda rows=rows: b12x(rows), rows)
            port = per_call(lambda rows=rows: tilelang(rows), rows)
            print(timing_line(rows, base, port), flush=True)
            if slower(base, port):
                failures.append(f"{kind} rows {rows}: TileLang slower")

for failure in failures:
    print("FAIL", failure)
sys.exit(1 if failures else 0)
