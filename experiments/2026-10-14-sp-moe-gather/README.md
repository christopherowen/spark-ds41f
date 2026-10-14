# Sequence-parallel MoE input: route and cast each rank's rows, gather MXFP8

Owner, 2026-10-14: set up as an experiment, one piece per branch.

## Why

Under prefill sequence parallelism every encoder layer (0-19, before the CED
boundary) all-gathers the FFN input as BF16 rows, T x 5120, and then every rank
repeats row-wise work on all T rows:

- it casts the rows to MXFP8 twice, with the same
  `per_token_cast(x, "e4m3", 32, round_sf=True, use_packed_ue8m0=True)`: once
  for the shared expert's gate/up (`_block32_linear` in
  `vllm/models/deepseek_v4_1/tilelang/linear.py`) and once inside the routed
  experts (`TileLangExperts.run` in `vllm/models/deepseek_v4_1/tilelang/moe.py`);
- it runs the router GEMM (BF16, 384 x 5120, replicated) and the top-k.

Each rank can do this work on its own L = T / 4 rows before the gather, and
gather the results: the MXFP8 row, its packed UE8M0 scale words and the six
expert ids and weights. That is 5,328 bytes a row in place of 10,240.

## Ceiling, from existing traces

Source: the 2026-10-13 fixed-row profile
([tilelang-wo](../2026-10-13-tilelang-wo/README.md), run `fused-fixed`, arm `fused`).
That arm is the r6j image with the fused WO cast that r6k ships; r6k changes
neither the MoE nor the sequence-parallel gathers. The profile is one cold
16,258-token prefill: two 8,096-token chunks and a 66-token tail, on every rank.
In each chunk the 20 encoder layers run sequence-parallel on 8,096 rows, and the
20 decoder layers run on the 128 rows CED keeps. That makes 40 sequence-parallel
MoE layers in the window.

[sp_moe_front.py](sp_moe_front.py) reads each sequence-parallel layer's MoE front
from a rank's trace (`results/private/determinism/fused-fixed/fused/prefill-trace/<node>/`
on dgx1). Medians per layer, rank 0; ranks 1-3 are within 2%:

| Main-stream work | Now | After | Saved |
| --- | ---: | ---: | ---: |
| MoE input all-gather (NCCL ring, 20.7 MB shard) | 2,741 µs | ~1,426 µs (52% of the bytes) | ~1,315 µs |
| MXFP8 casts of the gathered rows (two) | 1,027 µs | 130 µs (one cast of L rows, as the attention front's cast measures) | ~897 µs |
| Router GEMM | 610 µs | ~153 µs (a quarter of the rows) | ~457 µs |
| Top-k | 56 µs | ~14 µs | ~42 µs |
| **Total** | **4.43 ms** | **~1.72 ms** | **~2.7 ms** |

Bound: 40 layers x ~2.7 ms is about 108 ms of the 2,808 ms window, **about 3.9%
of a 16K prefill**. At 1M tokens a chunk takes about 1.84 s instead of 1.40 s,
because attention and the indexer grow with context, so the share falls to about
3%. The all-gather saving assumes the ring all-gather's time scales with its
bytes. It runs on the main stream with nothing overlapping it. The router figure
assumes its GEMM scales with rows. Decode never runs sequence-parallel and is
untouched.

## The change (planned: vLLM branch `r6k-sp-moe-gather`, on r6k's tree)

A module commit with its switch off, then a switch commit:

- **Module.**
  - Under sequence parallelism, `DeepseekV4MoE` runs the router GEMM, the top-k
    and one `per_token_cast` on the rank's L rows. It writes each row into a
    packed record (the MXFP8 row, then the scale words, the top-k ids, the
    top-k weights; 16-byte aligned) and all-gathers the records in one call.
    The decoder layer stops gathering the BF16 rows (`self.ffn(sp_rows.gather(x), ...)`
    in `nvidia/model.py`).
  - `_block32_linear` gets an entry that takes already-cast activations into the
    same `_block32_gemm`. The shared expert's gate/up uses it.
  - `TileLangExperts.run` takes already-cast tokens and skips its own cast. The
    dispatch reads the gathered ids and weights.
  - If a GEMM's token gather cannot take the record's row stride, use one
    grouped NCCL all-gather of three contiguous buffers instead. The kernel
    tests decide which.
- **Switch.** `sp_prefill.MOE_GATHER_MXFP8` on under the TileLang family.

Not changed: the attention front's gathers (compressor layers need BF16 rows),
the reduce-scatters, the decoder layers after the CED boundary, decode.

## Why the bits do not change

- `per_token_cast` scales each 32-column block of a row from that row alone. The
  cast of L rows has the bytes of those rows in a cast of T rows.
- The router GEMM and the top-k pass the migration's row-count gate: a row's
  output equals that row of the full batch. The bundle repeats it at L and T rows.
- The records arrive in rank order, as the BF16 rows do. The slice to T rows
  drops the same pad rows.
- The shared and routed experts then read the MXFP8 bytes they cast today.

## Tests (kernel bundle, planned)

- Cast: the gathered L-row casts equal the T-row cast bit for bit, at 205-8,192
  rows and 2-4 simulated ranks, pad rows included.
- Router and top-k: ids and weights from L rows equal those rows of the T-row
  call.
- MoE layer: with the switch on, the unreduced parts (shared + routed) equal
  those with the switch off, bit for bit, at 205, 1,024, 2,048 and 8,096 rows.
- Mixed steps: decode rows inside a sequence-parallel step keep their bits.

## Arms and windows (planned)

- `gather.json`: the changed files mounted over the r6k image (`make_arms.py`
  from `config/cluster-tp4.json`). The control is r6k.
- Window 1: sync; the kernel bundle on dgx1; a lean screen, candidate first,
  bracketed by r6k, with `token_determinism.py --wide`, distinct prompts and
  mixed traffic. Then a `profile` job (workload `prefill`) of both arms, read
  with `sp_moe_front.py`, to check the front shrinks as bounded above.

## Acceptance

- The temperature-0 references are r6k's, bit for bit. The long prompt prefills
  sequence-parallel through the new path.
- Decode is level.
- Prefill at 16K and 32K is faster than r6k by more than the bracket's spread,
  and the profile shows the gather, the casts and the router shrinking as
  bounded.
- TP3 boots the same path at the next re-cable (three ranks, L = T / 3).

## Status

Planned 2026-10-14. The ceiling is measured from existing traces. Nothing is
built or run.

## Review (2026-10-14, TileLang session)

The plan holds: 5,328 bytes a row (5,120 MXFP8 + 160 packed scale bytes + 24 id bytes
+ 24 weight bytes) against 10,240, and each row's cast, router and top-k depend only on
that row. Two corrections before building:

- **Token ids.** Under sequence parallelism the FFN receives the whole step's `input_ids`
  with the gathered rows (`self.ffn(sp_rows.gather(x), input_ids, ...)` in
  `nvidia/model.py`). `DeepseekV4MoE` reads them in two places:
  - the first `config.num_hash_layers` layers route by token id
    (`gate.tid2eid`, no router GEMM);
  - vision routing (`gate.bias_vl`) uses them to find image tokens.

  Routing a rank's L rows needs that rank's slice of `input_ids`. The bundle's
  router/top-k test must cover a hash layer and a step with image tokens.
- **Ceiling.** The hash layers run no router GEMM today, so they save only the gather
  and the casts there. Recompute the router saving over the non-hash encoder layers
  with `sp_moe_front.py` (it will shave a little off the ~3.9%).
